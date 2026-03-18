# EVOLVE-BLOCK-START
"""Best evolved program (Instruct): generation 40 output from evolve_instruct.py.

Accuracy: 0.6094 (78/128) | Avg time: 0.261 s/query
"""

import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_MODEL = None
_PROCESSOR = None
_IMAGE_CACHE = {}
_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_MAX_NEW_TOKENS = 32

_NUMBER_RE = re.compile(r"[-+]?(?:[0-9]*[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?%?")
_LAYOUT_RE = re.compile(r"([0-9]+) *(?:by|x) *([0-9]+)", re.IGNORECASE)

_NUMERIC_PHRASES = [
    "difference between consecutive numerical tick values",
    "spatially highest labeled tick",
    "spatially lowest labeled tick",
    "rightmost labeled tick",
    "leftmost labeled tick",
    "how many lines are there",
    "how many discrete labels are there in the legend",
    "maximum value of the tick labels on the continuous legend",
    "difference between the maximum and minimum values of the tick labels on the continuous legend",
    "total number of explicitly labeled ticks across all axes",
    "number of subplots",
]

_NA_HINTS = [
    "not applicable",
    "not available",
    "not provided",
    "not shown",
    "not visible",
    "not specified",
    "not given",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "unknown",
    "no legend",
    "no colorbar",
    "no title",
    "no label",
    "no axis label",
    "no axes",
    "no data",
    "not present",
    "does not exist",
    "no information",
    "insufficient information",
    "insufficient data",
    "not enough information",
    "not enough data",
    "unclear",
    "not clear",
    "not sure",
    "unsure",
    "cannot tell",
    "can't tell",
    "hard to tell",
]


def _get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        _MODEL = AutoModelForImageTextToText.from_pretrained(
            _MODEL_NAME,
            dtype=dtype,
            device_map="auto",
        )
        _MODEL.eval()
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_NAME, use_fast=False)
    return _MODEL, _PROCESSOR


def _get_image_inputs(image_path):
    if image_path not in _IMAGE_CACHE:
        from PIL import Image
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        image_inputs, _ = process_vision_info(messages)
        _IMAGE_CACHE[image_path] = image_inputs
    return _IMAGE_CACHE[image_path]


def _build_prompt(question):
    q = " ".join(question.lower().split())
    lines = [
        "Answer the chart question with ONLY the final answer.",
        "Return a single line with no extra punctuation or quotes.",
        "Do not explain your reasoning.",
        "Do not repeat the question.",
        "If the answer is absent, unclear, or not explicitly supported by the chart, answer exactly: Not Applicable.",
        "If you are not completely sure, answer exactly: Not Applicable.",
        "Use the chart text exactly when possible. Do not add extra words.",
        "Formats:",
        "- numbers/counts/ticks: only the value",
        "- yes/no questions: Yes, No, or Not Applicable",
        "- subplot layout: n by m",
        "- legend names: label1, label2, label3 (comma-separated, no 'and')",
        "- titles/labels: output the exact text only",
        "- trend: short phrase only (e.g., increasing, decreasing, fluctuating, no clear trend)",
    ]
    if any(phrase in q for phrase in _NUMERIC_PHRASES) or "tick" in q or "how many" in q or "value" in q:
        lines.append("For numeric answers, output a single number only (no units, words, ranges, or extra numbers).")
    if "continuous legend" in q:
        lines.append("Use only a continuous legend or colorbar. If none exists, answer exactly: Not Applicable.")
    if "legend" in q and "continuous legend" not in q:
        lines.append("Use only the relevant discrete legend. If there is no legend for this plot, answer exactly: Not Applicable.")
    if "layout of the subplots" in q:
        lines.append("Layout means rows by columns. Two side-by-side subplots are 1 by 2. Two stacked subplots are 2 by 1.")
    if "do any lines intersect" in q:
        lines.append("If there are no plotted data lines, answer exactly: Not Applicable.")
    lines.append(f"Question: {question.strip()}")
    return chr(10).join(lines)


def _dedupe(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _candidates(raw_text):
    text = raw_text.split("</think>")[-1].replace("<think>", " ")
    text = text.replace("**", " ").replace("`", " ")
    lines = [line.strip(" -:") for line in text.splitlines() if line.strip()]
    candidates = list(reversed(lines))
    if ":" in text:
        candidates.append(text.split(":")[-1].strip())
    extras = []
    for match in re.finditer(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", text, re.IGNORECASE):
        frag = re.split(r"[.;]|\bexplanation\b", match.group(1).strip(), 1, flags=re.IGNORECASE)[0]
        if frag:
            extras.append(frag.strip(" -:"))
    candidates = extras + candidates
    candidates.append(text.strip())
    return _dedupe([" ".join(candidate.split()) for candidate in candidates if candidate.strip()])


def _is_not_applicable(candidate, question):
    lower = candidate.lower().strip()
    if not lower:
        return True
    if lower in {"n/a", "na", "none"}:
        return True
    if "not applicable" in lower:
        return True
    if any(hint in lower for hint in _NA_HINTS):
        return True
    q = " ".join(question.lower().split())
    if "legend" in q and lower in {"no", "none"}:
        return True
    if ("label" in q or "title" in q) and lower in {"no", "none"}:
        return True
    if "intersect" in q and "line" in q:
        if re.search(r"\bno\s+lines?\s+(are\s+)?(shown|present|plotted|visible|available|drawn|displayed)\b", lower):
            return True
        if "not a line plot" in lower or "not a line graph" in lower or "no line plot" in lower or "no line graph" in lower:
            return True
    return False


def _normalize_text_answer(question, candidates):
    q = " ".join(question.lower().split())
    for candidate in candidates:
        if _is_not_applicable(candidate, q):
            return "Not Applicable"
        value = candidate.strip().strip('"').strip("'")
        if "names of the labels in the legend" in q:
            prefixes = [
                "the names of the labels in the legend are ",
                "the labels in the legend are ",
                "the labels are ",
                "labels are ",
            ]
            lower_value = value.lower()
            for prefix in prefixes:
                if lower_value.startswith(prefix):
                    value = value[len(prefix):]
                    break
            value = ", ".join(part.strip() for part in value.split(",") if part.strip())
        elif "label of the x-axis" in q:
            for prefix in ["the x-axis label is ", "x-axis label is ", "label of the x-axis is "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "label of the y-axis" in q:
            for prefix in ["the y-axis label is ", "y-axis label is ", "label of the y-axis is "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "what is its title" in q:
            for prefix in ["the title is ", "title is ", "the plot title is "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "general trend of data from left to right" in q:
            for prefix in ["the general trend is ", "general trend is "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        value = value.strip().strip('"').strip("'").rstrip(" .;:")
        if value:
            return value
    return "Not Applicable"


def _normalize_answer(question, raw_text):
    q = " ".join(question.lower().split())
    candidates = _candidates(raw_text)
    if not candidates:
        return "Not Applicable"

    if "layout of the subplots" in q:
        for candidate in candidates:
            if _is_not_applicable(candidate, q):
                return "Not Applicable"
            match = _LAYOUT_RE.search(candidate)
            if match:
                return f"{match.group(1)} by {match.group(2)}"
        return "1 by 1"

    if "do any lines intersect" in q:
        for candidate in candidates:
            if _is_not_applicable(candidate, q):
                return "Not Applicable"
            lower = candidate.lower()
            words = [word for word in re.split(r"[^a-z]+", lower) if word]
            if "yes" in words:
                return "Yes"
            if "no" in words:
                return "No"
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        for candidate in candidates:
            if _is_not_applicable(candidate, q):
                return "Not Applicable"
            if "legend" in q and ("no legend" in candidate.lower() or "no colorbar" in candidate.lower() or "no continuous legend" in candidate.lower()):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        return "Not Applicable"

    return _normalize_text_answer(question, candidates)


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = _get_image_inputs(image_path)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    q = " ".join(question.lower().split())
    max_new_tokens = _MAX_NEW_TOKENS
    if "legend" in q or "title" in q or "label of the" in q:
        max_new_tokens = max(_MAX_NEW_TOKENS, 64)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _normalize_answer(question, raw_text)
# EVOLVE-BLOCK-END
