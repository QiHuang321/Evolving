# EVOLVE-BLOCK-START
"""Best evolved program (Thinking): output from evolve_thinking.py (hit API quota at gen 15).

Accuracy: 0.5547 (71/128) | Avg time: 0.296 s/query
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
_MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"
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
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}]}]
        image_inputs, _ = process_vision_info(messages)
        _IMAGE_CACHE[image_path] = image_inputs
    return _IMAGE_CACHE[image_path]


def _build_prompt(question):
    q = " ".join(question.lower().split())
    lines = [
        "You are answering a chart question.",
        "Return ONLY the final answer on a single line.",
        "Do NOT explain reasoning or repeat the question.",
        "Do NOT output bullet points, prefixes, or extra words.",
        "If the answer is absent or not explicitly supported by the chart, answer exactly: Not Applicable.",
        "Use the chart text exactly when possible.",
        "Formats:",
        "- numbers/counts/ticks: only the value",
        "- yes/no questions: Yes, No, or Not Applicable",
        "- subplot layout: n by m",
        "- legend names: label1, label2, label3",
    ]
    if "continuous legend" in q:
        lines.append("Use only a continuous legend or colorbar. If none exists, answer exactly: Not Applicable.")
    if "legend" in q and "continuous legend" not in q:
        lines.append("Use only the relevant discrete legend. If there is no legend for this plot, answer exactly: Not Applicable.")
    if "layout of the subplots" in q:
        lines.append("Layout means rows by columns. Two side-by-side subplots are 1 by 2. Two stacked subplots are 2 by 1.")
    if "do any lines intersect" in q:
        lines.append("If there are no plotted data lines, answer exactly: Not Applicable.")
    lines.append("After thinking, output ONLY the answer with no tags or punctuation.")
    lines.append(f"Question: {question.strip()}")
    return chr(10).join(lines)


def _dedupe(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _extract_candidate_blocks(raw_text):
    blocks = []
    if "</think>" in raw_text:
        after = raw_text.split("</think>")[-1]
        if after.strip():
            blocks.append(after)
        think_blocks = re.findall(r"<think>(.*?)</think>", raw_text, flags=re.S)
        if think_blocks:
            last_think = think_blocks[-1]
            if last_think.strip():
                blocks.append(last_think)
    if not blocks:
        blocks = [raw_text]
    return blocks


def _candidates(raw_text):
    candidates = []
    for text in _extract_candidate_blocks(raw_text):
        text = text.replace("<think>", " ").replace("</think>", " ")
        text = text.replace("**", " ").replace("`", " ")
        lines = [line.strip(" -:") for line in text.splitlines() if line.strip()]
        candidates.extend(reversed(lines))
        if ":" in text:
            candidates.append(text.split(":")[-1].strip())
        candidates.append(text.strip())
    return _dedupe([" ".join(candidate.split()) for candidate in candidates if candidate.strip()])


def _normalize_text_answer(question, candidates):
    q = " ".join(question.lower().split())
    first_word = q.split()[0] if q.split() else ""
    is_yesno = first_word in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "has", "have", "had"}
    for candidate in candidates:
        lower = candidate.lower()
        if "not applicable" in lower:
            return "Not Applicable"
        if is_yesno:
            m = re.search(r"\b(yes|no|true|false)\b", lower)
            if m:
                return "Yes" if m.group(1) in {"yes", "true"} else "No"
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
            if "," in value:
                parts = [part.strip() for part in value.split(",") if part.strip()]
            elif " and " in value.lower():
                parts = [part.strip() for part in re.split(r"\band\b", value, flags=re.I) if part.strip()]
            else:
                parts = [value.strip()] if value.strip() else []
            value = ", ".join(parts)
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
            match = _LAYOUT_RE.search(candidate)
            if match:
                return f"{match.group(1)} by {match.group(2)}"
        return "1 by 1"

    if "do any lines intersect" in q:
        for candidate in candidates:
            lower = candidate.lower()
            if "not applicable" in lower or "no lines" in lower or "not a line plot" in lower:
                return "Not Applicable"
            words = [word for word in re.split(r"[^a-z]+", lower) if word]
            if "yes" in words:
                return "Yes"
            if "no" in words:
                return "No"
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        for candidate in candidates:
            lower = candidate.lower()
            if "not applicable" in lower:
                return "Not Applicable"
            if "legend" in q and ("no legend" in lower or "no colorbar" in lower or "no continuous legend" in lower):
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
    text = text + "<think>" + chr(10) + "</think>" + chr(10) + chr(10)
    image_inputs = _get_image_inputs(image_path)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _normalize_answer(question, raw_text)
# EVOLVE-BLOCK-END
