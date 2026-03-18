# EVOLVE-BLOCK-START
"""Best-overall CharXiv inference: evolved_instruct + post-processing bug fixes."""

import re

import torch
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
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

_NA_PHRASES = [
    "not applicable",
    "not available",
    "not provided",
    "not shown",
    "not visible",
    "not specified",
    "not given",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "unclear",
    "not clear",
    "unknown",
    "none",
    "no legend",
    "no colorbar",
    "no continuous legend",
    "no data",
    "not present",
    "does not exist",
    "not a line plot",
    "no lines",
    "not sure",
    "unsure",
    "cannot tell",
    "can't tell",
    "hard to tell",
    "insufficient information",
    "insufficient data",
    "no information",
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


def _preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    max_dim = max(w, h)

    # Upscale small charts to improve text readability; cap very large charts.
    if max_dim < 700:
        scale = 1600 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_dim < 960:
        scale = 1400 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_dim > 2600:
        scale = 2200 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # Enhance contrast/sharpness for crisp chart text and lines.
    img = ImageOps.autocontrast(img, cutoff=0.5)
    img = ImageEnhance.Color(img).enhance(0.98)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=170, threshold=2))
    img = ImageEnhance.Sharpness(img).enhance(1.08)
    return img


def _get_image_inputs(image_path):
    if image_path not in _IMAGE_CACHE:
        img = _preprocess_image(image_path)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        image_inputs, _ = process_vision_info(messages)
        _IMAGE_CACHE[image_path] = (image_inputs, img)
    return _IMAGE_CACHE[image_path]


def _build_prompt(question):
    q = " ".join(question.lower().split())
    lines = [
        "You are a strict chart QA assistant.",
        "Your answer must be a SINGLE line with ONLY the final answer.",
        "Do NOT explain, do NOT repeat the question, do NOT add extra words.",
        "If the answer is absent or not explicitly supported by the chart, answer exactly: Not Applicable.",
        "Use chart text exactly when possible. Preserve original capitalization and spacing.",
        "Never add trailing punctuation or quotes.",
        "Strict formats:",
        "- numbers/counts/ticks: digits only (no commas, no units, no % sign, no words)",
        "- yes/no questions: Yes, No, or Not Applicable",
        "- subplot layout: n by m (rows by columns)",
        "- legend labels: label1, label2, label3 (comma + space, no 'and')",
        "- short text (titles/axis labels): exact phrase only (no leading 'the', 'a', 'an' unless shown)",
        "Do not round, estimate, or add approximations.",
    ]

    # Targeted guidance for specific question types.
    if "label of the x-axis" in q or "x-axis label" in q:
        lines.append("For x-axis label questions, answer ONLY the exact axis title text. No extra words.")
        lines.append("If no x-axis title is shown, answer exactly: Not Applicable.")
    if "label of the y-axis" in q or "y-axis label" in q:
        lines.append("For y-axis label questions, answer ONLY the exact axis title text. No extra words.")
        lines.append("If no y-axis title is shown, answer exactly: Not Applicable.")
    if "names of the labels in the legend" in q or "labels in the legend" in q:
        lines.append("For legend label questions, list labels exactly as shown, in order, separated by comma + space.")
        lines.append("Do not add 'legend', 'labels', or 'and'.")
    if "general trend of data from left to right" in q:
        lines.append("Answer with a very short phrase only, e.g., increasing, decreasing, stable, or no clear trend.")
    if "layout of the subplots" in q:
        lines.append("Layout means rows by columns. Two side-by-side subplots are 1 by 2. Two stacked subplots are 2 by 1.")
        lines.append("If only one subplot, answer 1 by 1.")
    if "continuous legend" in q:
        lines.append("Use only a continuous legend or colorbar. If none exists, answer exactly: Not Applicable.")
    if "legend" in q and "continuous legend" not in q:
        lines.append("Use only the relevant discrete legend. If there is no legend for this plot, answer exactly: Not Applicable.")
        lines.append("If listing legend labels, keep the order shown in the legend (top-to-bottom or left-to-right).")
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
    # Strip list markers but preserve leading negative signs
    stripped = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Remove leading bullet markers like "- " or ": " but keep "-1.00"
        while s and s[0] in (' ', ':'):
            s = s[1:]
        if s.startswith('- ') and not (len(s) > 2 and s[1] == ' ' and s[2:3].isdigit()):
            s = s[2:]
        stripped.append(s)
    candidates = list(reversed(stripped))
    if ":" in text:
        candidates.append(text.split(":")[-1].strip())
    candidates.append(text.strip())
    return _dedupe([" ".join(candidate.split()) for candidate in candidates if candidate.strip()])


def _looks_not_applicable(text):
    lower = text.lower().strip()
    if lower in {"n/a", "na", "none", "null"}:
        return True
    return any(phrase in lower for phrase in _NA_PHRASES)



def _normalize_text_answer(question, candidates):
    q = " ".join(question.lower().split())
    for candidate in candidates:
        if _looks_not_applicable(candidate):
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
        if "general trend of data from left to right" in q:
            trend_map = {
                "increasing": "increases",
                "decreasing": "decreases",
            }
            value = trend_map.get(value.lower(), value)
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
            if _looks_not_applicable(candidate):
                return "Not Applicable"
        return "1 by 1"

    if "do any lines intersect" in q:
        for candidate in candidates:
            if _looks_not_applicable(candidate):
                return "Not Applicable"
            words = [word for word in re.split(r"[^a-z]+", candidate.lower()) if word]
            if "yes" in words:
                return "Yes"
            if "no" in words:
                return "No"
        return "Not Applicable"

    # For legend count questions, if model outputs a comma-separated label list
    # instead of a count, deduplicate and return the count.
    if "how many discrete labels are there in the legend" in q:
        for candidate in candidates:
            if _looks_not_applicable(candidate):
                return "Not Applicable"
            # If candidate contains commas, it's a label list; count unique labels.
            if ',' in candidate:
                parts = [p.strip() for p in candidate.split(',') if p.strip()]
                if parts:
                    return str(len(set(parts)))
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        # For tick value questions (leftmost/rightmost/lowest/highest), ticks can
        # be text strings (e.g. "SSM", "Heter-unaware"), not only numbers.
        is_positional_tick = any(p in q for p in [
            "leftmost labeled tick", "rightmost labeled tick",
            "spatially lowest labeled tick", "spatially highest labeled tick",
        ])
        # For positional ticks, if the raw answer looks like a text string
        # (contains letters and is not just a number), treat as text answer.
        if is_positional_tick and candidates:
            first = candidates[0] if len(candidates) == 1 else candidates[-1]  # last = original raw
            raw_clean = first.strip()
            # Preserve caret scientific notation like 10^-6, 10^1
            if raw_clean and '^' in raw_clean and re.fullmatch(r'[-+]?[0-9]+\^[-+]?[0-9]+', raw_clean):
                return raw_clean
            # If it has letters and isn't pure NA, it's likely a text tick label
            if raw_clean and any(c.isalpha() for c in raw_clean) and not _looks_not_applicable(raw_clean):
                has_pure_number = bool(re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?%?', raw_clean))
                if not has_pure_number:
                    return _normalize_text_answer(question, candidates)
        for candidate in candidates:
            if _looks_not_applicable(candidate):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        # If no number found but this is a positional tick question,
        # fall through to text normalization (tick might be a word).
        if is_positional_tick:
            return _normalize_text_answer(question, candidates)
        return "Not Applicable"

    return _normalize_text_answer(question, candidates)


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)

    image_inputs, img = _get_image_inputs(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    # Dynamic token budget
    q_lower = " ".join(question.lower().split())
    max_tokens = _MAX_NEW_TOKENS
    if "legend" in q_lower or "title" in q_lower or "label of the" in q_lower:
        max_tokens = max(_MAX_NEW_TOKENS, 64)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    base_answer = _normalize_answer(question, raw_text)

    # Continuous legend verifier
    if "continuous legend" in q_lower and base_answer != "Not Applicable":
        probe_prompt = chr(10).join([
            "Look at this chart carefully.",
            "Does this chart contain a continuous legend (also called a colorbar)?",
            "A colorbar is a gradient bar mapping colors to numeric values.",
            "A discrete legend listing category names is NOT a colorbar.",
            "Answer exactly: Yes or No.",
        ])
        probe_msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": probe_prompt}]}]
        probe_text = processor.apply_chat_template(probe_msgs, tokenize=False, add_generation_prompt=True)
        probe_inputs = processor(text=[probe_text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            probe_ids = model.generate(**probe_inputs, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=pad_token_id)
        probe_trimmed = [o[len(i):] for i, o in zip(probe_inputs.input_ids, probe_ids)]
        probe_answer = processor.batch_decode(probe_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()
        if "no" in probe_answer:
            return "Not Applicable"

    return base_answer
# EVOLVE-BLOCK-END
