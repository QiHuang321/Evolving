"""Best-accuracy v2: back-port all post-processing fixes + verifiers onto
evolved_instruct 0.6094 base.

This file starts from the evolved_instruct.py (0.6094) code — the LATEST evolved
output — and applies every post-processing improvement from best_accuracy.py:
  1. Image preprocessing (adaptive resize, autocontrast, sharpening)
  2. Safer candidate stripping (preserves leading negative signs)
  3. Legend count dedup from label lists
  4. Positional tick text fallback
  5. Trend synonym normalization (increasing→increases)
  6. Caret scientific notation preservation
  7. Panel marker detection for title verifier
  8. 3-view title verifier (title_sharp, title_gray variants)
  9. Colorbar existence verifier (single-probe)

The base evolved code (prompts, NA synonyms, answer-is extraction, gating)
is preserved from evolved_instruct.py since it reached 0.6094 through evolution.
"""
# EVOLVE-BLOCK-START

import re

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_MODEL = None
_PROCESSOR = None
_RAW_IMAGE_CACHE = {}
_IMAGE_CACHE = {}
_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_MAX_NEW_TOKENS = 32

_NUMBER_RE = re.compile(r"[-+]?(?:[0-9]*[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?%?")
_LAYOUT_RE = re.compile(r"([0-9]+) *(?:by|x) *([0-9]+)", re.IGNORECASE)
_PANEL_RE = re.compile(r"^\(?[a-zA-Z]\)?$")

# Evolved NA synonym list (broader than v1, discovered by evolution)
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
    "cannot be determined",
    "not a line plot",
    "no lines",
    "no continuous legend",
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


# ---------- Image preprocessing (from best_accuracy.py) ----------

def _load_raw_image(image_path):
    if image_path not in _RAW_IMAGE_CACHE:
        _RAW_IMAGE_CACHE[image_path] = Image.open(image_path).convert("RGB")
    return _RAW_IMAGE_CACHE[image_path]


def _resize_chart(img, small_target=1600, medium_target=1400, large_cap=2200):
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 700:
        scale = small_target / max_dim
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    if max_dim < 960:
        scale = medium_target / max_dim
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    if max_dim > 2600:
        scale = large_cap / max_dim
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img.copy()


def _make_view(image_path, variant="default"):
    base = _load_raw_image(image_path)

    if variant == "default":
        img = _resize_chart(base, small_target=1600, medium_target=1400, large_cap=2200)
        img = ImageOps.autocontrast(img, cutoff=0.5)
        img = ImageEnhance.Color(img).enhance(0.98)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=170, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.08)
        return img

    if variant == "title_sharp":
        img = _resize_chart(base, small_target=1800, medium_target=1600, large_cap=2400)
        img = ImageOps.autocontrast(img, cutoff=0.3)
        img = ImageEnhance.Color(img).enhance(0.97)
        img = ImageEnhance.Contrast(img).enhance(1.22)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.14)
        return img

    if variant == "title_gray":
        img = _resize_chart(base, small_target=1850, medium_target=1650, large_cap=2400)
        img = ImageOps.grayscale(img).convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=0.2)
        img = ImageEnhance.Contrast(img).enhance(1.28)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=190, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.12)
        return img

    raise ValueError(f"Unknown view variant: {variant}")


def _get_image_inputs(image_path, variant="default"):
    key = (image_path, variant)
    if key not in _IMAGE_CACHE:
        img = _make_view(image_path, variant)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        image_inputs, _ = process_vision_info(messages)
        _IMAGE_CACHE[key] = (image_inputs, img)
    return _IMAGE_CACHE[key]


# ---------- Prompt (evolved version from 0.6094) ----------

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
    # Extra hints from best_accuracy for axis/legend/title
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
    lines.append(f"Question: {question.strip()}")
    return chr(10).join(lines)


def _build_title_probe_prompt(question):
    lines = [
        "Use ONLY visible chart evidence.",
        "Return ONLY the final answer.",
        "Single line only.",
        "If the answer is not explicitly visible, answer exactly: Not Applicable.",
        "Copy the exact plot title text only. If no explicit title exists, answer Not Applicable.",
        "If the title is only a letter or panel marker such as (a), answer exactly: Not Applicable.",
        f"Question: {question.strip()}",
    ]
    return chr(10).join(lines)


# ---------- Output normalization ----------

def _dedupe(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _candidates(raw_text):
    """Extract answer candidates from raw model output.
    Combines evolved 'answer is' extraction with safe negative-sign stripping."""
    text = raw_text.split("</think>")[-1].replace("<think>", " ")
    text = text.replace("**", " ").replace("`", " ")
    # Safe stripping: remove bullet markers but preserve leading negative signs
    stripped = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        while s and s[0] in (' ', ':'):
            s = s[1:]
        if s.startswith('- ') and not (len(s) > 2 and s[1] == ' ' and s[2:3].isdigit()):
            s = s[2:]
        stripped.append(s)
    candidates = list(reversed(stripped))
    if ":" in text:
        candidates.append(text.split(":")[-1].strip())
    # Evolved: "answer is" extraction (discovered by evolution)
    extras = []
    for match in re.finditer(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", text, re.IGNORECASE):
        frag = re.split(r"[.;]|\bexplanation\b", match.group(1).strip(), 1, flags=re.IGNORECASE)[0]
        if frag:
            extras.append(frag.strip(" -:"))
    candidates = extras + candidates
    candidates.append(text.strip())
    return _dedupe([" ".join(candidate.split()) for candidate in candidates if candidate.strip()])


def _is_not_applicable(candidate, question):
    """Evolved NA detection (broader synonym list discovered by evolution)."""
    lower = candidate.lower().strip()
    if not lower:
        return True
    if lower in {"n/a", "na", "none", "null"}:
        return True
    if "not applicable" in lower:
        return True
    if any(hint in lower for hint in _NA_HINTS):
        return True
    q = " ".join(question.lower().split()) if question else ""
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
        if _is_not_applicable(candidate, question):
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
        # Trend synonym normalization (from best_accuracy)
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
            if _is_not_applicable(candidate, question):
                return "Not Applicable"
        return "1 by 1"

    if "do any lines intersect" in q:
        for candidate in candidates:
            if _is_not_applicable(candidate, question):
                return "Not Applicable"
            lower = candidate.lower()
            words = [word for word in re.split(r"[^a-z]+", lower) if word]
            if "yes" in words:
                return "Yes"
            if "no" in words:
                return "No"
        return "Not Applicable"

    # Legend count dedup (from best_accuracy): if model outputs label list, count it
    if "how many discrete labels are there in the legend" in q:
        for candidate in candidates:
            if _is_not_applicable(candidate, question):
                return "Not Applicable"
            if ',' in candidate:
                parts = [p.strip() for p in candidate.split(',') if p.strip()]
                if parts:
                    return str(len(set(parts)))
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        # Positional tick text handling (from best_accuracy)
        is_positional_tick = any(p in q for p in [
            "leftmost labeled tick", "rightmost labeled tick",
            "spatially lowest labeled tick", "spatially highest labeled tick",
        ])
        if is_positional_tick and candidates:
            first = candidates[0] if len(candidates) == 1 else candidates[-1]
            raw_clean = first.strip()
            # Preserve caret scientific notation like 10^-6
            if raw_clean and '^' in raw_clean and re.fullmatch(r'[-+]?[0-9]+\^[-+]?[0-9]+', raw_clean):
                return raw_clean
            # If it has letters and isn't pure NA, it's likely a text tick label
            if raw_clean and any(c.isalpha() for c in raw_clean) and not _is_not_applicable(raw_clean, question):
                has_pure_number = bool(re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?%?', raw_clean))
                if not has_pure_number:
                    return _normalize_text_answer(question, candidates)
        for candidate in candidates:
            if _is_not_applicable(candidate, question):
                return "Not Applicable"
            if "legend" in q and ("no legend" in candidate.lower() or "no colorbar" in candidate.lower() or "no continuous legend" in candidate.lower()):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        if is_positional_tick:
            return _normalize_text_answer(question, candidates)
        return "Not Applicable"

    return _normalize_text_answer(question, candidates)


# ---------- Inference helpers ----------

def _run_single(model, processor, image_path, question, prompt, variant="default"):
    image_inputs, img = _get_image_inputs(image_path, variant)
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
    return _normalize_answer(question, raw_text)


# ---------- Verifiers (from best_accuracy.py) ----------

def _is_title_question(question):
    return "what is its title" in " ".join(question.lower().split())


def _is_continuous_legend_question(question):
    q = " ".join(question.lower().split())
    return "continuous legend" in q


def _looks_like_panel_marker(answer):
    cleaned = answer.strip()
    if cleaned == "Not Applicable":
        return False
    if _PANEL_RE.fullmatch(cleaned):
        return True
    if len(cleaned) <= 3 and cleaned.replace(".", "").replace("-", "").isalpha():
        return True
    return False


def _continuous_legend_verifier(model, processor, image_path):
    """Probe whether the chart has a continuous legend (colorbar)."""
    probe_prompt = chr(10).join([
        "Look at this chart carefully.",
        "Does this chart contain a continuous legend (also called a colorbar)?",
        "A colorbar is a gradient bar mapping colors to numeric values.",
        "A discrete legend listing category names is NOT a colorbar.",
        "Answer exactly: Yes or No.",
    ])
    probe_answer = _run_single(model, processor, image_path, "", probe_prompt, variant="default")
    lower = probe_answer.strip().lower()
    if "yes" in lower and "no" not in lower:
        return True
    if "no" in lower:
        return False
    return None


def _title_verifier(model, processor, image_path, question):
    """Multi-view title verifier: 3 views must agree to override."""
    prompt = _build_title_probe_prompt(question)
    answers = []
    for variant in ["default", "title_sharp", "title_gray"]:
        answers.append(_run_single(model, processor, image_path, question, prompt, variant=variant))
    if answers[0] == answers[1] == answers[2]:
        return answers[0]
    return None


# ---------- Main entry point ----------

def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)
    base_answer = _run_single(model, processor, image_path, question, prompt, variant="default")

    # Colorbar verifier: if model returned a value but there's no colorbar
    if _is_continuous_legend_question(question) and base_answer != "Not Applicable":
        has_colorbar = _continuous_legend_verifier(model, processor, image_path)
        if has_colorbar is False:
            return "Not Applicable"

    # Title verifier: if answer looks like a panel marker, verify with 3 views
    if _is_title_question(question) and _looks_like_panel_marker(base_answer):
        verified = _title_verifier(model, processor, image_path, question)
        if verified is not None and verified != base_answer:
            return verified

    return base_answer
# EVOLVE-BLOCK-END
