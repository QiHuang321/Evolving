# EVOLVE-BLOCK-START
"""Best-accuracy CharXiv inference with a targeted title verifier."""

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
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unclear",
    "unknown",
    "none",
    "no legend",
    "no colorbar",
    "no continuous legend",
    "no data",
    "not a line plot",
    "no lines",
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

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        for candidate in candidates:
            if _looks_not_applicable(candidate):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        return "Not Applicable"

    return _normalize_text_answer(question, candidates)


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


def _is_title_question(question):
    return "what is its title" in " ".join(question.lower().split())


def _looks_like_panel_marker(answer):
    cleaned = answer.strip()
    if cleaned == "Not Applicable":
        return False
    if _PANEL_RE.fullmatch(cleaned):
        return True
    if len(cleaned) <= 3 and cleaned.replace(".", "").replace("-", "").isalpha():
        return True
    return False


def _title_verifier(model, processor, image_path, question):
    prompt = _build_title_probe_prompt(question)
    answers = []
    for variant in ["default", "title_sharp", "title_gray"]:
        answers.append(_run_single(model, processor, image_path, question, prompt, variant=variant))
    if answers[0] == answers[1] == answers[2]:
        return answers[0]
    return None


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)
    base_answer = _run_single(model, processor, image_path, question, prompt, variant="default")

    if _is_title_question(question) and _looks_like_panel_marker(base_answer):
        verified = _title_verifier(model, processor, image_path, question)
        if verified is not None and verified != base_answer:
            return verified

    return base_answer
# EVOLVE-BLOCK-END
