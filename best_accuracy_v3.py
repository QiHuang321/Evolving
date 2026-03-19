"""Best-accuracy v3: Instruct/Thinking question-type router.  **NEGATIVE RESULT.**

Hypothesis (from head-to-head error analysis on 128 samples):
- Instruct wins +24 on value-extraction (ticks, labels, intersections)
- Thinking wins +18 on absence-detection (Not Applicable)
- Router: continuous-legend questions → Thinking, everything else → Instruct

Result: 0.7500  (WORSE than best_accuracy.py = 0.7969)

Why it failed:
  The colorbar-existence verifier in best_accuracy.py already captures the
  NA-detection advantage that Thinking offered in the *unverified* comparison.
  Instruct + colorbar verifier scores 16/16 on continuous-legend questions,
  leaving no room for Thinking to help.  Routing to Thinking actually loses 6
  questions because the Thinking model's value-extraction is weaker.

Lesson: verifier-augmented prompting can absorb the accuracy gap that would
otherwise motivate a multi-model router.  This script is retained as a
documented negative result.
"""

import re
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ════════════════════════════════════════════════════════════════
#  Shared state
# ════════════════════════════════════════════════════════════════

_INSTRUCT_MODEL = None
_INSTRUCT_PROCESSOR = None
_THINKING_MODEL = None
_THINKING_PROCESSOR = None
_RAW_IMAGE_CACHE = {}
_IMAGE_CACHE = {}
_THINKING_IMAGE_CACHE = {}

_INSTRUCT_NAME = "Qwen/Qwen3-VL-2B-Instruct"
_THINKING_NAME = "Qwen/Qwen3-VL-2B-Thinking"
_MAX_NEW_TOKENS = 32

_NUMBER_RE = re.compile(r"[-+]?(?:[0-9]*[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?%?")
_LAYOUT_RE = re.compile(r"([0-9]+) *(?:by|x) *([0-9]+)", re.IGNORECASE)
_PANEL_RE = re.compile(r"^\(?[a-zA-Z]\)?$")


def _get_instruct_model():
    global _INSTRUCT_MODEL, _INSTRUCT_PROCESSOR
    if _INSTRUCT_MODEL is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        _INSTRUCT_MODEL = AutoModelForImageTextToText.from_pretrained(
            _INSTRUCT_NAME, dtype=dtype, device_map="auto")
        _INSTRUCT_MODEL.eval()
    if _INSTRUCT_PROCESSOR is None:
        _INSTRUCT_PROCESSOR = AutoProcessor.from_pretrained(_INSTRUCT_NAME, use_fast=False)
    return _INSTRUCT_MODEL, _INSTRUCT_PROCESSOR


def _get_thinking_model():
    global _THINKING_MODEL, _THINKING_PROCESSOR
    if _THINKING_MODEL is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
        _THINKING_MODEL = AutoModelForImageTextToText.from_pretrained(
            _THINKING_NAME, dtype=dtype, device_map="auto")
        _THINKING_MODEL.eval()
    if _THINKING_PROCESSOR is None:
        _THINKING_PROCESSOR = AutoProcessor.from_pretrained(_THINKING_NAME, use_fast=False)
    return _THINKING_MODEL, _THINKING_PROCESSOR


# ════════════════════════════════════════════════════════════════
#  Question classification
# ════════════════════════════════════════════════════════════════

def _is_continuous_legend_question(question):
    return "continuous legend" in " ".join(question.lower().split())


def _is_title_question(question):
    return "what is its title" in " ".join(question.lower().split())


# ════════════════════════════════════════════════════════════════
#  INSTRUCT PATH (from best_accuracy.py — full pipeline)
# ════════════════════════════════════════════════════════════════

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
    "not applicable", "not available", "not provided", "not shown", "not visible",
    "not specified", "not given", "cannot determine", "can't determine",
    "cannot be determined", "unable to determine", "unclear", "not clear",
    "unknown", "none", "no legend", "no colorbar", "no continuous legend",
    "no data", "not present", "does not exist", "not a line plot", "no lines",
    "not sure", "unsure", "cannot tell", "can't tell", "hard to tell",
    "insufficient information", "insufficient data", "no information",
]


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
        img = _resize_chart(base, 1600, 1400, 2200)
        img = ImageOps.autocontrast(img, cutoff=0.5)
        img = ImageEnhance.Color(img).enhance(0.98)
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=170, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.08)
        return img
    if variant == "title_sharp":
        img = _resize_chart(base, 1800, 1600, 2400)
        img = ImageOps.autocontrast(img, cutoff=0.3)
        img = ImageEnhance.Color(img).enhance(0.97)
        img = ImageEnhance.Contrast(img).enhance(1.22)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.14)
        return img
    if variant == "title_gray":
        img = _resize_chart(base, 1850, 1650, 2400)
        img = ImageOps.grayscale(img).convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=0.2)
        img = ImageEnhance.Contrast(img).enhance(1.28)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=190, threshold=2))
        img = ImageEnhance.Sharpness(img).enhance(1.12)
        return img
    raise ValueError(f"Unknown view variant: {variant}")


def _get_instruct_image_inputs(image_path, variant="default"):
    key = (image_path, variant)
    if key not in _IMAGE_CACHE:
        img = _make_view(image_path, variant)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        image_inputs, _ = process_vision_info(messages)
        _IMAGE_CACHE[key] = (image_inputs, img)
    return _IMAGE_CACHE[key]


def _build_instruct_prompt(question):
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
    return chr(10).join([
        "Use ONLY visible chart evidence.",
        "Return ONLY the final answer.",
        "Single line only.",
        "If the answer is not explicitly visible, answer exactly: Not Applicable.",
        "Copy the exact plot title text only. If no explicit title exists, answer Not Applicable.",
        "If the title is only a letter or panel marker such as (a), answer exactly: Not Applicable.",
        f"Question: {question.strip()}",
    ])


def _dedupe(items):
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _instruct_candidates(raw_text):
    text = raw_text.split("</think>")[-1].replace("<think>", " ")
    text = text.replace("**", " ").replace("`", " ")
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
    candidates.append(text.strip())
    return _dedupe([" ".join(c.split()) for c in candidates if c.strip()])


def _looks_not_applicable(text, question=""):
    lower = text.lower().strip()
    if not lower:
        return True
    if lower in {"n/a", "na", "none", "null"}:
        return True
    if any(phrase in lower for phrase in _NA_PHRASES):
        return True
    q = " ".join(question.lower().split()) if question else ""
    if q and "legend" in q and lower in {"no", "none"}:
        return True
    if q and ("label" in q or "title" in q) and lower in {"no", "none"}:
        return True
    if q and "intersect" in q and "line" in q:
        if re.search(r"\bno\s+lines?\s+(are\s+)?(shown|present|plotted|visible|available|drawn|displayed)\b", lower):
            return True
        if "not a line plot" in lower or "not a line graph" in lower or "no line plot" in lower:
            return True
    return False


def _normalize_instruct_text(question, candidates):
    q = " ".join(question.lower().split())
    for candidate in candidates:
        if _looks_not_applicable(candidate, question):
            return "Not Applicable"
        value = candidate.strip().strip('"').strip("'")
        if "names of the labels in the legend" in q:
            for prefix in ["the names of the labels in the legend are ",
                           "the labels in the legend are ", "the labels are ", "labels are "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
            value = ", ".join(p.strip() for p in value.split(",") if p.strip())
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
            value = {"increasing": "increases", "decreasing": "decreases"}.get(value.lower(), value)
        if value:
            return value
    return "Not Applicable"


def _normalize_instruct_answer(question, raw_text):
    q = " ".join(question.lower().split())
    candidates = _instruct_candidates(raw_text)
    if not candidates:
        return "Not Applicable"

    if "layout of the subplots" in q:
        for c in candidates:
            match = _LAYOUT_RE.search(c)
            if match:
                return f"{match.group(1)} by {match.group(2)}"
            if _looks_not_applicable(c, question):
                return "Not Applicable"
        return "1 by 1"

    if "do any lines intersect" in q:
        for c in candidates:
            if _looks_not_applicable(c, question):
                return "Not Applicable"
            words = [w for w in re.split(r"[^a-z]+", c.lower()) if w]
            if "yes" in words:
                return "Yes"
            if "no" in words:
                return "No"
        return "Not Applicable"

    if "how many discrete labels are there in the legend" in q:
        for c in candidates:
            if _looks_not_applicable(c, question):
                return "Not Applicable"
            if ',' in c:
                parts = [p.strip() for p in c.split(',') if p.strip()]
                if parts:
                    return str(len(set(parts)))
            matches = _NUMBER_RE.findall(c.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        is_positional_tick = any(p in q for p in [
            "leftmost labeled tick", "rightmost labeled tick",
            "spatially lowest labeled tick", "spatially highest labeled tick",
        ])
        if is_positional_tick and candidates:
            first = candidates[0] if len(candidates) == 1 else candidates[-1]
            raw_clean = first.strip()
            if raw_clean and '^' in raw_clean and re.fullmatch(r'[-+]?[0-9]+\^[-+]?[0-9]+', raw_clean):
                return raw_clean
            if raw_clean and any(c.isalpha() for c in raw_clean) and not _looks_not_applicable(raw_clean, question):
                if not re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?%?', raw_clean):
                    return _normalize_instruct_text(question, candidates)
        for c in candidates:
            if _looks_not_applicable(c, question):
                return "Not Applicable"
            if "legend" in q and ("no legend" in c.lower() or "no colorbar" in c.lower() or "no continuous legend" in c.lower()):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(c.replace(",", ""))
            if matches:
                return matches[-1].rstrip("%")
        if is_positional_tick:
            return _normalize_instruct_text(question, candidates)
        return "Not Applicable"

    return _normalize_instruct_text(question, candidates)


def _run_instruct(model, processor, image_path, question, prompt, variant="default"):
    image_inputs, img = _get_instruct_image_inputs(image_path, variant)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    q_lower = " ".join(question.lower().split())
    max_tokens = _MAX_NEW_TOKENS
    if "legend" in q_lower or "title" in q_lower or "label of the" in q_lower:
        max_tokens = max(_MAX_NEW_TOKENS, 64)
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True, pad_token_id=pad_id)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _instruct_inference(image_path, question):
    """Full Instruct pipeline: preprocessing + prompt + normalization + verifiers."""
    model, processor = _get_instruct_model()
    prompt = _build_instruct_prompt(question)
    raw = _run_instruct(model, processor, image_path, question, prompt, "default")
    base_answer = _normalize_instruct_answer(question, raw)

    # Colorbar verifier
    if _is_continuous_legend_question(question) and base_answer != "Not Applicable":
        probe_prompt = chr(10).join([
            "Look at this chart carefully.",
            "Does this chart contain a continuous legend (also called a colorbar)?",
            "A colorbar is a gradient bar mapping colors to numeric values.",
            "A discrete legend listing category names is NOT a colorbar.",
            "Answer exactly: Yes or No.",
        ])
        probe_raw = _run_instruct(model, processor, image_path, "", probe_prompt, "default")
        probe_answer = _normalize_instruct_answer("", probe_raw)
        lower = probe_answer.strip().lower()
        if "no" in lower and "yes" not in lower:
            return "Not Applicable"

    # Title verifier
    if _is_title_question(question) and _PANEL_RE.fullmatch(base_answer.strip()) or (
        _is_title_question(question) and len(base_answer.strip()) <= 3
        and base_answer.strip().replace(".", "").replace("-", "").isalpha()
        and base_answer.strip() != "Not Applicable"
    ):
        probe_prompt = _build_title_probe_prompt(question)
        answers = []
        for variant in ["default", "title_sharp", "title_gray"]:
            raw_v = _run_instruct(model, processor, image_path, question, probe_prompt, variant)
            answers.append(_normalize_instruct_answer(question, raw_v))
        if answers[0] == answers[1] == answers[2] and answers[0] != base_answer:
            return answers[0]

    return base_answer


# ════════════════════════════════════════════════════════════════
#  THINKING PATH (for continuous-legend questions)
# ════════════════════════════════════════════════════════════════

def _get_thinking_image_inputs(image_path):
    if image_path not in _THINKING_IMAGE_CACHE:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}]}]
        image_inputs, _ = process_vision_info(messages)
        _THINKING_IMAGE_CACHE[image_path] = image_inputs
    return _THINKING_IMAGE_CACHE[image_path]


def _thinking_candidates(raw_text):
    candidates = []
    blocks = []
    if "</think>" in raw_text:
        after = raw_text.split("</think>")[-1]
        if after.strip():
            blocks.append(after)
        think_blocks = re.findall(r"<think>(.*?)</think>", raw_text, flags=re.S)
        if think_blocks and think_blocks[-1].strip():
            blocks.append(think_blocks[-1])
    if not blocks:
        blocks = [raw_text]
    for text in blocks:
        text = text.replace("<think>", " ").replace("</think>", " ")
        text = text.replace("**", " ").replace("`", " ")
        lines = [line.strip(" -:") for line in text.splitlines() if line.strip()]
        candidates.extend(reversed(lines))
        if ":" in text:
            candidates.append(text.split(":")[-1].strip())
        candidates.append(text.strip())
    return _dedupe([" ".join(c.split()) for c in candidates if c.strip()])


def _thinking_normalize(question, raw_text):
    """Normalize Thinking model output for continuous-legend questions."""
    q = " ".join(question.lower().split())
    candidates = _thinking_candidates(raw_text)
    if not candidates:
        return "Not Applicable"

    for c in candidates:
        lower = c.lower()
        if "not applicable" in lower or "no colorbar" in lower or "no continuous legend" in lower:
            return "Not Applicable"
        if "no legend" in lower:
            return "Not Applicable"
        # Extract number for value questions
        matches = _NUMBER_RE.findall(c.replace(",", ""))
        if matches:
            return matches[-1].rstrip("%")

    return "Not Applicable"


def _thinking_inference(image_path, question):
    """Thinking model for continuous-legend questions (better at NA detection)."""
    model, processor = _get_thinking_model()
    q = " ".join(question.lower().split())
    lines = [
        "You are answering a chart question.",
        "Return ONLY the final answer on a single line.",
        "Do NOT explain reasoning or repeat the question.",
        "If the answer is absent or not explicitly supported by the chart, answer exactly: Not Applicable.",
        "Use only a continuous legend or colorbar. If none exists, answer exactly: Not Applicable.",
        "If there is no colorbar or continuous legend in the chart, answer exactly: Not Applicable.",
        "numbers/counts/ticks: output only the value (no commas, no units, no words)",
        "After thinking, output ONLY the answer with no tags or punctuation.",
        f"Question: {question.strip()}",
    ]
    prompt = chr(10).join(lines)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + "<think>" + chr(10) + "</think>" + chr(10) + chr(10)
    image_inputs = _get_thinking_image_inputs(image_path)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False, use_cache=True, pad_token_id=pad_id)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _thinking_normalize(question, raw)


# ════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════

def vlm_inference(image_path, question="Describe this image in detail."):
    """Question-type router: Thinking for continuous-legend, Instruct for everything else."""
    if _is_continuous_legend_question(question):
        return _thinking_inference(image_path, question)
    return _instruct_inference(image_path, question)
