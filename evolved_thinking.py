# EVOLVE-BLOCK-START
"""Manual CharXiv inference optimized for Qwen/Qwen3-VL-2B-Thinking."""

import re
from decimal import Decimal, InvalidOperation

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

_SYSTEM_PROMPT = (
    "You are a strict chart QA assistant. "
    "Output ONLY the final answer string. "
    "No reasoning, no preambles, no tags, no extra words. "
    "Single line only. If not explicitly supported, output exactly: Not Applicable."
)

_NUMBER_RE = re.compile(r"[-+]?(?:[0-9]*[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?%?")
_NUMBER_FULL_RE = re.compile(r"^\s*[-+]?(?:[0-9]*[.][0-9]+|[0-9]+)(?:[eE][-+]?[0-9]+)?%?\s*$")
_LAYOUT_RE = re.compile(r"([0-9]+) *(?:by|x) *([0-9]+)", re.IGNORECASE)
_LAYOUT_WORD_RE = re.compile(r"([0-9]+) *(?:rows?|row) *by *([0-9]+) *(?:columns?|cols?|column)?", re.IGNORECASE)

_WORD_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS.keys()) + r")\b", re.IGNORECASE)

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
    "how many subplots",
    "number of panels",
]

_NA_PHRASES = [
    "not applicable",
    "n/a",
    "not available",
    "not provided",
    "not shown",
    "not visible",
    "not given",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "no data",
    "no information",
]
_LEGEND_NA_PHRASES = [
    "no legend",
    "legend not shown",
    "legend not provided",
    "legend absent",
    "no colorbar",
    "no continuous legend",
    "no discrete legend",
    "without legend",
]
_TITLE_NA_PHRASES = [
    "no title",
    "untitled",
    "title not shown",
    "title not provided",
]
_AXIS_NA_PHRASES = [
    "no axis label",
    "axis label not shown",
    "axis labels not shown",
    "unlabeled",
    "no x-axis label",
    "no x axis label",
    "x-axis label not shown",
    "x axis label not shown",
    "no y-axis label",
    "no y axis label",
    "y-axis label not shown",
    "y axis label not shown",
]
_GENERAL_NA_SINGLETONS = {"none", "n/a", "na", "not applicable"}

_YESNO_STARTS = (
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "was ",
    "were ",
    "can ",
    "could ",
    "should ",
    "has ",
    "have ",
    "had ",
    "will ",
    "would ",
    "may ",
    "might ",
)


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
        "Answer the chart question with ONLY the final answer.",
        "CRITICAL: Output must be a SINGLE LINE with JUST the answer text.",
        "Do NOT include any reasoning, explanation, or extra words.",
        "Do NOT include any prefix like 'Answer:' or 'Final answer:'.",
        "Do NOT output any <think> tags.",
        "Any extra characters will be graded as wrong.",
        "If the answer is absent or not explicitly supported by the chart, answer exactly: Not Applicable.",
        "Use the chart text exactly when possible.",
        "Formats:",
        "- numbers/counts/ticks: only the value (no units unless shown)",
        "- yes/no questions: Yes, No, or Not Applicable",
        "- subplot layout: n by m",
        "- legend names: label1, label2, label3 (comma-separated, no 'and')",
    ]
    if "continuous legend" in q:
        lines.append("Use only a continuous legend or colorbar. If none exists, answer exactly: Not Applicable.")
    if "legend" in q and "continuous legend" not in q:
        lines.append("Use only the relevant discrete legend. If there is no legend for this plot, answer exactly: Not Applicable.")
    if "names of the labels in the legend" in q or "labels in the legend" in q:
        lines.append("Return legend labels exactly as shown, in order, comma-separated, no extra words.")
    if "label of the x-axis" in q or "label of the x axis" in q:
        lines.append("For x-axis label questions, answer with the exact x-axis label text only (case and spacing).")
    if "label of the y-axis" in q or "label of the y axis" in q:
        lines.append("For y-axis label questions, answer with the exact y-axis label text only (case and spacing).")
    if "what is its title" in q or "plot title" in q:
        lines.append("For title questions, answer with the exact title text only.")
    if "general trend of data from left to right" in q:
        lines.append("For trend questions, answer with a short phrase only (e.g., increasing, decreasing, flat, mixed). Prefer one word.")
    if "layout of the subplots" in q:
        lines.append("Layout means rows by columns. Two side-by-side subplots are 1 by 2. Two stacked subplots are 2 by 1. Single subplot is 1 by 1.")
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


def _extract_answer_text(raw_text: str) -> str:
    text = raw_text
    if "</think>" in text:
        after = text.rsplit("</think>", 1)[1]
        if after.strip():
            text = after
        else:
            before = text.rsplit("</think>", 1)[0]
            if "<think>" in before:
                text = before.rsplit("<think>", 1)[1]
            else:
                text = before
    elif "<think>" in text:
        text = text.rsplit("<think>", 1)[1]
    return text


def _candidates(raw_text):
    segments = []
    primary = _extract_answer_text(raw_text)
    segments.append(primary)
    if primary.strip() != raw_text.strip():
        segments.append(raw_text)
    candidates = []
    for seg in segments:
        text = seg.replace("<think>", " ").replace("</think>", " ")
        text = text.replace("**", " ").replace("`", " ")
        lines = [line.strip(" -:") for line in text.splitlines() if line.strip()]
        candidates.extend(list(reversed(lines)))
        if ":" in text:
            candidates.append(text.split(":")[-1].strip())
        candidates.append(text.strip())
    return _dedupe([" ".join(candidate.split()) for candidate in candidates if candidate.strip()])


def _replace_word_numbers(text):
    return _WORD_NUMBER_RE.sub(lambda m: _WORD_NUMBERS[m.group(1).lower()], text)


def _format_number_token(token):
    if token is None:
        return None
    s = token.strip().replace("−", "-")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    if s.endswith("%"):
        s = s[:-1]
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d == d.to_integral():
        return str(int(d))
    s = format(d.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _is_not_applicable(question, candidate):
    if candidate is None:
        return True
    lower = candidate.strip().lower()
    if not lower:
        return True
    if lower in _GENERAL_NA_SINGLETONS:
        return True
    for phrase in _NA_PHRASES:
        if phrase in lower:
            return True

    q = " ".join(question.lower().split())
    if "legend" in q:
        for phrase in _LEGEND_NA_PHRASES:
            if phrase in lower:
                return True
    if "continuous legend" in q and ("colorbar" in lower and "no" in lower):
        return True
    if "label of the x-axis" in q or "label of the x axis" in q:
        for phrase in _AXIS_NA_PHRASES:
            if phrase in lower:
                return True
        if "x-axis" in lower and "no" in lower and "label" in lower:
            return True
    if "label of the y-axis" in q or "label of the y axis" in q:
        for phrase in _AXIS_NA_PHRASES:
            if phrase in lower:
                return True
        if "y-axis" in lower and "no" in lower and "label" in lower:
            return True
    if "plot title" in q or "what is its title" in q:
        for phrase in _TITLE_NA_PHRASES:
            if phrase in lower:
                return True
    if "do any lines intersect" in q:
        if "no lines" in lower or "not a line plot" in lower or "no line plot" in lower:
            return True
    return False


def _is_yesno_question(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith(_YESNO_STARTS)


def _normalize_text_answer(question, candidates):
    q = " ".join(question.lower().split())
    is_yesno = _is_yesno_question(question)
    for candidate in candidates:
        if _is_not_applicable(question, candidate):
            return "Not Applicable"
        value = candidate.strip().strip('"').strip("'")
        for prefix in ["final answer is ", "final answer: ", "answer is ", "answer: "]:
            if value.lower().startswith(prefix):
                value = value[len(prefix):]
                break
        if "names of the labels in the legend" in q or "labels in the legend" in q:
            prefixes = [
                "the names of the labels in the legend are ",
                "the labels in the legend are ",
                "the labels are ",
                "labels are ",
                "legend labels are ",
                "legend labels: ",
            ]
            lower_value = value.lower()
            for prefix in prefixes:
                if lower_value.startswith(prefix):
                    value = value[len(prefix):]
                    break
            value = value.replace(" and ", ", ").replace(" & ", ", ").replace(";", ",")
            value = ", ".join(part.strip() for part in value.split(",") if part.strip())
        elif "label of the x-axis" in q or "label of the x axis" in q:
            for prefix in [
                "the x-axis label is ",
                "x-axis label is ",
                "label of the x-axis is ",
                "the x axis label is ",
                "x axis label is ",
                "label of the x axis is ",
            ]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "label of the y-axis" in q or "label of the y axis" in q:
            for prefix in [
                "the y-axis label is ",
                "y-axis label is ",
                "label of the y-axis is ",
                "the y axis label is ",
                "y axis label is ",
                "label of the y axis is ",
            ]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "what is its title" in q or "plot title" in q:
            for prefix in ["the title is ", "title is ", "the plot title is ", "plot title is ", "title: "]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
        elif "general trend of data from left to right" in q:
            for prefix in [
                "the general trend is ",
                "general trend is ",
                "overall trend is ",
                "trend is ",
                "the trend is ",
            ]:
                if value.lower().startswith(prefix):
                    value = value[len(prefix):]
                    break
            lower_val = value.lower()
            if "increasing" in lower_val and "decreasing" not in lower_val:
                value = "increasing"
            elif "decreasing" in lower_val and "increasing" not in lower_val:
                value = "decreasing"
            elif "flat" in lower_val or "stable" in lower_val or "constant" in lower_val:
                value = "flat"
            elif "mixed" in lower_val or "varied" in lower_val:
                value = "mixed"
        value = value.strip().strip('"').strip("'").rstrip(" .;:,-!?")
        if value:
            lower_val = value.lower()
            if lower_val == "yes" or lower_val == "true":
                return "Yes"
            if lower_val == "no" or lower_val == "false":
                return "No"
            if is_yesno:
                words = [w for w in re.split(r"[^a-z]+", lower_val) if w]
                if "yes" in words or "true" in words:
                    return "Yes"
                if "no" in words or "false" in words:
                    return "No"
            if _NUMBER_FULL_RE.match(value):
                value = _format_number_token(value)
            if _is_not_applicable(question, value):
                return "Not Applicable"
            return value
    return "Not Applicable"


def _normalize_answer(question, raw_text):
    q = " ".join(question.lower().split())
    candidates = _candidates(raw_text)
    if not candidates:
        return "Not Applicable"

    if "layout of the subplots" in q:
        for candidate in candidates:
            if _is_not_applicable(question, candidate):
                return "Not Applicable"
            candidate2 = _replace_word_numbers(candidate)
            match = _LAYOUT_RE.search(candidate2)
            if not match:
                match = _LAYOUT_WORD_RE.search(candidate2)
            if match:
                return f"{match.group(1)} by {match.group(2)}"
        return "1 by 1"

    if "do any lines intersect" in q:
        for candidate in candidates:
            lower = candidate.lower()
            if _is_not_applicable(question, candidate):
                return "Not Applicable"
            words = [word for word in re.split(r"[^a-z]+", lower) if word]
            if "yes" in words or "true" in words:
                return "Yes"
            if "no" in words or "false" in words:
                return "No"
        return "Not Applicable"

    if any(phrase in q for phrase in _NUMERIC_PHRASES):
        for candidate in candidates:
            if _is_not_applicable(question, candidate):
                return "Not Applicable"
            matches = _NUMBER_RE.findall(candidate.replace(",", ""))
            if matches:
                return _format_number_token(matches[-1])
            word_match = _WORD_NUMBER_RE.search(candidate)
            if word_match:
                return _WORD_NUMBERS[word_match.group(1).lower()]
        return "Not Applicable"

    return _normalize_text_answer(question, candidates)


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    prompt = _build_prompt(question)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
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
