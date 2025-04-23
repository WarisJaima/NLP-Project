import re
import torch
import language_tool_python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import difflib

# === Load model & tokenizer ===
model_dir = "grammar_correction_model/multitask-gec-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Load LanguageTool ===
tool = language_tool_python.LanguageTool('en-US')

# === Postprocessing rules ===
def simplify_comparative(text):
    pattern = r"\bmore ([a-zA-Z]+?)\b"
    irregular = {"good": "better", "bad": "worse", "far": "farther", "angry": "angrier", "simple": "simpler", "little": "less"}
    blocked = {"better", "worse", "less", "more"}

    def replace(match):
        word = match.group(1)
        if word in blocked:
            return word
        if word in irregular:
            return irregular[word]
        elif len(word) <= 6 and not word.endswith("ly"):
            if word.endswith("y"):
                return word[:-1] + "ier"
            return word + "er"
        return "more " + word

    return re.sub(pattern, replace, text)

def fix_conditional_third(text):
    return re.sub(r"(If I had .*?), I will", r"\1, I would", text)

def fix_time_expressions(text):
    return re.sub(r"\bsince (days|weeks|months|years)\b", r"for \1", text)

def detect_remaining_errors(text):
    patterns = [
        r"\bmore [a-zA-Z]+?\b",
        r"\b[a-zA-Z]+\s+(don't|doesn't|didn't)\s+[a-zA-Z]+\b",
        r"\b[a-zA-Z]+\s+have\s+[a-zA-Z]+ed\b",
        r"\bsince (days|weeks|months|years)\b"
    ]
    return sum(len(re.findall(pat, text)) for pat in patterns)

# === Correction logic ===
def correct_paragraph(paragraph):
    lt_corrected = tool.correct(paragraph)
    input_ids = tokenizer("gec: " + lt_corrected, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=512)
    corrected = tokenizer.decode(output[0], skip_special_tokens=True)
    corrected = simplify_comparative(corrected)
    corrected = fix_conditional_third(corrected)
    corrected = fix_time_expressions(corrected)
    return corrected

def correct_text(text):
    paragraphs = re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())
    corrected_paragraphs = [correct_paragraph(p) for p in paragraphs]
    corrected = " ".join(corrected_paragraphs)
    return corrected

def generate_diff_html(original, corrected):
    differ = difflib.Differ()
    diff = list(differ.compare(original.split(), corrected.split()))
    html = []

    for word in diff:
        if word.startswith("- "):
            html.append(f"<span style='color:red;text-decoration:line-through'>{word[2:]}</span>")
        elif word.startswith("+ "):
            html.append(f"<span style='color:green;font-weight:bold'>{word[2:]}</span>")
        elif word.startswith("  "):
            html.append(word[2:])
    return " ".join(html)

def correct_text(text):
    paragraphs = re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())
    corrected_paragraphs = [correct_paragraph(p) for p in paragraphs]
    corrected = " ".join(corrected_paragraphs)
    diff_html = generate_diff_html(text, corrected)
    return diff_html