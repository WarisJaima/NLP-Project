import language_tool_python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import difflib
import re
import streamlit as st
import streamlit.components.v1 as components

# Load model & tokenizer
model_dir = "./multitask-gec-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

# Load LanguageTool
tool = language_tool_python.LanguageTool('en-US')

# Postprocessing rules
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
    count = 0
    for pat in patterns:
        matches = re.findall(pat, text)
        count += len(matches)
    return count

def correct_paragraph(paragraph):
    lt_corrected = tool.correct(paragraph)
    input_ids = tokenizer("gec: " + lt_corrected, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_length=512)
    corrected = tokenizer.decode(output[0], skip_special_tokens=True)
    corrected = simplify_comparative(corrected)
    corrected = fix_conditional_third(corrected)
    corrected = fix_time_expressions(corrected)
    return corrected

def correct_text(text, debug=False):
    paragraphs = re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())
    corrected_paragraphs = [correct_paragraph(p) for p in paragraphs]
    corrected = " ".join(corrected_paragraphs)

    if debug:
        print("🔍 Original :", text)
        print("✅ Corrected:", corrected)
        missed = detect_remaining_errors(corrected)
        print(f"🚨 Remaining suspicious patterns: {missed}")
        if "since" in corrected and "for" not in corrected:
            print("⚠️ Possible misuse of 'since' instead of 'for'. Consider reviewing time expressions.")

    return corrected

# JS + HTML for diff highlighting
diff_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/diff_match_patch/20121119/diff_match_patch.js"></script>
<script>
function diffText(original, corrected) {
  var dmp = new diff_match_patch();
  var diffs = dmp.diff_main(original, corrected);
  dmp.diff_cleanupSemantic(diffs);
  let html = "";
  diffs.forEach(function(part) {
    const [op, text] = part;
    if (op === -1) html += `<span style='color:red;text-decoration:line-through;'>${text}</span>`;
    else if (op === 1) html += `<span style='color:limegreen;'>${text}</span>`;
    else html += text;
  });
  document.getElementById("diff-output").innerHTML = html;
}
</script>
"""

# Streamlit UI
st.set_page_config(page_title="Grammar Correction App", layout="wide")
st.title("📜 Grammar Correction")

input_text = st.text_area("✏️ Input Sentence", height=200, label_visibility="visible")

if input_text:
    corrected = correct_text(input_text)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🖍️ Visual Diff")
        components.html(f"""
            {diff_js}
            <div id="diff-output" style="width: 100%; height: 200px; background-color: #2c2c2c; color: white; padding: 10px; font-family: 'Source Code Pro', monospace; border-radius: 6px; border: 1px solid #555; overflow: auto;"></div>
            <script>diffText(`{input_text}`, `{corrected}`)</script>
        """, height=250)

    with col2:
        st.markdown("### ✅ Corrected Output")
        st.text_area("Corrected Text", corrected, height=200, label_visibility="collapsed", disabled=True, key="corrected-output")

    missed = detect_remaining_errors(corrected)
    if missed:
        st.warning(f"🚨 Remaining suspicious patterns: {missed}")
    if "since" in corrected and "for" not in corrected:
        st.info("⚠️ Possible misuse of 'since' instead of 'for'. Consider reviewing time expressions.")
    st.success("✅ Correction complete")