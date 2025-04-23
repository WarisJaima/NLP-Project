# === models/essay_scoring.py ===
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

# === Custom LoRA Model Wrapper ===
class DebertaWithLoRA(nn.Module):
    def __init__(self, base_model, lora_config):
        super().__init__()
        self.backbone = get_peft_model(base_model, lora_config)
        self.regressor = nn.Linear(base_model.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.regressor(cls_output)

# === Load Paths ===
load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "essay_scoring_model", "lora_v2_5_model"))

# === Load Tokenizer & Base Model ===
tokenizer = AutoTokenizer.from_pretrained(load_path)
base_model = AutoModel.from_pretrained(load_path)

# === LoRA Config (must match training) ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_proj", "key_proj", "value_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

# === Build Full Model ===
model = DebertaWithLoRA(base_model, lora_config)
model.backbone.load_state_dict(torch.load(os.path.join(load_path, "lora_adapter.bin"), map_location="cpu"), strict=False)
model.regressor.load_state_dict(torch.load(os.path.join(load_path, "regression_head.pt"), map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Postprocessing Logic ===
def postprocess_scores(scores):
    scores = scores * 5 + 4
    scores = torch.round(scores * 2) / 2
    return torch.clamp(scores, min=4.0, max=9.0)

# === Score Function ===
def score_essay(prompt, essay):
    text = f"Prompt: {prompt}\nEssay: {essay}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs).squeeze()
        scores = postprocess_scores(outputs)

    return {
        "Task Achievement": scores[0].item(),
        "Coherence & Cohesion": scores[1].item(),
        "Lexical Resource": scores[2].item(),
        "Grammar": scores[3].item(),
        "Overall Band": scores[4].item()
    }
