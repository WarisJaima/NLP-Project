# NLP Project: IELTS Essay Generation & Grammar Correction

This project focuses on two main tasks in NLP:

1. **IELTS Writing Task 2 Essay Generation**  
   Using T5-base with LoRA fine-tuning: [YouTube Demo](https://www.youtube.com/watch?v=bh9voRM_CkU)

2. **Grammar Correction and Simplification**  
   Using T5-base and rule-based postprocessing: [YouTube Demo](https://www.youtube.com/watch?v=ysy86-DAu-o)

3. **Essay Scoring Model**  
   Using DeBERTa (RegressionTrainer): [YouTube Demo](https://www.youtube.com/watch?v=t_tLX0fkJQc)

4. **Essay Evaluation Model**  
   Using Mistral: [YouTube Demo](https://www.youtube.com/watch?v=xUbeBi-gDyM)

---

## 1. Essay Generation (IELTS Writing Task 2)

### Tools Used
- **Transformers (Hugging Face)**: T5-base model and tokenizer
- **PEFT (LoRA)**: Lightweight fine-tuning with low-rank adapters
- **WandB**: Logging, tracking, and visualization of experiments
- **Hugging Face Datasets**: For loading `chillies/IELTS-writing-task-2-evaluation`
- **SentenceTransformers (SBERT)**: Semantic similarity evaluation
- **Facebook BART-MNLI**: NLI-based contradiction and relevance detection
- **Scikit-learn**: F1 score metrics
- **LangChain**: Prompt chaining for multi-paragraph generation
- **LanguageTool**: Rule-based grammar and coherence checking

### Preprocessing
- **Filtering**: Essays with band ≥ 7.0 and >220 words
- **Paragraph Splitting**: Intro, Body 1, Body 2 (merged), Conclusion
- **Data Cleaning**: Removed essays with poor structure or high redundancy
- **Tokenization**: T5 tokenizer with truncation and padding
- **Input Augmentation**:
  - Prompt masking with `<extra_id_0>`
  - Paraphrased intros (light T5-based model)
  - Contrastive augmentation for Body 2

### Modeling
- **Architecture**: T5-base (220M params)
- **LoRA Configuration**: `r=32`, `alpha=64`, applied to "q" and "v" projection layers
- **Training**:
  - 4 epochs
  - Batch size: 2
  - Learning rate: 3e-4
  - FP16 enabled
- **Custom DPO-Style Loss**:
  - Cross-Entropy + penalties for:
    - Low lexical diversity
    - Body1–Body2 overlap
    - Contradictions (via MNLI)
    - Topic irrelevance
    - Missing arguments (keyword-based)
    - Reused n-grams

### Inference Pipeline
- Built using **LangChain SequentialChain**:
  - Intro → Body1 → Body2 (context-aware) → Conclusion
- **Evaluation**:
  - Word count & coherence check
  - Body1–Body2 contrast score
  - Argument keyword presence
  - Grammar issue detection (via LanguageTool)
- **Essay Regeneration**: If conditions not met (max 5 retries)

### Postprocessing Highlights
- Essays must meet:
  - Min 220 words
  - Max 60% Body1–Body2 lexical overlap
  - Argument quality verified via keyword scan
  - Coherence score from discourse markers
  - Conflict detection between intro and conclusion
  - Repeated words flagged (>2 times if >3 characters)
  - Grammar errors reported (not corrected) using LanguageTool

---

## 2. Grammar Correction System

### Tools Used
- **T5-base**: (`vennify/grammar-correction` model)
- **Hugging Face Datasets**: Grammarly CoEdit
- **nlpaug**: WordNet synonym augmentation
- **NLTK**: POS tagging
- **LanguageTool**: Grammar analysis
- **Streamlit**: Web UI
- **JavaScript Diff Viewer**: Visual diff between input/output

### Preprocessing & Augmentation
- Tasks used: `gec`, `clarity`, `simplification`, `paraphrase`
- Synonym-based augmentation with `nlpaug` for lexical variety
- Added task prefixes (e.g., `gec:`) to support multi-task training

### Modeling
- **Fine-tuned with**:
  - Batch size: 8
  - Epochs: 3
  - Learning rate: 3e-5
  - FP16 mixed precision
  - Dataset: 5% of Grammarly CoEdit (filtered and augmented)

### Postprocessing
- **LanguageTool Pre-check**: Cleaned up basic errors before model input
- **Rule-Based Corrections**:
  - Comparatives: e.g., "more simple" → "simpler"
  - Conditionals: e.g., "If I had..., I will" → "...I would"
  - Time Phrases: e.g., "since years" → "for years"

### Inference & Interface
- **Streamlit-based UI** with:
  - Paragraph-wise correction
  - Interactive grammar check with LanguageTool
  - Before/after diff highlighting via JS

---

## 3. Essay Scoring

### Tools Used
- **Transformers (Hugging Face)**: `microsoft/deberta-v3-base` for base architecture
- **PEFT (LoRA)**: Efficient fine-tuning with low-rank adapters (`r=8`, `alpha=16`)
- **Hugging Face Datasets**: Used `chillies/IELTS-writing-task-2-evaluation` for training
- **Scikit-learn**: For evaluation metrics (RMSE)
- **PyTorch**: Custom regression head and training loop
- **Torch Mixed Precision (fp16)**: Faster training

### Preprocessing & Tokenization
- **Band Extraction**: Parsed evaluation field to get scores for:
  - Task Achievement
  - Coherence
  - Lexical Resource
  - Grammar
  - Overall
- **Filtering**: Removed examples without all scores or with unparsable bands
- **Normalization**: All scores scaled to range [0, 1] using `(band - 4) / 5`
- **Input Format**: `"Prompt: ... \nEssay: ..."` tokenized with max length 512

### Model Architecture
- **Backbone**: DeBERTa-v3-base with LoRA injected into query, key, value projections
- **LoRA Configuration**: `r=8`, `alpha=16`, `dropout=0.05`
- **Regression Head**: Linear layer outputting 5 values for the 5 band criteria

### Training Setup
- Epochs: 5
- Batch Size: 4
- Learning Rate: 2e-4
- Loss Function: Weighted MSE loss across the 5 criteria
  - Weights: 0.2 (each sub-score) + 0.4 (Overall Band)
- Evaluation Metric: RMSE (Root Mean Square Error)

### Inference Pipeline
- **Input**: Prompt + Essay
- **Output**: 5 continuous scores → scaled back to band range [4.0 – 9.0], rounded to nearest 0.5
- **Postprocessing**:
  - Band scores scaled via `score * 5 + 4`
  - Rounded to nearest 0.5 and clamped to [4.0, 9.0]
