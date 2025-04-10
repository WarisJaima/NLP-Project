NLP Project: IELTS Essay Generation & Grammar Correction
This project focuses on two main tasks in NLP:

IELTS Writing Task 2 Essay Generation using T5-base with LoRA fine-tuning: https://www.youtube.com/watch?v=bh9voRM_CkU

Grammar Correction and Simplification using T5-base and rule-based postprocessing: https://www.youtube.com/watch?v=ysy86-DAu-o

1. Essay Generation (IELTS Writing Task 2)
Tools Used
Transformers (Hugging Face) – T5-base model and tokenizer

PEFT (LoRA) – Lightweight fine-tuning with low-rank adapters

WandB – Logging, tracking and visualization of experiments

Hugging Face Datasets – For loading chillies/IELTS-writing-task-2-evaluation

SentenceTransformers (SBERT) – Semantic similarity evaluation

Facebook BART-MNLI – NLI-based contradiction and relevance detection

Scikit-learn – F1 score metrics

LangChain – Prompt chaining for multi-paragraph generation

LanguageTool – Rule-based grammar and coherence checking

Preprocessing
Filtering: Only essays with band ≥ 7.0 and >220 words

Paragraph Splitting: Intro, Body 1, Body 2 (merged), Conclusion

Data Cleaning: Removed essays with poor structure, high redundancy

Tokenization: T5 tokenizer with truncation and padding

Input Augmentation:

Prompt masking with <extra_id_0>

Paraphrased intros (light T5-based model)

Contrastive augmentation for Body 2

Modeling
Architecture: T5-base (220M params)

LoRA Configuration: r=32, alpha=64, on "q" and "v" projection layers

Training: 4 epochs, batch size 2, learning rate 3e-4, FP16 enabled

Custom DPO-Style Loss:

Cross-Entropy + penalties for:

Low lexical diversity

Body1–Body2 overlap

Contradictions (via MNLI)

Topic irrelevance

Missing arguments (keyword-based)

Reused n-grams

Inference Pipeline
Built using LangChain SequentialChain: Intro → Body1 → Body2 (context-aware) → Conclusion

Evaluation included:

Word count & coherence check

Body1–Body2 contrast score

Argument keyword presence

Grammar issue detection (via LanguageTool)

Essay regeneration if conditions not met (max 5 retries)

Postprocessing Highlights
Essays must meet:

Min 220 words

Max 60% Body1–Body2 lexical overlap

Argument quality verified via keyword scan

Coherence score from discourse markers

Conflict detection between intro and conclusion

Repeated words flagged (>2 times if >3 characters)

Grammar errors reported (not corrected) using LanguageTool

2. Grammar Correction System
Tools Used
T5-base (vennify/grammar-correction) model

Hugging Face Datasets – Grammarly CoEdit

nlpaug – WordNet synonym augmentation

NLTK – POS tagging

LanguageTool – Grammar analysis

Streamlit – Web UI

JavaScript Diff Viewer – Visual diff between input/output

Preprocessing & Augmentation
Tasks used: gec, clarity, simplification, paraphrase

Synonym-based augmentation with nlpaug for lexical variety

Added task prefixes (e.g., gec:) to support multi-task training

Modeling
Fine-tuned with:

Batch size: 8

Epochs: 3

Learning rate: 3e-5

FP16 mixed precision

Dataset: 5% of Grammarly CoEdit (filtered and augmented)

Postprocessing
LanguageTool Pre-check: Cleaned up basic errors before model input

Rule-Based Corrections:

Comparatives: e.g., "more simple" → "simpler"

Conditionals: e.g., "If I had..., I will" → "...I would"

Time Phrases: e.g., "since years" → "for years"

Inference & Interface
Streamlit-based UI with:

Paragraph-wise correction

Interactive grammar check with LanguageTool

Before/after diff highlighting via JS

Evaluation Metrics
For essay generation:

Semantic Score – SBERT cosine similarity

Diversity Score – distinct-2 n-gram ratio

F1 Score – Word-level overlap

Final Score – Weighted average of above