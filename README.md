## Essay Generation & Grammar Correction
This project focuses on two main tasks in NLP:

IELTS Writing Task 2 Essay Generation using T5-base with LoRA fine-tuning: 
[![Watch Video](https://github.com/WarisJaima/NLP-Project/blob/main/Screenshot%20from%202025-04-11%2016-33-16.png)](https://www.youtube.com/watch?v=bh9voRM_CkU)

Grammar Correction and Simplification using T5-base and rule-based postprocessing: 
[![Watch Video](https://github.com/WarisJaima/NLP-Project/blob/main/Screenshot%20from%202025-04-11%2016-37-20.png)](https://www.youtube.com/watch?v=ysy86-DAu-o)


## 1. Essay Generation (IELTS Writing Task 2)
 * Tools Used
        --> Transformers (Hugging Face) – T5-base model and tokenizer
        --> PEFT (LoRA) – Lightweight fine-tuning with low-rank adapters
        --> WandB – Logging, tracking and visualization of experiments
        --> Hugging Face Datasets – For loading chillies/IELTS-writing-task-2-evaluation
        --> SentenceTransformers (SBERT) – Semantic similarity evaluation
        --> Facebook BART-MNLI – NLI-based contradiction and relevance detection
        --> Scikit-learn – F1 score metrics
        --> LangChain – Prompt chaining for multi-paragraph generation
        --> LanguageTool – Rule-based grammar and coherence checking

 * Preprocessing
         --> Filtering: Only essays with band ≥ 7.0 and >220 words
         --> Paragraph Splitting: Intro, Body 1, Body 2 (merged), Conclusion
         --> Data Cleaning: Removed essays with poor structure, high redundancy
         --> Tokenization: T5 tokenizer with truncation and padding
         --> Input Augmentation:
         --> Prompt masking with <extra_id_0>
         --> Paraphrased intros (light T5-based model)
         --> Contrastive augmentation for Body 2

* Modeling
         --> Architecture: T5-base (220M params)
         --> LoRA Configuration: r=32, alpha=64, on "q" and "v" projection layers
         --> Training: 4 epochs, batch size 2, learning rate 3e-4, FP16 enabled
         --> Custom DPO-Style Loss:
         --> Cross-Entropy + penalties for:
         --> Low lexical diversity
         --> Body1–Body2 overlap
         --> Contradictions (via MNLI)
         --> Topic irrelevance
         --> Missing arguments (keyword-based)
         --> Reused n-grams

* Inference Pipeline
         --> Built using LangChain SequentialChain: Intro → Body1 → Body2 (context-aware) → Conclusion
         --> Evaluation included:
         --> Word count & coherence check
         --> Body1–Body2 contrast score
         --> Argument keyword presence

* Grammar issue detection (via LanguageTool)

* Essay regeneration if conditions not met (max 5 retries)

* Postprocessing Highlights Essays must meet:
         --> Min 220 words
         --> Max 60% Body1–Body2 lexical overlap
         --> Argument quality verified via keyword scan
         --> Coherence score from discourse markers
         --> Conflict detection between intro and conclusion
         --> Repeated words flagged (>2 times if >3 characters)

* Grammar errors reported (not corrected) using LanguageTool

## 2. Grammar Correction System
* Tools Used
         --> T5-base (vennify/grammar-correction) model
         --> Hugging Face Datasets – Grammarly CoEdit
         --> nlpaug – WordNet synonym augmentation
         --> NLTK – POS tagging
         --> LanguageTool – Grammar analysis
         --> Streamlit – Web UI
         --> JavaScript Diff Viewer – Visual diff between input/output

* Preprocessing & Augmentation
         --> Tasks used: gec, clarity, simplification, paraphrase
         --> Synonym-based augmentation with nlpaug for lexical variety
         --> Added task prefixes (e.g., gec:) to support multi-task training

* Modeling
         --> Fine-tuned with:
         --> Batch size: 8
         --> Epochs: 3
         --> Learning rate: 3e-5
         --> FP16 mixed precision

* Dataset: 5% of Grammarly CoEdit (filtered and augmented)

* Postprocessing
         --> LanguageTool Pre-check: Cleaned up basic errors before model input
         --> Rule-Based Corrections:
         --> Comparatives: e.g., "more simple" → "simpler"
         --> Conditionals: e.g., "If I had..., I will" → "...I would"
         --> Time Phrases: e.g., "since years" → "for years"

* Inference & Interface
         --> Streamlit-based UI with:
         --> Paragraph-wise correction
         --> Interactive grammar check with LanguageTool
         --> Before/after diff highlighting via JS

* Evaluation Metrics For essay generation:
         --> Semantic Score – SBERT cosine similarity
         --> Diversity Score – distinct-2 n-gram ratio
         --> F1 Score – Word-level overlap

Final Score – Weighted average of above
=======
# NLP-Project -WebApp
### Landing Page Overview
The landing page of the "Essay Gen & Scoring" web application serves as a clean and simple introduction to the platform’s functionality. It presents users with a bold statement about its capabilities—AI-powered essay scoring and generation—tailored for academic and English writing. The layout includes a prominent call-to-action button labeled “Try the Web App,” encouraging immediate engagement. Navigation links at the top of the page—Home, Try It, About, and Contact—help users explore the app with ease. This page is designed as the entry point that clearly communicates the purpose of the tool and invites users into a smooth, guided experience.
![image](https://github.com/user-attachments/assets/372d9924-35c3-40e4-91e7-65a4b2b24820)

### What This Application Is About
This web application is built to help users—especially English learners and IELTS candidates—improve their writing skills through two main features. First, the Essay Scoring functionality allows users to paste their written essays into the system, which then evaluates the text and returns a band score with detailed, actionable feedback. Second, the Essay Generation feature allows users to input a prompt, and the application responds with a machine-generated academic essay based on that prompt. These features are powered by state-of-the-art NLP models including T5, GPT-2, and DPO, which work together to analyze grammar, coherence, lexical resource, and task achievement.

### Current Limitations
Despite its strong foundation, the application currently has several limitations. It does not yet support real-time grammar correction, a feature that is crucial for dynamic learning and will be added in future updates. The scoring model relies on a limited dataset, which may affect the consistency and precision of the band score predictions. The interface, while functional, requires improvements in design, layout, and mobile responsiveness. Additionally, the evaluation and feedback generated by the app may sometimes feel too general, lacking the personalization and depth that more advanced systems provide.

### Future Progress
Several improvements are planned for future versions of the application. One major upgrade will be the implementation of live grammar correction, allowing users to view and accept suggestions directly within their essays. The app will also introduce interactive highlighted feedback, helping users visually understand where their writing can be improved, similar to how tools like Grammarly function. There are plans to enhance the essay generation model so that its outputs better match IELTS-specific task responses.
Another major improvement will be in the evaluation system, which will:
- Provide band-based scoring aligned with IELTS criteria,
- Offer targeted writing suggestions for improvement,
- Highlight weaknesses in grammar, cohesion, and lexical resource,
- Show a before-and-after comparison by providing an improved version of the user's essay.

Additionally, there are plans to improve UI/UX design, make the platform fully responsive across devices, and introduce user accounts, enabling users to save essays and track progress over time.

### Web App Structure
The web app is organized into a few key components to ensure an intuitive user flow:
1. Landing Page – Introduces the platform and invites users to start.
2. Essay Input Interface – Users can either:
- Paste or write their own essay, or
- Enter a prompt and let the app generate a full essay.
3. Results Page – Once an essay is submitted or generated, the app returns:
- A band score prediction,
- A breakdown of evaluation criteria,
- Specific feedback and suggestions, and
- An improved version of the essay for comparison.
This structure supports both learning and productivity, helping users either practice their writing or generate structured drafts efficiently.

>>>>>>> 8ce192cfe7bf2b36fb8c1c7bc6374bc92821a5ee
