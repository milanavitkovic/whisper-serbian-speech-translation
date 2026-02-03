# Serbian → English Speech Translation (Whisper)

Implementation of a **speech-to-text translation system**
that translates **Serbian audio directly into English text** using OpenAI’s Whisper model.

The project focuses on **data engineering, model fine-tuning, and robust evaluation**
for a low-resource speech translation scenario.

---

## Problem

Build a system that:
- takes **Serbian speech audio** as input
- generates **English text** as output
- avoids a cascaded ASR → MT pipeline
- works reliably on real-world news speech data

---

## Solution Overview

- Fine-tuned **Whisper-small** for **direct speech translation (ST)**
- Constructed a **high-quality parallel Serbian–English corpus**
- Trained on **multiple datasets** to improve robustness
- Evaluated on a **held-out, domain-consistent test set**

---

## Data

### Training
- **Južne vesti (train split)**
- **ParlaSpeech (train split)**  
→ combined into a single training dataset

### Validation & Testing
- **Južne vesti (validation & test splits only)**  
→ ensures fair evaluation on unseen, in-domain data

---

## Parallel Corpus Construction

Since English translations were not available:

- Serbian transcriptions were translated automatically using **GPT-4o**
- All translations were **manually reviewed and corrected**
- Result: a **clean, high-quality parallel Serbian–English dataset**

This step significantly improved translation quality compared to raw automatic translations.

---

## Model & Training

- **Model**: `openai/whisper-small`
- **Task**: Speech Translation (sr → en)
- **Framework**: HuggingFace Transformers
- **Training setup**:
  - mixed-domain training data
  - gradient accumulation
  - FP16 training
  - checkpointing and best-model selection

The model is explicitly forced into translation mode:

```python
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="translate"
)
```

---

## Evaluation & Testing

- **Validation Set (Južne Vesti validation split)**  
  - Used during training to monitor model performance and select the best epoch  
  - Metric used: **Word Error Rate (WER)**  
    → model checkpoint with lowest WER on validation set is saved as the best model

- **Test Set (Južne Vesti test split)**  
  - Used for final evaluation after training  
  - Metrics computed: **WER**, **BLEU**, **METEOR**  
    → provides a comprehensive assessment of transcription accuracy and translation quality


## Model Fine-tuning Comparison

In addition to fine-tuning **Whisper-small**, we also performed fine-tuning on **Whisper-medium** to compare performance.  

- **Models evaluated**:
  - `whisper-small` (fine-tuned)
  - `whisper-medium` (fine-tuned)
  - Baseline models without fine-tuning for reference

- **Comparison Metrics**:
  - **WER** – lower is better (transcription accuracy)
  - **BLEU** – higher is better (translation quality)
  - **METEOR** – higher is better (semantic and lexical precision)

- **Findings**:
  - Fine-tuning significantly improved performance for both models compared to baselines
  - **Whisper-medium** generally achieved slightly better metrics than **Whisper-small**, but required more computational resources
  - This comparison highlights the trade-off between model size, performance, and training cost

