# Interruption Dataset and Fine-Tuned Models for Real-Time Interruption Framework

This repository contains the **custom dataset** and **fine-tuned DistilBERT models** developed as part of the Master’s thesis 
*“Developing a Framework for Semantic Disagreement Interruption Turn-taking Model”*.

The work focuses on enabling conversational agents to **interrupt users during moments of semantic disagreement** by combining real-time speech transcription, semantic similarity search, and stance classification.

---

## 📁 Repository Structure

├── dataset/ # Custom dataset developed for interruption modelling
│ ├── dataset_for_thesis_interruption.zip
│ └── ...
│
├── models/ # Fine-tuned DistilBERT checkpoints under varied configurations
│ ├── lr1.5e-5_bs8_noep_20_dr_0.2_fz_0.zip
│ ├── lr1.5e-5_bs8_noep_20_dr_0.3_fz_0.zip
│ ├── lr1.5e-5_bs16_noep_30_dr_0.2_fz_1.zip
│ └── lr2e-5_bs8_noep_20_dr_0.1_fz_0.zip
│
├── .gitattributes # LFS tracking configuration
└── .gitignore # Ignored files and temporary artifacts

---

## 🧠 Dataset Overview

The dataset extends the **Fake News Challenge (FNC-1)** corpus into a *conversational, speech-like format*, representing simulated user utterances paired with system reference statements.  
Each entry is labelled across **eight semantically nuanced classes** representing stance and interruption behaviour:

| Label | Description |
|-------|--------------|
| 0 | Agreement |
| 1 | Disagreement — Interruption |
| 2 | Discussion |
| 3 | Unrelated |
| 4 | Disagreement — No Interruption |
| 5 | Introductory / Insufficient Context |
| 6 | Topical Deviation |
| 7 | Partial Agreement |

The final dataset contains **1,471 labelled examples** distributed across these categories.

---

## Model Overview

Four DistilBERT encoder-only transformer models were fine-tuned on the dataset under varied training configurations.  
Key hyperparameters include:
- **Learning Rates:** 1.5e-5 → 2e-5  
- **Batch Sizes:** 8 → 32  
- **Dropout:** 0.1 → 0.3  
- **Frozen Layers:** 0 → 2  

Evaluation metrics:
- **Accuracy:** up to 0.66  
- **Macro F1:** up to 0.51  
- **Interruptive F1 (Label 1):** up to 0.48

---


To reproduce dataset experiments or framework integration, refer to the implementation repository:
https://github.com/shivachandra45/thesisProject





## Citation

If you use this dataset or model in your research, please cite:

Shiva Chandra Kandula. (2025). Developing a Framework for Semantic Disagreement Interruption Turn-taking Model.
Master’s Thesis, Blekinge Institute of Technology (BTH), Sweden.

---




## License

This work is made available for academic and non-commercial use only.
For inquiries or collaborations, please contact: shivachandra150@gmail.com

---


## Usage

To load a model checkpoint:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/lr1.5e-5_bs8_noep_20_dr_0.2_fz_0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
