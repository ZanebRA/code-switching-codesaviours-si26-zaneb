# Code-Switching Language Identification (Roman Urdu & English)

An NLP model fine-tuned on Transformer architecture to perform token-level language identification in code-switched Roman Urdu and English text.

---

## 📌 Why This Matters
In multilingual societies like Pakistan, social media users, chat messaging platforms, and digital content frequently mix English and Roman Urdu within a single sentence. Standard Natural Language Processing (NLP) models trained on monolingual datasets struggle to process code-switched text. This project solves that problem by classifying each word in a mixed-language sentence as either **Roman Urdu** (`URD`) or **English** (`ENG`), enabling better sentiment analysis, content moderation, and search indexing for regional communication.

---

## 🔗 Project Links & Demo

* 🚀 **Model Hub:** [Fine-Tuned XLM-RoBERTa Model on Hugging Face]((https://huggingface.co/zaneb-217/code-switching-codesaviours-si26-zaneb))
* 📹 **Video Demo (Loom):** [Watch Demo Video](https://www.loom.com/share/0a7d400d27d54165910f2486d7d5c0c0)

---

## 🛠️ How It Works
1. **Dataset Creation & Annotation:** A custom dataset containing code-switched Roman Urdu and English sentences was tokenized and labeled at the word level (`URD` for Roman Urdu, `ENG` for English).
2. **Token Alignment:** Because pre-trained subword tokenizers split words into sub-tokens, subword alignment was implemented using offsets to map entity labels (`URD` / `ENG`) accurately across subword boundaries.
3. **Model Fine-Tuning:** Fine-tuned `xlm-roberta-base` for token classification using PyTorch and Hugging Face `Transformers` / `Evaluate` frameworks.
4. **Prediction & Deployment:** The model predicts language tags for all words in a sentence and is published on Hugging Face Model Hub for easy inference.

---

## 📊 Evaluation & Results

The fine-tuned XLM-RoBERTa model achieved high classification performance across both target classes on the evaluation set:

| Class Label | Language | F1-Score |
| :--- | :--- | :--- |
| **`URD`** | Roman Urdu | **0.98** |
| **`ENG`** | English | **0.97** |
| **`MIX`** | Ambiguous/Mixed | *N/A (No tokens in dataset)* |

---

## 💻 How to Run Locally

### 1. Install Dependencies

```bash
pip install transformers torch datasets evaluate
```
### 2. Run Inference in Python

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "zaneb-217/code-switching-codesaviours-si26-zaneb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

text = "Aaj meri meeting bohot important hai"
results = nlp(text)

for entity in results:
    print(f"{entity['word']}: {entity['entity_group']}")
```
## 👤 Built By
**Zaneb Rasool Ahmed**  
*Machine Learning Intern* | **Code Saviours SI-26** | 2026
