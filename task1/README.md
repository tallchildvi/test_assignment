# Task 1 — Mountain Named Entity Recognition (NER)

A complete pipeline for training, evaluating, and running a Transformer-based Named Entity Recognition (NER) model that identifies **mountain names** in unstructured text.

The project fine-tunes **DistilBERT (`distilbert-base-uncased`)** for token classification using the BIO tagging scheme and includes everything required for dataset generation, training, inference, and evaluation.

---

## Features

- Fine-tuned **DistilBERT** model for mountain name extraction
- BIO tagging (`B-MOUNTAIN`, `I-MOUNTAIN`, `O`)
- Dataset generation pipeline
- Training notebook
- Command-line inference
- Interactive Jupyter demo
- Robust handling of challenging negative examples to reduce false positives

---

## FIles Structure

```text
task1/
├── mountain_ner_model/      # Fine-tuned model (download separately)
├── dataset_positive.json    # Positive training samples
├── dataset_negative.json    # Negative training samples
├── Dataset.ipynb            # Dataset generation & analysis
├── Train.ipynb              # Model training notebook
├── demo.ipynb               # Interactive demo & evaluation
├── inference.py             # Command-line inference
├── requirements.txt         # Project dependencies
├── .env                     # Example environment variables
└── README.md
```

---

# Model Weights

The trained model is not included in the repository because of GitHub file size limitations.

Download it here:

**Google Drive:**  
**<https://drive.google.com/file/d/1ejL3T1l7bc1hYXNogzhHSkbh---H-EJZ/view?usp=sharing>**

After downloading:

1. Extract the archive.
2. Place the `mountain_ner_model` directory inside `task1/`.

The project structure should look like:

```text
task1/
├── mountain_ner_model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
├── inference.py
└── ...
```

---

# Installation

## 1. Create a Virtual Environment


### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r task1/requirements.txt
```

---

## 3. Configure Environment Variables

The dataset generation notebook (`Dataset.ipynb`) uses the **Google Gemini API**.

Create a `.env` file inside the `task1/` directory:

```text
task1/
├── .env
├── Dataset.ipynb
└── ...
```

Add your Gemini API key:

```env
AI_API=YOUR_GEMINI_API_KEY
```

You can obtain an API key from **Google AI Studio**.

> **Note:** The `.env` file is only required for running **Dataset.ipynb**. Training, inference, and the demo notebook do not require it.

---

# Usage

## Command-Line Inference

Run inference using the default example:

```bash
python task1/inference.py
```

Or provide your own text:

```bash
python task1/inference.py --text "We are planning to climb Mount Rainier next summer."
```

---

## Interactive Demo

Launch Jupyter Notebook:

```bash
jupyter notebook task1/demo.ipynb
```

The demo notebook includes:

- automatic loading of the trained model
- model verification
- prediction visualization
- evaluation on positive and negative examples

---

# Dataset

The training data consists of two complementary datasets.

### Positive Samples

Sentences containing real mountain names with manually annotated entity spans.

Example:

```text
We climbed Mount Everest last year.
```

### Negative Samples

Hard negative examples designed to reduce false positives.

Examples include:

- software names
- hardware specifications
- IT logs
- application names
- technical documentation


---

# Model

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Task | Token Classification |
| Framework | Hugging Face Transformers |
| Labels | `B-MOUNTAIN`, `I-MOUNTAIN`, `O` |
| Tagging scheme | BIO |

---


# Future Improvements

- Multilingual mountain recognition
- ONNX export for faster inference
- Model quantization
- Larger training corpus
- Confidence score visualization
- Interactive web interface (Gradio / Streamlit)
