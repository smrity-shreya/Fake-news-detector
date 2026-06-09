# FakeGuard — Fake News Detector

A full-stack machine-learning web application that classifies news articles as **REAL** or **FAKE** using TF-IDF vectorisation and Logistic Regression.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python · Flask |
| ML | Scikit-learn · Logistic Regression |
| NLP | TF-IDF · NLTK stopwords |
| Frontend | HTML · CSS · Bootstrap 5 |
| Fonts | Syne · DM Sans · DM Mono |

---

## Project Structure

```
fake-news-detector/
├── app.py               # Flask server + prediction API
├── train_model.py       # Training pipeline
├── model.pkl            # Saved model  (generated after training)
├── vectorizer.pkl       # Saved TF-IDF (generated after training)
├── requirements.txt
├── templates/
│   └── index.html       # Frontend UI
├── static/
│   └── style.css
├── dataset/
│   └── news.csv         # Your dataset (or auto-generated synthetic data)
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your own dataset
Place a CSV file at `dataset/news.csv`.  
Required columns (auto-detected): one text column (`text`, `title`, or `content`) and one label column (`label`, `class`, or `target`) with values `REAL`/`FAKE` or `1`/`0`.

If no dataset is found, a synthetic one is generated automatically.

### 3. Train the model
```bash
python train_model.py
```
This prints **Accuracy, Precision, Recall, F1** and saves `model.pkl` + `vectorizer.pkl`.

### 4. Run the server
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## API Reference

### `POST /predict`
```json
{ "text": "Your news article or headline here" }
```
Returns:
```json
{
  "prediction":        "FAKE",
  "confidence":        87.43,
  "fake_probability":  87.43,
  "real_probability":  12.57,
  "text_preview":      "…",
  "timestamp":         "2024-01-01 12:00:00"
}
```

### `GET /history`
Returns the last 20 predictions.

### `GET /health`
Server + model status check.

---

## Keyboard Shortcut
Press **Ctrl + Enter** (or **Cmd + Enter** on Mac) to submit.

---

## Notes
- The model is trained on each run; larger datasets yield better accuracy.
- For best results use a real dataset such as the [LIAR dataset](https://huggingface.co/datasets/liar) or [Fake News Net](https://github.com/KaiDMML/FakeNewsNet).
- Prediction history resets when the server restarts (in-memory only).
