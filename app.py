import os
import re
import pickle
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

STOP_WORDS = set(stopwords.words('english'))

# ── In-memory prediction history (resets on server restart) ─────────────────
history: list[dict] = []

# ── Load model + vectorizer ──────────────────────────────────────────────────
MODEL_PATH      = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

model      = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓  Model and vectorizer loaded successfully.")
    else:
        print("⚠  model.pkl / vectorizer.pkl not found.  Run train_model.py first.")

load_artifacts()


# ── Text cleaning (mirrors train_model.py) ───────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({'error': 'Please enter some news text.'}), 400

    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    cleaned = clean_text(news_text)
    if not cleaned:
        return jsonify({'error': 'Text became empty after cleaning. Please enter meaningful content.'}), 400

    vec  = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    classes    = list(model.classes_)
    confidence = float(max(prob)) * 100
    fake_prob  = float(prob[classes.index('FAKE')]) * 100 if 'FAKE' in classes else 0.0
    real_prob  = float(prob[classes.index('REAL')]) * 100 if 'REAL' in classes else 0.0

    result = {
        'prediction': pred,
        'confidence': round(confidence, 2),
        'fake_probability': round(fake_prob, 2),
        'real_probability': round(real_prob, 2),
        'text_preview': news_text[:120] + ('…' if len(news_text) > 120 else ''),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    history.insert(0, result)
    if len(history) > 50:
        history.pop()

    return jsonify(result)


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history[:20])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'history_count': len(history),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
