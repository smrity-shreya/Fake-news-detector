import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ── NLTK setup ──────────────────────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove URLs / punctuation / numbers / stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'[^a-z\s]', ' ', text)                 # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()              # collapse whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Dataset creation (fallback when no CSV is present) ──────────────────────
def create_sample_dataset(path: str) -> None:
    """Generate a small but varied synthetic dataset."""
    print("  ⚠  No dataset found – generating synthetic training data …")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    real_headlines = [
        "Scientists discover new treatment for cancer using gene therapy",
        "NASA successfully launches new Mars rover mission",
        "Global leaders meet to discuss climate change solutions",
        "Federal Reserve raises interest rates to combat inflation",
        "New study shows benefits of Mediterranean diet for heart health",
        "Tech company announces breakthrough in quantum computing",
        "World Health Organization reports decline in malaria cases",
        "Stock markets reach record highs amid economic recovery",
        "Supreme Court rules on landmark civil rights case",
        "International Space Station receives new crew members",
        "Scientists publish research on COVID-19 vaccine effectiveness",
        "President signs infrastructure bill into law",
        "Olympic athletes break world records at summer games",
        "Central bank implements new monetary policy measures",
        "Researchers develop biodegradable plastic alternative",
        "UN report warns about accelerating biodiversity loss",
        "Major earthquake strikes coastal region causing damage",
        "Electric vehicle sales surge as battery costs decline",
        "Medical researchers announce breakthrough in Alzheimer's treatment",
        "New satellite data reveals extent of Arctic ice loss",
        "Trade agreement signed between major economic powers",
        "University study links social media use to mental health outcomes",
        "City announces new public transportation expansion plans",
        "International court issues ruling on territorial dispute",
        "Technology giant faces antitrust investigation",
        "New archaeological discovery sheds light on ancient civilization",
        "Government releases economic growth statistics for quarter",
        "Scientists confirm detection of gravitational waves",
        "Major bank announces plans to invest in renewable energy",
        "Health officials report progress in fighting antibiotic resistance",
    ]
    fake_headlines = [
        "Government secretly putting mind control chemicals in tap water",
        "Aliens have landed in Nevada and government is covering it up",
        "Bill Gates microchipping people through COVID vaccines secret plan",
        "Scientists confirm earth is actually flat NASA conspiracy revealed",
        "Secret society controls world economy from underground bunker",
        "Miracle cure for all diseases suppressed by big pharma companies",
        "5G towers causing widespread illness and death government denies",
        "Celebrity found to be reptilian shapeshifter in shocking video",
        "Ancient pyramids powered by free energy technology government hiding",
        "Moon landing was staged in Hollywood studio new evidence proves",
        "Chemtrails are weather control program targeting population",
        "Deep state operatives planning massive false flag operation soon",
        "New world order plans to eliminate 90 percent of population revealed",
        "Secret underground tunnels connect major cities for elite travel",
        "Politicians drinking adrenochrome harvested from children exclusive",
        "Sunscreen actually causes cancer big pharma conspiracy exposed",
        "Fluoride in water lowering IQ to create controllable population",
        "Ancient cure for cancer found in common household spice suppressed",
        "Satellites are actually government surveillance drones fake space",
        "Hollywood elites running human trafficking ring from pizza restaurant",
        "Quantum healing frequencies can cure any disease doctors hate this",
        "Time traveler from future warns about upcoming catastrophic event",
        "Secret cloning facility discovered under major government building",
        "Birds are not real they are government surveillance drones",
        "Magnetic nanoparticles injected during sleep to track population",
        "Ancient alien technology discovered in Antarctica kept secret",
        "Major political figure arrested for crimes media refuses to cover",
        "Hidden planet Nibiru on collision course with Earth scientists silent",
        "Government weather machine causes natural disasters on purpose",
        "Secret ingredient in fast food causes addiction and mind control",
    ]

    rows = (
        [{"text": h, "label": "REAL"} for h in real_headlines] +
        [{"text": h, "label": "FAKE"} for h in fake_headlines]
    )
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  ✓  Synthetic dataset saved → {path}  ({len(df)} rows)")


# ── Load dataset ─────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Try to auto-detect columns
    label_col = next(
        (c for c in df.columns if c.lower() in ('label', 'class', 'target', 'fake')), None
    )
    text_cols = [c for c in df.columns if c.lower() in ('text', 'title', 'content', 'article', 'news')]

    if label_col is None:
        raise ValueError(f"Cannot find a label column. Columns present: {list(df.columns)}")

    if not text_cols:
        raise ValueError(f"Cannot find a text column. Columns present: {list(df.columns)}")

    # Combine title + text if both exist
    if len(text_cols) > 1:
        df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    else:
        df['combined_text'] = df[text_cols[0]].fillna('')

    df = df[['combined_text', label_col]].rename(columns={label_col: 'label'})
    df.dropna(inplace=True)

    # Normalise labels to REAL / FAKE
    label_map = {}
    for val in df['label'].unique():
        v = str(val).strip().upper()
        if v in ('1', 'REAL', 'TRUE', 'LEGIT'):
            label_map[val] = 'REAL'
        elif v in ('0', 'FAKE', 'FALSE', 'UNRELIABLE'):
            label_map[val] = 'FAKE'
    if label_map:
        df['label'] = df['label'].map(label_map)
    df.dropna(subset=['label'], inplace=True)

    return df


# ── Main training pipeline ───────────────────────────────────────────────────
def train():
    DATASET_PATH = os.path.join('dataset', 'news.csv')

    if not os.path.exists(DATASET_PATH):
        create_sample_dataset(DATASET_PATH)

    print("\n📂  Loading dataset …")
    df = load_dataset(DATASET_PATH)
    print(f"    Rows: {len(df)}  |  Label distribution:\n{df['label'].value_counts().to_string()}\n")

    print("🧹  Cleaning text …")
    df['clean_text'] = df['combined_text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != '']

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    print("🔢  Fitting TF-IDF vectoriser …")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("🤖  Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    model.fit(X_train_vec, y_train)

    print("\n📊  Evaluation on test set:")
    y_pred = model.predict(X_test_vec)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='FAKE', zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label='FAKE', zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label='FAKE', zero_division=0)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    print("💾  Saving model and vectoriser …")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("    ✓  model.pkl  and  vectorizer.pkl  saved.\n")
    print("✅  Training complete!  Run  python app.py  to start the server.\n")


if __name__ == '__main__':
    train()
