"""
train_and_test.py — Fraudlock NG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains and tests RF, SVM, Naive Bayes, Logistic Regression.
Prints clean results for each model.
Saves the best model as the active production model.

Usage:
    python train_and_test.py --data dataset.csv
    python train_and_test.py --data dataset.csv --model svm
    python train_and_test.py --data dataset.csv --save-all
    python train_and_test.py --data dataset.csv --test-size 0.3
"""

import os
import re
import sys
import json
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import LinearSVC
from sklearn.naive_bayes      import MultinomialNB
from sklearn.linear_model     import LogisticRegression
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# ── Output paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ML_DIR          = os.path.join(BASE_DIR, 'ml_api', 'ml')
os.makedirs(ML_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(ML_DIR, 'sms_phishing_model_rf.pkl')
VECTORIZER_PATH = os.path.join(ML_DIR, 'tfidf_vectorizer_rf.pkl')
THRESHOLD_PATH  = os.path.join(ML_DIR, 'threshold.txt')
RESULTS_PATH    = os.path.join(ML_DIR, 'model_comparison.json')

SEP = "=" * 60

# ── Text preprocessing ─────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+',           ' URL ',       text)
    text = re.sub(r'\b\w+\.(xyz|info|cc|tk|ml|ga)\b', ' SUSPURL ',   text)
    text = re.sub(r'\b0[789][01]\d{8}\b|\+234\d{10}', '',            text)
    text = re.sub(r'[₦]\s*[\d,]+|\b[\d,]+\s*naira\b', ' MONEY ',     text, flags=re.I)
    text = re.sub(r'!{2,}',                            ' MULTIEXCL ', text)
    return ' '.join(text.split())

# ── Manual fraud features ──────────────────────────────────────────────────────
def fraud_features(texts) -> np.ndarray:
    urgency  = ['urgent','immediately','expires','suspended','blocked','verify now','act now','limited time']
    prize    = ['congratulations','winner','won','prize','lottery','selected','million','compensation']
    banking  = ['bvn','otp','pin','atm','cvv','account number','nin','password','credential','login']
    imperson = ['gtb','zenith','access bank','first bank','uba','jamb','waec','nnpc','efcc','mtn','airtel']

    rows = []
    for t in texts:
        t = t.lower() if isinstance(t, str) else ""
        letters    = [c for c in t if c.isalpha()]
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
        rows.append([
            sum(1 for w in urgency  if w in t),
            sum(1 for w in prize    if w in t),
            sum(1 for w in banking  if w in t),
            sum(1 for w in imperson if w in t),
            1 if re.search(r'https?://|www\.|bit\.ly', t) else 0,
            1 if re.search(r'\b\w+\.(xyz|info|cc|tk|ml|ga)\b', t) else 0,
            round(caps_ratio, 3),
            min(t.count('!'), 5),
        ])
    return np.array(rows)

# ── Build features ─────────────────────────────────────────────────────────────
def build_features(texts, vectorizer=None, fit=True):
    cleaned = [preprocess(t) for t in texts]
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, strip_accents='unicode',
        )
        tfidf = vectorizer.fit_transform(cleaned)
    else:
        tfidf = vectorizer.transform(cleaned)
    manual   = fraud_features(texts)
    combined = sp.hstack([tfidf, sp.csr_matrix(manual)])
    return combined, vectorizer

# ── Load dataset ───────────────────────────────────────────────────────────────
def load_data(path: str):
    print(f"\n📂  Loading: {path}")
    df = pd.read_csv(path)

    label_col = text_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['label','class','category','spam','target','type']:
            label_col = col
        if cl in ['message','text','sms','content','body','msg']:
            text_col = col

    if not label_col or not text_col:
        print(f"    Columns found: {list(df.columns)}")
        print("    ❌ Rename your columns to 'label' and 'message'")
        sys.exit(1)

    df = df[[text_col, label_col]].dropna()
    df.columns = ['message', 'label']

    spam_vals = {'spam','phishing','fraud','scam','1',1,True,'yes'}
    df['y'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() in spam_vals else 0)

    spam  = df['y'].sum()
    legit = len(df) - spam
    print(f"    Total  : {len(df)}")
    print(f"    Spam   : {spam}  ({spam/len(df)*100:.1f}%)")
    print(f"    Legit  : {legit} ({legit/len(df)*100:.1f}%)")
    return df['message'].tolist(), df['y'].tolist()

# ── Model zoo ──────────────────────────────────────────────────────────────────
def get_models():
    return {
        'svm': {
            'name':    'SVM — LinearSVC',
            'nb_safe': True,
            'model':   CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight='balanced',
                          max_iter=2000, random_state=42),
                cv=3,
            ),
        },
        'rf': {
            'name':    'Random Forest',
            'nb_safe': True,
            'model':   RandomForestClassifier(
                n_estimators=200, class_weight='balanced',
                random_state=42, n_jobs=-1,
            ),
        },
        'lr': {
            'name':    'Logistic Regression',
            'nb_safe': True,
            'model':   LogisticRegression(
                C=1.0, class_weight='balanced',
                max_iter=1000, solver='lbfgs', random_state=42,
            ),
        },
        'nb': {
            'name':    'Naive Bayes — Multinomial',
            'nb_safe': False,
            'model':   MultinomialNB(alpha=0.1),
        },
    }

# ── Find best threshold ────────────────────────────────────────────────────────
def best_threshold(model, X, y) -> float:
    probs = model.predict_proba(X)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(y, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return round(float(best_t), 2)

# ── Print results in clean format ──────────────────────────────────────────────
def print_results(name, acc, auc, cv_f1, cv_std, report, cm):
    print(f"\n{SEP}")
    print(f"{name}")
    print(SEP)
    auc_str = f"{auc:.4f}" if auc is not None else " N/A  "
    print(f"Accuracy : {acc:.4f}  |  AUC-ROC : {auc_str}")
    print(f"CV F1    : {cv_f1:.4f} ± {cv_std:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

# ── Main ───────────────────────────────────────────────────────────────────────
def run(data_path: str, model_filter=None, test_size=0.2, save_all=False):
    texts, labels = load_data(data_path)
    y = np.array(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n    Train: {len(X_train_raw)} | Test: {len(X_test_raw)} ({int(test_size*100)}% holdout)")

    print("\n  Building features...")
    X_train, vectorizer = build_features(X_train_raw, fit=True)
    X_test, _           = build_features(X_test_raw, vectorizer=vectorizer, fit=False)
    tfidf_cols = X_train.shape[1] - 8
    print(f"    Feature matrix: {X_train.shape[0]} × {X_train.shape[1]}")

    all_models = get_models()
    if model_filter:
        if model_filter not in all_models:
            print(f"❌  Unknown model '{model_filter}'. Choose: rf, svm, nb, lr")
            sys.exit(1)
        all_models = {model_filter: all_models[model_filter]}

    results = {}
    trained = {}

    print(f"\n   Training {len(all_models)} model(s)...\n")

    for key, cfg in all_models.items():
        name  = cfg['name']
        model = cfg['model']
        X_tr  = X_train[:, :tfidf_cols] if not cfg['nb_safe'] else X_train
        X_te  = X_test[:,  :tfidf_cols] if not cfg['nb_safe'] else X_test

        model.fit(X_tr, y_train)
        trained[key] = (model, X_tr, X_te)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

        acc  = accuracy_score(y_test,  y_pred)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cm   = confusion_matrix(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test,    y_pred, zero_division=0)

        report = classification_report(
            y_test, y_pred,
            target_names=['Legitimate', 'Spam'],
            digits=4
        )

        # 5-fold CV
        X_full, _ = build_features(texts, vectorizer=vectorizer, fit=False)
        if not cfg['nb_safe']:
            X_full = X_full[:, :tfidf_cols]
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1s = cross_val_score(model, X_full, y, cv=cv, scoring='f1', n_jobs=-1)

        thresh = best_threshold(model, X_te, y_test) if y_proba is not None else 0.5

        print_results(name, acc, auc, cv_f1s.mean(), cv_f1s.std(), report, cm)

        results[key] = {
            'name':      name,
            'accuracy':  round(acc,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'f1':        round(f1,   4),
            'auc':       round(auc,  4) if auc else None,
            'cv_f1':     round(float(cv_f1s.mean()), 4),
            'cv_f1_std': round(float(cv_f1s.std()),  4),
            'threshold': thresh,
        }

    # ── Summary ──────────────────────────────
    print(f"\n{SEP}")
    print("🏆  SUMMARY")
    print(SEP)
    print(f"\n  {'Model':<30} {'Accuracy':>9} {'F1':>8} {'Recall':>8} {'AUC':>8}")
    print("  " + "─" * 58)

    best_key = max(results, key=lambda k: results[k]['f1'])
    for key, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        flag    = "  ← BEST" if key == best_key else ""
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "   N/A"
        print(
            f"  {r['name']:<30}"
            f" {r['accuracy']*100:>8.2f}%"
            f" {r['f1']*100:>7.2f}%"
            f" {r['recall']*100:>7.2f}%"
            f" {auc_str:>8}"
            f"{flag}"
        )

    # ── Save best ──────────────────────────────
    print(f"\n💾  Saving: {results[best_key]['name']}")
    best_model  = trained[best_key][0]
    best_thresh = results[best_key]['threshold']

    joblib.dump(best_model,  MODEL_PATH)
    joblib.dump(vectorizer,  VECTORIZER_PATH)
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(best_thresh))
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    if save_all:
        for key, (m, _, _) in trained.items():
            p = os.path.join(ML_DIR, f'model_{key}.pkl')
            joblib.dump(m, p)
            print(f"    Saved {key.upper()} → {p}")

    print(f"\n✅  Done!")
    print(f"    Best model : {results[best_key]['name']}")
    print(f"    F1         : {results[best_key]['f1']*100:.2f}%")
    print(f"    Threshold  : {best_thresh}")
    print(f"    Saved to   : {ML_DIR}")
    print(f"\n    Restart Django to load the new model.\n")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & test Fraudlock ML models')
    parser.add_argument('--data',      required=True,  help='Path to dataset CSV')
    parser.add_argument('--model',     default=None,   choices=['rf','svm','nb','lr'])
    parser.add_argument('--test-size', default=0.2,    type=float)
    parser.add_argument('--save-all',  action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\n❌  File not found: {args.data}\n")
        sys.exit(1)

    run(
        data_path=args.data,
        model_filter=args.model,
        test_size=args.test_size,
        save_all=args.save_all,
    )