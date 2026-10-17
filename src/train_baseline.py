"""Baseline classifier: TF-IDF (word uni+bi-grams) + Logistic Regression.

Fair comparison with the transformer:
  * the SAME resume-level split (data/processed/splits.csv) and label names;
  * trained on the 60 TRAIN resumes only (TF-IDF vocabulary and IDF weights are
    fitted inside the pipeline on training data only - no leakage);
  * no hyper-parameter tuning, so the validation resumes are only reported;
  * evaluated once on the 20 TEST resumes.
Extra (baseline only, cheap): stratified 5-fold cross-validation over all 100
resumes, to show how much the score varies with the split.

Run: python -m src.train_baseline
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from .data_split import cv_folds, get_splits, load_dataset
from .evaluate import compute_metrics, plot_confusion, report_frame, save_json

TEXT_COL = "classification_text"     # lowercased, lemmatised, stopword-free, skill-aware tokens
BASELINE_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(C=10, max_iter=2000, random_state=RANDOM_SEED)),
    ])


def predict_frame(pipe: Pipeline, frame: pd.DataFrame, split: str) -> pd.DataFrame:
    proba = pipe.predict_proba(frame[TEXT_COL])
    out = pd.DataFrame({"resume_id": frame["resume_id"], "source_type": frame.get("source_type"),
                        "true": frame["category"], "predicted": pipe.classes_[proba.argmax(1)],
                        "confidence": proba.max(1).round(4), "split": split})
    for j, c in enumerate(pipe.classes_):
        out[f"p_{c}"] = proba[:, j].round(4)
    return out


def top_features(pipe: Pipeline, k: int = 10) -> dict:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_
    return {c: vocab[np.argsort(coef[i])[::-1][:k]].tolist() for i, c in enumerate(pipe.named_steps["clf"].classes_)}


def main():
    df = load_dataset()
    df["split"] = df["resume_id"].map(get_splits(df).set_index("resume_id")["split"])
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))

    pipe = make_pipeline().fit(train[TEXT_COL], train["category"])     # fitted on TRAIN only
    joblib.dump(pipe, BASELINE_PATH)
    preds = pd.concat([predict_frame(pipe, test, "test"), predict_frame(pipe, val, "val")])
    preds.to_csv(RESULTS_DIR / "predictions_baseline.csv", index=False)
    t = preds[preds["split"] == "test"]
    report_frame(t["true"], t["predicted"]).to_csv(RESULTS_DIR / "classification_report_baseline_test.csv")
    plot_confusion(t["true"], t["predicted"], f"TF-IDF + LogReg - test resumes (n={len(t)})",
                   "confusion_baseline_test.png")
    results = {"train": compute_metrics(train["category"], pipe.predict(train[TEXT_COL])),
               "val": compute_metrics(val["category"], pipe.predict(val[TEXT_COL])),
               "test": compute_metrics(t["true"], t["predicted"])}

    # extra: 5-fold CV over all resumes (fresh pipeline per fold)
    rows = []
    for k, (tr, te) in enumerate(cv_folds(df)):
        p = make_pipeline().fit(df.iloc[tr][TEXT_COL], df.iloc[tr]["category"])
        rows.append(pd.DataFrame({"resume_id": df.iloc[te]["resume_id"].values, "true": df.iloc[te]["category"].values,
                                  "predicted": p.predict(df.iloc[te][TEXT_COL]), "fold": k}))
    cv = pd.concat(rows)
    cv.to_csv(RESULTS_DIR / "predictions_baseline_cv.csv", index=False)
    results["cv_5fold_all_100"] = compute_metrics(cv["true"], cv["predicted"])

    feats = top_features(pipe)
    (RESULTS_DIR / "baseline_top_features.json").write_text(json.dumps(feats, indent=1))
    save_json({"model": "TF-IDF + Logistic Regression", "text_column": TEXT_COL,
               "vectorizer": "word uni+bi-grams, sublinear tf, max_df=0.9", "classifier": "LogisticRegression(C=10)",
               "trained_on": f"{len(train)} train resumes", "n_features": int(len(pipe.named_steps['tfidf'].vocabulary_)),
               "results": results}, RESULTS_DIR / "metrics_baseline.json")
    print(json.dumps(results, indent=1))
    print("saved", BASELINE_PATH)


if __name__ == "__main__":
    main()
