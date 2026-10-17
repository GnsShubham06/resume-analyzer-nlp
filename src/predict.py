"""End-to-end analysis of ONE resume (used by the CLI and the Streamlit app).

    python -m src.predict path/to/resume.pdf [--out analysis.json]

Steps: text extraction -> cleaning/anonymization -> skill & information
extraction (Task B) -> NER -> domain classification (Task A) with the
fine-tuned transformer (if trained) and the TF-IDF baseline.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import FINETUNED_MODEL_DIR, MODELS_DIR, RESULTS_DIR
from .pdf_extraction import extract_text
from .preprocessing import anonymize, clean_text, preprocess, validate_script_language
from .skill_extraction import analyze_resume

CLASSIFIER_WARNING = (
    "PROTOTYPE CLASSIFIER: trained on 60 resumes (12 per class) from a 100-resume dataset in which 82 resumes are "
    "synthetic. Scores measured on 20 test resumes do not prove accuracy on real-world resumes; use the "
    "predictions as a demonstration of the pipeline, not as a decision tool.")


def find_transformer_dir() -> Path | None:
    """The fine-tuned checkpoint saved by src/train_transformer.py (best validation epoch)."""
    d = Path(FINETUNED_MODEL_DIR)
    return d if (d / "label_map.json").exists() and (d / "config.json").exists() else None


@lru_cache(maxsize=2)
def load_transformer(model_dir: str):
    """Load the SAVED fine-tuned model (not the pretrained one) and its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    meta = json.loads((Path(model_dir) / "label_map.json").read_text())
    return tok, model, meta


def predict_transformer(text: str, model_dir: str) -> dict:
    """Same input format as in training: first `max_length` tokens of the cleaned resume."""
    import torch
    tok, model, meta = load_transformer(str(model_dir))
    enc = tok(text, truncation=True, max_length=meta.get("max_length", 512), return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    probs = torch.softmax(logits, -1)[0].numpy()
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    order = np.argsort(probs)[::-1]
    return {"model": f"fine-tuned transformer ({Path(model_dir).name}, base: {Path(meta.get('base_model_used', '?')).name}, "
                     f"best epoch {meta.get('best_epoch')})",
            "predicted": id2label[int(order[0])],
            "probabilities": {id2label[int(i)]: round(float(probs[i]), 4) for i in order}}


def predict_baseline(text_raw: str) -> dict | None:
    path = MODELS_DIR / "baseline_tfidf_logreg.joblib"
    if not path.exists():
        return None
    import joblib
    pipe = joblib.load(path)
    toks = preprocess(text_raw, "classification")["text"]
    p = pipe.predict_proba([toks])[0]
    order = np.argsort(p)[::-1]
    return {"model": "TF-IDF + Logistic Regression", "predicted": pipe.classes_[order[0]],
            "probabilities": {pipe.classes_[i]: round(float(p[i]), 4) for i in order}}


def analyze_file(path: str | Path, use_transformer: bool = True) -> dict:
    ex = extract_text(path)
    if ex["status"] != "ok":
        return {"file": str(path), "status": ex["status"], "error": ex["error"]}
    return analyze_text(ex["text"], file=str(path), n_pages=ex["n_pages"], use_transformer=use_transformer)


def analyze_text(raw: str, file: str = "<text>", n_pages: int | None = None, use_transformer: bool = True) -> dict:
    from .ner import pretrained_entities, rule_based_entities
    anon, anon_counts = anonymize(raw)
    cleaned = clean_text(anon)
    info = analyze_resume(cleaned)
    out = {"file": file, "status": "ok", "n_pages": n_pages, "n_characters": len(raw),
           "anonymization": anon_counts, "language_validation": validate_script_language(cleaned),
           "skills": info["skills"], "education": info["education"], "job_titles": info["job_titles"],
           "organizations": info["organizations"], "experience": info["experience"],
           "lexicon_domain_scores": info["label_check"]["domain_scores"],
           "entities": {"pretrained_spacy": pretrained_entities(cleaned)[:60],
                        "rule_based": [{k: e[k] for k in ("text", "label", "source")}
                                       for e in rule_based_entities(cleaned)][:80]},
           "classification": {"warning": CLASSIFIER_WARNING}}
    tdir = find_transformer_dir()
    if use_transformer and tdir is not None:
        try:
            out["classification"]["transformer"] = predict_transformer(preprocess(anon, "transformer")["text"], tdir)
        except Exception as e:  # e.g. torch missing
            out["classification"]["transformer_error"] = str(e)
    else:
        out["classification"]["transformer"] = None
        out["classification"]["transformer_note"] = ("no fine-tuned checkpoint in models/transformer_finetuned/ - "
                                                     "run: python -m src.train_transformer")
    out["classification"]["baseline"] = predict_baseline(anon)
    out["cleaned_text_preview"] = cleaned[:1500]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_transformer", action="store_true")
    a = ap.parse_args()
    res = analyze_file(a.path, use_transformer=not a.no_transformer)
    out = Path(a.out) if a.out else RESULTS_DIR / f"analysis_{Path(a.path).stem}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    c = res.get("classification", {})
    print("File:", res["file"])
    print("Skills:", ", ".join(s["skill"] for s in res.get("skills", [])[:20]))
    print("Job titles:", res.get("job_titles"))
    print("Education:", res.get("education"))
    if c.get("transformer"):
        print("Transformer prediction:", c["transformer"]["predicted"], c["transformer"]["probabilities"])
    if c.get("baseline"):
        print("Baseline prediction:   ", c["baseline"]["predicted"], c["baseline"]["probabilities"])
    print("NOTE:", CLASSIFIER_WARNING)
    print("Saved", out)


if __name__ == "__main__":
    main()
