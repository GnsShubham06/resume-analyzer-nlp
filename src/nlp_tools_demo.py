"""Practical 1: NLP applications and tools - executable mini-demos of NLTK,
spaCy and Hugging Face Transformers on a real resume sentence, plus a status
check for AllenNLP and a comparison of cloud NLP platforms (not called: no
paid accounts or API keys are required by this project).

Run: python -m src.nlp_tools_demo
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATASET_CSV, LOCAL_ROBERTA_DIR, RESULTS_DIR
from .nlp_utils import ensure_nltk, get_spacy

OUT = RESULTS_DIR / "practical1_tools"

CLOUD_PLATFORMS = [
    {"platform": "Google Cloud Natural Language API", "features": "entities, sentiment, syntax, content classification",
     "custom_models": "AutoML / Vertex AI", "access": "REST/gRPC, API key or service account (billed)"},
    {"platform": "AWS Comprehend", "features": "entities, key phrases, sentiment, PII detection, language detection",
     "custom_models": "custom classifier & custom entity recognizer", "access": "boto3 SDK, IAM credentials (billed)"},
    {"platform": "Azure AI Language (Text Analytics)", "features": "NER, PII redaction, key phrases, sentiment, language",
     "custom_models": "custom text classification / custom NER", "access": "REST/SDK, endpoint + key (billed)"},
]


def demo(sentence: str) -> dict:
    out = {"sentence": sentence}
    ensure_nltk()
    import nltk
    toks = nltk.word_tokenize(sentence)
    out["nltk"] = {"version": nltk.__version__, "tokens": toks, "pos": nltk.pos_tag(toks)}

    nlp = get_spacy()
    import spacy
    doc = nlp(sentence)
    out["spacy"] = {"version": spacy.__version__, "model": nlp.meta.get("name"),
                    "lemmas": [(t.text, t.lemma_) for t in doc],
                    "entities": [(e.text, e.label_) for e in doc.ents],
                    "noun_chunks": [c.text for c in doc.noun_chunks]}

    try:
        import transformers
        from transformers import pipeline
        try:   # needs Hugging Face Hub access (e.g. Colab)
            clf = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
            z = clf(sentence, candidate_labels=["engineering", "human resources", "finance", "healthcare",
                                                "agriculture"])
            out["transformers"] = {"version": transformers.__version__, "task": "zero-shot classification",
                                   "labels": z["labels"], "scores": [round(s, 3) for s in z["scores"]]}
        except Exception as hub_err:
            model = str(LOCAL_ROBERTA_DIR) if (Path(LOCAL_ROBERTA_DIR) / "config.json").exists() else None
            if model is None:
                raise hub_err
            fe = pipeline("feature-extraction", model=model, tokenizer=model)
            vec = fe(sentence)[0]
            out["transformers"] = {"version": transformers.__version__, "task": "feature-extraction (local RoBERTa)",
                                   "n_tokens": len(vec), "dim": len(vec[0]),
                                   "note": f"Hub not reachable ({type(hub_err).__name__}); used local model"}
    except Exception as e:
        out["transformers"] = {"error": str(e)}

    try:
        import allennlp  # noqa: F401
        out["allennlp"] = "installed"
    except Exception as e:
        out["allennlp"] = (f"not available ({type(e).__name__}). AllenNLP was archived by AI2 in 2022 and "
                           "pins old torch/Python versions, so it is documented but not executed here.")
    out["cloud_platforms"] = CLOUD_PLATFORMS
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    text = df.set_index("resume_id").loc["ENG_05", "cleaned_text"]
    sentence = "Experience with advanced material selection for rapid prototyping, advanced manufacturing, " \
               "welding and 3D printing."
    if sentence not in text:
        sentence = text.split("\n")[5]
    res = demo(sentence)
    (OUT / "tools_demo.json").write_text(json.dumps(res, indent=1, default=str))
    pd.DataFrame(CLOUD_PLATFORMS).to_csv(OUT / "cloud_platforms.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "cloud_platforms"}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
