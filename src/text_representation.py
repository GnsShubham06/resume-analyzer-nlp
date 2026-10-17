"""Practical 6: text representations - N-grams, TF-IDF, Word2Vec, GloVe and
contextual (BERT-family) embeddings - and a fair comparison of all of them
as features for the same 5-fold logistic-regression classifier.

Resources
---------
* Word2Vec : trained here from scratch on the resume corpus (gensim).
* GloVe    : pretrained glove-wiki-gigaword-50 (gensim-data release).  Looked
             up in models/external/; in Colab it is fetched with
             gensim.downloader.load("glove-wiki-gigaword-50").
* Contextual: a pretrained transformer encoder, mean-pooled.  Uses the local
             RoBERTa (models/pretrained/roberta-base-spacy) when present,
             otherwise "bert-base-uncased" from the Hugging Face Hub.

Run: python -m src.text_representation
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATASET_CSV, LOCAL_ROBERTA_DIR, MODELS_DIR, PLOTS_DIR, RANDOM_SEED, RESULTS_DIR
from .evaluate import compute_metrics

OUT = RESULTS_DIR / "practical6_representations"
GLOVE_PATH = MODELS_DIR / "external" / "glove-wiki-gigaword-50.gz"


def ngram_analysis(df):
    res = {}
    for n in (1, 2, 3):
        cv = CountVectorizer(ngram_range=(n, n), min_df=2)
        X = cv.fit_transform(df["classification_text"])
        freq = np.asarray(X.sum(axis=0)).ravel()
        vocab = cv.get_feature_names_out()
        res[f"{n}-gram"] = {"vocabulary_size": int(len(vocab)),
                            "top": [(vocab[i], int(freq[i])) for i in freq.argsort()[::-1][:12]]}
    per_cat = {}
    cv = CountVectorizer(ngram_range=(2, 2), min_df=1)
    for cat, g in df.groupby("category"):
        X = cv.fit_transform(g["classification_text"])
        f = np.asarray(X.sum(axis=0)).ravel()
        v = cv.get_feature_names_out()
        per_cat[cat] = [(v[i], int(f[i])) for i in f.argsort()[::-1][:8]]
    res["top_bigrams_per_category"] = per_cat
    return res


def tfidf_analysis(df):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.9)
    X = vec.fit_transform(df["classification_text"])
    vocab = vec.get_feature_names_out()
    top = {}
    for i, rid in enumerate(df["resume_id"]):
        row = X[i].toarray().ravel()
        top[rid] = [(vocab[j], round(float(row[j]), 3)) for j in row.argsort()[::-1][:6]]
    S = cosine_similarity(X)
    return X, top, S


def train_word2vec(df):
    from gensim.models import Word2Vec
    sents = [str(t).split() for t in df["classification_text"]]
    model = Word2Vec(sents, vector_size=100, window=5, min_count=2, sg=1, epochs=60,
                     seed=RANDOM_SEED, workers=1)
    probes = ["payroll", "nurse", "engineering", "accounting", "patient", "budget", "employee", "design"]
    sims = {w: [(s, round(float(c), 3)) for s, c in model.wv.most_similar(w, topn=6)]
            for w in probes if w in model.wv}
    return model, sims


def load_glove():
    from gensim.models import KeyedVectors
    if GLOVE_PATH.exists():
        with gzip.open(GLOVE_PATH, "rt", encoding="utf-8") as f:
            first = f.readline().split()
        no_header = len(first) > 2   # gensim-data glove files are in plain GloVe text format
        return KeyedVectors.load_word2vec_format(str(GLOVE_PATH), binary=False, no_header=no_header)
    import gensim.downloader as api   # Colab / internet
    return api.load("glove-wiki-gigaword-50")


def mean_vectors(token_lists, kv, dim):
    out = np.zeros((len(token_lists), dim))
    cover = []
    for i, toks in enumerate(token_lists):
        words = [w for t in toks for w in t.split("_")]
        vecs = [kv[w] for w in words if w in kv]
        cover.append(len(vecs) / max(len(words), 1))
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out, float(np.mean(cover))


def contextual_model_name():
    return str(LOCAL_ROBERTA_DIR) if (Path(LOCAL_ROBERTA_DIR) / "config.json").exists() else "bert-base-uncased"


def contextual_embeddings(texts, model_name=None, max_length=512):
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_name = model_name or contextual_model_name()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    vecs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, truncation=True, max_length=max_length, return_tensors="pt")
            h = model(**enc).last_hidden_state[0]
            m = enc["attention_mask"][0].unsqueeze(-1)
            vecs.append(((h * m).sum(0) / m.sum()).numpy())
    return np.vstack(vecs), model_name, (tok, model)


def context_demo(tok, model):
    """Same word, different context -> different contextual vectors."""
    import torch
    pairs = [("She provided patient care in the oncology unit.", "care"),
             ("Customer care and account management for retail clients.", "care"),
             ("Managed the hospital's annual operating budget.", "budget")]
    vecs = []
    with torch.no_grad():
        for sent, word in pairs:
            enc = tok(sent, return_tensors="pt")
            h = model(**enc).last_hidden_state[0]
            ids = enc["input_ids"][0].tolist()
            target = tok(" " + word, add_special_tokens=False)["input_ids"]
            pos = next((i for i in range(len(ids)) if ids[i:i + len(target)] == target), None)
            if pos is None:
                target = tok(word, add_special_tokens=False)["input_ids"]
                pos = next(i for i in range(len(ids)) if ids[i:i + len(target)] == target)
            vecs.append(h[pos:pos + len(target)].mean(0).numpy())
    c = cosine_similarity(np.vstack(vecs))
    return {"sentences": pairs, "cos(care_medical, care_customer)": round(float(c[0, 1]), 4),
            "note": "a static embedding (Word2Vec/GloVe) would give cosine 1.0 for the same word"}


def cv_score(X, y, scale=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = make_pipeline(StandardScaler(with_mean=not hasattr(X, "tocsr")), LogisticRegression(max_iter=3000, C=1.0)) \
        if scale else LogisticRegression(max_iter=3000, C=10)
    pred = cross_val_predict(clf, X, y, cv=skf)
    return compute_metrics(y, pred)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    y = df["category"].to_numpy(dtype=object)   # plain numpy (pandas 3 uses Arrow strings)
    report = {}

    report["ngrams"] = ngram_analysis(df)
    X_tfidf, top_tfidf, S = tfidf_analysis(df)
    report["tfidf_top_terms"] = top_tfidf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(S, xticklabels=df["resume_id"], yticklabels=df["resume_id"], cmap="viridis", ax=ax)
    ax.set_title("TF-IDF cosine similarity between resumes", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tfidf_similarity_heatmap.png", dpi=150)
    plt.close(fig)

    w2v, sims = train_word2vec(df)
    report["word2vec_most_similar"] = sims
    tok_lists = [str(t).split() for t in df["classification_text"]]
    X_w2v = np.vstack([np.mean([w2v.wv[t] for t in toks if t in w2v.wv] or [np.zeros(100)], axis=0)
                       for toks in tok_lists])

    glove = load_glove()
    X_glove, cover = mean_vectors(tok_lists, glove, glove.vector_size)
    report["glove"] = {"vectors": "glove-wiki-gigaword-50 (400k words, 50-d)", "token_coverage": round(cover, 3),
                       "most_similar": {w: [(s, round(float(c), 3)) for s, c in glove.most_similar(w, topn=6)]
                                        for w in ["payroll", "nurse", "accounting", "agronomy"]}}

    try:   # needs a local model or Hugging Face Hub access
        X_ctx, ctx_name, (tok, model) = contextual_embeddings(df["transformer_text"].tolist())
        report["contextual"] = {"model": ctx_name, "pooling": "mean over first 512 tokens",
                                "dim": int(X_ctx.shape[1]), "context_demo": context_demo(tok, model)}
    except Exception as ex:
        X_ctx, ctx_name = None, None
        report["contextual"] = {"skipped": f"no transformer model available offline ({type(ex).__name__}); "
                                           "run in Colab or with internet access"}
        print("[warn] contextual embeddings skipped:", type(ex).__name__)

    comp = []
    for name, X, scale in [
        ("Bag of n-grams (1-2, counts)", CountVectorizer(ngram_range=(1, 2)).fit_transform(df["classification_text"]), False),
        ("TF-IDF (1-2 grams)", X_tfidf, False),
        ("Word2Vec (trained on corpus, mean)", X_w2v, True),
        ("GloVe 50-d (pretrained, mean)", X_glove, True),
        (f"Contextual {Path(ctx_name).name} (mean-pooled)" if ctx_name else None, X_ctx, True),
    ]:
        if X is None:
            continue
        m = cv_score(X, y, scale)
        comp.append({"representation": name, "dimensions": int(X.shape[1]), **m})
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS_DIR / "representation_comparison.csv", index=False)
    report["comparison_5fold_logreg"] = comp.to_dict(orient="records")
    (OUT / "representations_report.json").write_text(json.dumps(report, indent=1, default=str))
    ax = comp.set_index("representation")[["accuracy", "macro_f1"]].plot(kind="barh", figsize=(8, 3.6))
    ax.set_xlim(0, 1)
    ax.set_title("Representations as features (same 5-fold logistic regression)", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "representation_comparison.png", dpi=150)
    plt.close()
    print(json.dumps({k: report[k] for k in ("ngrams",)}, indent=1)[:1500])
    print("Word2Vec:", json.dumps(sims)[:800])
    print("GloVe:", json.dumps(report["glove"])[:600])
    print("Contextual:", json.dumps(report["contextual"], default=str)[:600])
    print(comp.to_string())


if __name__ == "__main__":
    main()
