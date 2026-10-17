"""Semantic analysis with WordNet (compulsory in the project guidelines).

On real resume vocabulary we show:
1. Word senses (synsets) and glosses - lexical ambiguity of resume words.
2. Synonyms (lemmas) and hypernyms ("is-a" chains) of job-title and skill nouns.
3. Wu-Palmer and path similarity between job-title nouns - which titles are
   semantically close (nurse/physician) and which are not (nurse/accountant).
4. Word-sense disambiguation with the Lesk algorithm on real resume sentences.
5. A semantic domain profile per resume: every noun is mapped to its WordNet
   synsets and the best Wu-Palmer similarity to five domain "anchor" concepts
   is accumulated.  This is compared with the folder label (an independent,
   knowledge-based check that needs no training data).

Run: python -m src.wordnet_semantics
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from .config import CATEGORIES, DATASET_CSV, RESULTS_DIR
from .nlp_utils import ensure_nltk
from .preprocessing import clean_text, split_sentences

OUT = RESULTS_DIR / "wordnet_semantics"

# one or more anchor synsets per domain (WordNet sense names)
ANCHORS = {
    "Engineering": ["engineering.n.02", "machine.n.01", "software.n.01"],
    "HR": ["personnel.n.01", "employment.n.02", "recruitment.n.01"],
    "Finance": ["finance.n.01", "accounting.n.02", "money.n.01"],
    "Healthcare": ["medicine.n.02", "patient.n.01", "health_care.n.01"],
    "Agriculture": ["agriculture.n.02", "plant.n.02", "crop.n.01"],
}
TITLE_WORDS = ["engineer", "technician", "accountant", "auditor", "nurse", "physician", "recruiter",
               "manager", "farmer", "biologist"]
PROBE_WORDS = ["payroll", "audit", "patient", "crop", "welding", "recruitment", "budget", "pesticide"]


def _wn():
    ensure_nltk()
    from nltk.corpus import wordnet as wn
    return wn


def senses(word: str, k: int = 4) -> list[dict]:
    wn = _wn()
    return [{"synset": s.name(), "definition": s.definition()} for s in wn.synsets(word)[:k]]


def synonyms_hypernyms(word: str) -> dict:
    wn = _wn()
    ss = wn.synsets(word, pos=wn.NOUN)
    if not ss:
        return {"word": word, "synonyms": [], "hypernym_path": []}
    s = ss[0]
    syn = sorted({l.name().replace("_", " ") for x in ss[:3] for l in x.lemmas()} - {word})
    path = [p.name().split(".")[0] for p in s.hypernym_paths()[0]]
    return {"word": word, "sense": s.name(), "synonyms": syn[:8], "hypernym_path": path}


def similarity_matrix(words: list[str]) -> pd.DataFrame:
    wn = _wn()
    syn = {w: wn.synsets(w, pos=wn.NOUN)[0] for w in words}
    return pd.DataFrame([[round(syn[a].wup_similarity(syn[b]) or 0, 2) for b in words] for a in words],
                        index=words, columns=words)


def lesk_examples(df: pd.DataFrame) -> list[dict]:
    """Disambiguate ambiguous words in real resume sentences with simplified Lesk."""
    from nltk import word_tokenize
    from nltk.wsd import lesk
    wn = _wn()
    targets = {"plant", "care", "account", "record", "service", "charge"}
    out = []
    for _, r in df.iterrows():
        for s in split_sentences(clean_text(r["anonymized_text"])):
            toks = word_tokenize(s.lower())
            hit = targets.intersection(toks)
            if hit and 6 <= len(toks) <= 30:
                w = sorted(hit)[0]
                sense = lesk(toks, w, pos=wn.NOUN)
                if sense:
                    out.append({"resume_id": r["resume_id"], "word": w, "sentence": s,
                                "sense": sense.name(), "definition": sense.definition()})
                    targets.discard(w)
        if not targets:
            break
    return out


def semantic_domain_profile(text: str, max_nouns: int = 300) -> dict:
    """Accumulate the best Wu-Palmer similarity of each noun to each domain anchor."""
    import spacy  # noqa: F401
    from .nlp_utils import get_spacy
    wn = _wn()
    anchors = {d: [wn.synset(a) for a in lst] for d, lst in ANCHORS.items()}
    nouns = Counter(t.lemma_.lower() for t in get_spacy()(text[:20000])
                    if t.pos_ == "NOUN" and t.is_alpha and len(t) > 3)
    scores = {d: 0.0 for d in ANCHORS}
    for noun, cnt in nouns.most_common(max_nouns):
        ss = wn.synsets(noun, pos=wn.NOUN)[:2]
        if not ss:
            continue
        for d, an in anchors.items():
            best = max((s.wup_similarity(a) or 0) for s in ss for a in an)
            if best >= 0.5:                # only clearly related nouns contribute
                scores[d] += best * cnt
    tot = sum(scores.values()) or 1
    return {d: round(v / tot, 3) for d, v in scores.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    rep = {"senses": {w: senses(w) for w in ["plant", "care", "account", "engineer", "record"]},
           "synonyms_hypernyms": [synonyms_hypernyms(w) for w in PROBE_WORDS + ["nurse", "accountant"]]}
    sim = similarity_matrix(TITLE_WORDS)
    sim.to_csv(OUT / "job_title_wup_similarity.csv")
    rep["job_title_similarity"] = sim.to_dict()
    rep["lesk"] = lesk_examples(df)
    rows = []
    for _, r in df.iterrows():
        prof = semantic_domain_profile(r["cleaned_text"])
        best = max(prof, key=prof.get)
        rows.append({"resume_id": r["resume_id"], "folder_label": r["category"], "wordnet_domain": best,
                     **{f"sem_{d}": prof[d] for d in CATEGORIES}, "agrees": best == r["category"]})
    prof_df = pd.DataFrame(rows)
    prof_df.to_csv(RESULTS_DIR / "wordnet_semantic_profile.csv", index=False)
    rep["semantic_profile_agreement"] = {
        "agree_all": int(prof_df["agrees"].sum()), "n": len(prof_df),
        "agree_unflagged": int(prof_df[~df["review_required"].astype(bool)]["agrees"].sum()),
        "n_unflagged": int((~df["review_required"].astype(bool)).sum())}
    (OUT / "wordnet_report.json").write_text(json.dumps(rep, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from .config import PLOTS_DIR
    fig, ax = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, ax=ax, annot_kws={"size": 8})
    ax.set_title("WordNet Wu-Palmer similarity between job-title nouns", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "wordnet_title_similarity.png", dpi=150)
    plt.close(fig)

    print("Senses:", json.dumps(rep["senses"]["plant"], indent=1)[:500])
    for x in rep["synonyms_hypernyms"][:5]:
        print(x)
    print(sim.to_string())
    for x in rep["lesk"]:
        print(f"[{x['resume_id']}] {x['word']} -> {x['sense']}: {x['definition']}\n   {x['sentence'][:110]}")
    print(prof_df.to_string())
    print(rep["semantic_profile_agreement"])


if __name__ == "__main__":
    main()
