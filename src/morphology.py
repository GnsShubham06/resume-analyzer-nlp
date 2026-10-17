"""Practical 7: morphological analysis of English resume sentences.

For every word we show: surface form, lemma (root), POS, spaCy morphological
features (Number, Tense, VerbForm, Degree, ...), Porter stem, and a simple
affix (prefix/suffix) segmentation that separates inflectional suffixes
(-s, -ed, -ing) from derivational ones (-ment, -tion, -er, -ity, ...).

Run: python -m src.morphology
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from .config import DATASET_CSV, RESULTS_DIR
from .nlp_utils import ensure_nltk, get_spacy
from .preprocessing import clean_text, split_sentences

OUT = RESULTS_DIR / "practical7_morphology"

PREFIXES = ["multi", "inter", "trans", "under", "over", "super", "micro", "non", "pre", "re", "un",
            "dis", "mis", "co", "sub", "in", "im"]
DERIV_SUFFIXES = ["ization", "ational", "ation", "ition", "ment", "ness", "ance", "ence", "ship", "ity",
                  "ism", "ist", "ive", "able", "ible", "ful", "less", "ous", "ical", "al", "ize", "ise",
                  "ify", "er", "or", "ly"]
INFL_SUFFIXES = ["ing", "ed", "es", "s"]


def segment(word: str, lemma: str) -> dict:
    w = word.lower()
    infl = next((s for s in INFL_SUFFIXES if w.endswith(s) and w != lemma.lower() and len(w) > len(s) + 2), "")
    base = w[: -len(infl)] if infl else w
    deriv = next((s for s in DERIV_SUFFIXES if base.endswith(s) and len(base) > len(s) + 3), "")
    # accept a prefix only if what remains is itself an English word
    # ("multitasker" -> multi + tasker, but NOT "communicator" -> co + mmunicator)
    from nltk.corpus import wordnet as wn
    pre = next((p for p in PREFIXES if w.startswith(p) and len(w) > len(p) + 3
                and (wn.synsets(lemma.lower()[len(p):]) or wn.synsets(base[len(p):]))), "")
    parts = ([pre + "-"] if pre else []) + [base[len(pre): len(base) - len(deriv)] if deriv else base[len(pre):]] \
        + (["-" + deriv] if deriv else []) + (["-" + infl] if infl else [])
    return {"prefix": pre, "derivational_suffix": deriv, "inflectional_suffix": infl,
            "segmentation": " + ".join(parts)}


def analyse_sentence(sentence: str) -> pd.DataFrame:
    ensure_nltk()
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    rows = []
    for t in get_spacy()(sentence):
        if t.is_punct or t.is_space:
            continue
        rows.append({"word": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
                     "morph_features": str(t.morph) or "-", "porter_stem": ps.stem(t.text.lower()),
                     **segment(t.text, t.lemma_)})
    return pd.DataFrame(rows)


def corpus_statistics(df: pd.DataFrame) -> dict:
    nlp = get_spacy()
    feats, deriv, infl, verb_forms = Counter(), Counter(), Counter(), Counter()
    lemma_families = {}
    for text in df["cleaned_text"]:
        doc = nlp(text[:20000])
        for t in doc:
            if not t.is_alpha:
                continue
            for f in t.morph:
                feats[f] += 1
            if t.pos_ == "VERB":
                verb_forms[f"{t.tag_}"] += 1
            seg = segment(t.text, t.lemma_)
            if seg["derivational_suffix"]:
                deriv["-" + seg["derivational_suffix"]] += 1
            if seg["inflectional_suffix"]:
                infl["-" + seg["inflectional_suffix"]] += 1
            lemma_families.setdefault(t.lemma_.lower(), set()).add(t.text.lower())
    rich = sorted(lemma_families.items(), key=lambda kv: -len(kv[1]))[:12]
    return {"top_morph_features": feats.most_common(15), "verb_tags": verb_forms.most_common(),
            "derivational_suffixes": deriv.most_common(12), "inflectional_suffixes": infl.most_common(),
            "largest_lemma_families": [(k, sorted(v)) for k, v in rich]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    examples = []
    for rid in ["HR_02", "ENG_04", "FIN_03", "HLT_01", "AGR_02"]:
        text = clean_text(df.set_index("resume_id").loc[rid, "anonymized_text"])
        sent = next(s for s in split_sentences(text) if 6 <= len(s.split()) <= 16 and "[" not in s)
        tab = analyse_sentence(sent)
        tab.to_csv(OUT / f"morphology_{rid}.csv", index=False)
        examples.append({"resume_id": rid, "sentence": sent, "analysis": tab.to_dict(orient="records")})
        print(f"\n=== [{rid}] {sent}\n{tab.to_string(index=False)}")
    stats = corpus_statistics(df)
    (OUT / "morphology_report.json").write_text(json.dumps({"examples": examples, "corpus": stats},
                                                           indent=1, default=str))
    print("\nCorpus statistics:", json.dumps(stats, indent=1, default=str)[:2000])


if __name__ == "__main__":
    main()
