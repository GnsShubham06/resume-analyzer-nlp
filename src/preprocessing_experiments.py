"""The FIVE mandatory preprocessing experiments (Practicals 3 & 4).

Each experiment is an independent function that takes real resume text,
prints a before/after comparison and returns a dict that is saved under
results/preprocessing/.  Grouping is kept modular: EXPERIMENTS maps a name
to a function, so the professor's grouping can be changed without touching
the implementation in src/preprocessing.py.

Run all:   python -m src.preprocessing_experiments
Run one:   python -m src.preprocessing_experiments --exp 3 --resume HR_01
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from . import preprocessing as pp
from .config import DATASET_CSV, RESULTS_DIR
from .skill_extraction import extract_skills

OUT = RESULTS_DIR / "preprocessing"
OUT.mkdir(parents=True, exist_ok=True)


def _show(title, before, after, n=600):
    print(f"\n--- {title}: BEFORE ---\n{before[:n]}\n--- {title}: AFTER ---\n{after[:n]}")


# ---------------------------------------------------------------- Exp 1
def exp1_cleaning_normalization(raw: str) -> dict:
    """Aim: remove HTML tags, mojibake, special characters, extra whitespace;
    lowercase; convert number words to digits; (separately) remove numbers."""
    html_version = "<html><body><h2>" + raw.split("\n")[0] + "</h2><p>" + \
        "</p><p>".join(raw.split("\n")[1:6]) + "</p></body></html>"
    steps = {"0_raw": raw}
    steps["1_html_removed"] = pp.remove_html_tags(html_version)
    steps["2_noise_handled"] = pp.handle_noise(raw)
    steps["3_special_chars"] = pp.handle_special_characters(steps["2_noise_handled"])
    steps["4_whitespace"] = pp.normalize_whitespace(steps["3_special_chars"])
    steps["5_lowercase"] = pp.lowercase(steps["4_whitespace"])
    steps["6_numwords_to_digits"] = pp.words_to_numbers(steps["5_lowercase"])
    steps["7_numbers_removed(optional)"] = pp.remove_numbers(steps["6_numwords_to_digits"])
    steps["7b_numbers_removed_keep_years"] = pp.remove_numbers(steps["6_numwords_to_digits"], keep_years=True)
    demo = "Over twenty-five years of experience, managed fifty three staff and a $15M budget since 2009."
    nums = {"input": demo, "words_to_numbers": pp.words_to_numbers(demo),
            "remove_numbers": pp.normalize_whitespace(pp.remove_numbers(pp.words_to_numbers(demo))),
            "remove_numbers_keep_years": pp.normalize_whitespace(
                pp.remove_numbers(pp.words_to_numbers(demo), keep_years=True))}
    _show("Exp1 HTML", html_version, steps["1_html_removed"], 300)
    _show("Exp1 full cleaning", raw, steps["6_numwords_to_digits"])
    print("\nNumber handling:", json.dumps(nums, indent=1))
    stats = [{"step": k, "chars": len(v), "whitespace_tokens": len(v.split())} for k, v in steps.items()]
    return {"html_input": html_version, "steps": {k: v[:1500] for k, v in steps.items()},
            "number_demo": nums, "stats": stats}


# ---------------------------------------------------------------- Exp 2
def exp2_sentences_contractions(raw: str) -> dict:
    """Aim: resume-aware sentence boundary detection + contraction expansion."""
    text = pp.normalize_whitespace(pp.handle_special_characters(pp.handle_noise(raw)))
    naive = pp.split_sentences(text.replace("\n", " "), method="nltk")
    resume_aware = pp.split_sentences(text, method="nltk")
    spacy_sents = pp.split_sentences(text, method="spacy")
    real_contr = [s for s in resume_aware if "'" in s and any(c in s.lower() for c in ["n't", "'s", "'m", "'re", "'ve", "'ll", "'d"])]
    demo = "I'm a detail-oriented HR generalist. I've managed onboarding and I'd like a role where I can't stop learning."
    contr = {"real_resume_lines_with_apostrophes": [{"before": s, "after": pp.expand_contractions(s)}
                                                   for s in real_contr[:6]],
             "demo_sentence": {"before": demo, "after": pp.expand_contractions(demo)}}
    print(f"\nExp2 sentence counts: naive (newlines ignored)={len(naive)}, "
          f"resume-aware NLTK={len(resume_aware)}, spaCy={len(spacy_sents)}")
    for s in resume_aware[:8]:
        print("  •", s)
    print("Contractions:", json.dumps(contr, indent=1)[:900])
    return {"n_sentences": {"naive_nltk_joined": len(naive), "resume_aware_nltk": len(resume_aware),
                            "resume_aware_spacy": len(spacy_sents)},
            "first_sentences_resume_aware": resume_aware[:15],
            "first_sentences_naive": naive[:5], "contractions": contr}


# ---------------------------------------------------------------- Exp 3
def exp3_tokenization_skills(raw: str) -> dict:
    """Aim: compare tokenizers; keep multi-word skills as one token;
    extract skills and noun phrases."""
    text = pp.clean_text(raw)
    low = text.lower()
    toks = {m: pp.tokenize(low, m) for m in ("whitespace", "regex", "nltk", "spacy")}
    skill_toks = pp.skill_aware_tokenize(low)
    mwe = sorted({t for t in skill_toks if "_" in t})
    skills = extract_skills(text)
    phrases = Counter(pp.extract_noun_phrases(text)).most_common(15)
    demo = "Skilled in human resource management, data entry, machine learning and financial analysis (C++, R&D)."
    demo_out = {"nltk_word_tokenize": pp.tokenize(demo.lower()),
                "skill_aware": pp.skill_aware_tokenize(demo)}
    print("\nExp3 token counts:", {k: len(v) for k, v in toks.items()}, "skill-aware:", len(skill_toks))
    print("Multi-word skills kept as one token:", mwe)
    print("Demo:", json.dumps(demo_out))
    print("Dictionary skills:", [s["skill"] for s in skills][:20])
    print("Top noun phrases:", phrases[:10])
    return {"token_counts": {**{k: len(v) for k, v in toks.items()}, "skill_aware": len(skill_toks)},
            "first_tokens": {k: v[:25] for k, v in toks.items()},
            "skill_aware_first_tokens": skill_toks[:40], "multiword_skill_tokens": mwe,
            "demo": demo_out, "extracted_skills": skills, "top_noun_phrases": phrases}


# ---------------------------------------------------------------- Exp 4
def exp4_stopwords_stem_lemma(raw: str) -> dict:
    """Aim: stopword removal, stemming (Porter/Snowball/Lancaster) and
    lemmatization (WordNet+POS, spaCy) on resume tokens."""
    text = pp.clean_text(raw).lower()
    toks = [t for t in pp.skill_aware_tokenize(text) if t.isalpha() or "_" in t]
    no_sw = pp.remove_stopwords(toks)
    res = {"tokens": len(toks), "after_stopwords": len(no_sw),
           "stopword_reduction_pct": round(100 * (1 - len(no_sw) / max(len(toks), 1)), 1)}
    sample = no_sw[:40]
    table = pd.DataFrame({"token": sample,
                          "porter": pp.stem_tokens(sample, "porter"),
                          "snowball": pp.stem_tokens(sample, "snowball"),
                          "lancaster": pp.stem_tokens(sample, "lancaster"),
                          "wordnet_lemma": pp.lemmatize_tokens(sample, "wordnet"),
                          "spacy_lemma": pp.lemmatize_tokens(sample, "spacy")[:len(sample)]})
    table.to_csv(OUT / "exp4_stem_vs_lemma.csv", index=False)
    vocab = {"original": len(set(no_sw)), "porter": len(set(pp.stem_tokens(no_sw, "porter"))),
             "lancaster": len(set(pp.stem_tokens(no_sw, "lancaster"))),
             "wordnet_lemma": len(set(pp.lemmatize_tokens(no_sw)))}
    res["vocabulary_size"] = vocab
    print("\nExp4:", res)
    print(table.head(20).to_string())
    return {**res, "table_head": table.head(25).to_dict(orient="records")}


# ---------------------------------------------------------------- Exp 5
def exp5_noise_anonymization_script(raw: str) -> dict:
    """Aim: remove template noise, mask personal information, validate
    script / language (including line-level multilingual detection)."""
    noisy = pp.handle_noise(raw)
    synthetic_header = ("Jane Doe\njane.doe@example.com | +1 (555) 123-4567 | 42 Maple Street, "
                        "Springfield IL 62704\nwww.linkedin.com/in/janedoe\n")
    masked_demo, demo_counts = pp.anonymize(synthetic_header + raw[:300])
    masked_real, real_counts = pp.anonymize(raw)
    multilingual_demo = "Experienced HR generalist.\nSaya bekerja sebagai staf keuangan selama lima tahun di Jakarta.\nविश्लेषण कौशल"
    res = {"noise_removed_chars": len(raw) - len(noisy),
           "noise_examples_before": [l for l in raw.split("\n") if "Company Name" in l][:3],
           "noise_examples_after": [pp.handle_noise(l) for l in raw.split("\n") if "Company Name" in l][:3],
           "anonymization_real_resume_counts": real_counts,
           "anonymization_demo_input": synthetic_header + raw[:300],
           "anonymization_demo_output": masked_demo, "anonymization_demo_counts": demo_counts,
           "script_validation_real": pp.validate_script_language(pp.clean_text(raw)),
           "script_validation_multilingual_demo": pp.validate_script_language(multilingual_demo)}
    print("\nExp5 noise:", res["noise_examples_before"], "->", res["noise_examples_after"])
    print("Anonymization demo (synthetic contact header + real resume text):\n", masked_demo[:400])
    print("Script validation (real):", res["script_validation_real"])
    print("Script validation (mixed-script demo):", res["script_validation_multilingual_demo"])
    return res


EXPERIMENTS = {1: exp1_cleaning_normalization, 2: exp2_sentences_contractions,
               3: exp3_tokenization_skills, 4: exp4_stopwords_stem_lemma,
               5: exp5_noise_anonymization_script}


def preprocessing_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Per-resume effect of each preprocessing configuration."""
    rows = []
    for _, r in df.iterrows():
        raw = r["extracted_text"]
        cleaned = pp.clean_text(raw)
        toks = pp.tokenize(cleaned.lower())
        no_sw = pp.remove_stopwords([t for t in toks if t.isalnum()])
        rows.append({"resume_id": r["resume_id"], "category": r["category"],
                     "raw_chars": len(raw), "cleaned_chars": len(cleaned),
                     "raw_whitespace_tokens": len(raw.split()), "nltk_tokens": len(toks),
                     "alnum_tokens_no_stopwords": len(no_sw),
                     "classification_tokens(lemma,skill-aware)": len(str(r["classification_text"]).split()),
                     "unique_raw_tokens": len(set(raw.lower().split())),
                     "unique_classification_tokens": len(set(str(r["classification_text"]).split())),
                     "multiword_skill_tokens": sum("_" in t for t in str(r["classification_text"]).split()),
                     "skills_extracted": r["skill_count"]})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "preprocessing_comparison.csv", index=False)
    return out


def plot_comparison(cmp: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .config import PLOTS_DIR
    agg = cmp.groupby("category")[["raw_whitespace_tokens", "nltk_tokens", "alnum_tokens_no_stopwords",
                                   "classification_tokens(lemma,skill-aware)"]].mean()
    ax = agg.plot(kind="bar", figsize=(8, 4), rot=0)
    ax.set_ylabel("mean tokens per resume")
    ax.set_title("Effect of preprocessing on token counts (mean per category)", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "preprocessing_token_counts.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, default=0, help="1-5, 0 = all")
    ap.add_argument("--resume", default="HR_01", help="resume_id used as input")
    a = ap.parse_args()
    df = pd.read_csv(DATASET_CSV)
    raw = df.set_index("resume_id").loc[a.resume, "extracted_text"]
    todo = [a.exp] if a.exp else list(EXPERIMENTS)
    for k in todo:
        print(f"\n================ EXPERIMENT {k}: {EXPERIMENTS[k].__name__} (input: {a.resume}) ================")
        res = EXPERIMENTS[k](raw)
        (OUT / f"exp{k}_{a.resume}.json").write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    if not a.exp:
        cmp = preprocessing_comparison(df)
        plot_comparison(cmp)
        print("\nPreprocessing comparison (first rows):\n", cmp.head().to_string())


if __name__ == "__main__":
    main()
