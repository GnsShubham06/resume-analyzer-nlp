"""Practical 9: Named Entity Recognition for resumes.

Three clearly separated sources - nothing here is a custom-TRAINED NER model:
1. PRETRAINED statistical NER : spaCy en_core_web_sm (CNN, OntoNotes labels:
   ORG, PERSON, GPE, DATE, ...).
2. PRETRAINED transformer NER : spaCy en_core_web_trf (RoBERTa-based) when it
   is installed; in Colab the Hugging Face pipeline "dslim/bert-base-NER" can
   be used instead (`hf_transformer_ner`).
3. RULE-BASED extraction      : spaCy EntityRuler patterns built from our
   lexicons for SKILL, DEGREE, QUALIFICATION and JOB_TITLE, plus a date-range
   regex for EXPERIENCE.  Pretrained models do not know these labels.

Run: python -m src.ner
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

import pandas as pd

from .config import DATASET_CSV, RESULTS_DIR
from .nlp_utils import get_spacy
from .skill_extraction import _RANGE_RE
from .skill_lexicon import DEGREE_PATTERNS, JOB_TITLE_HEADWORDS, QUALIFICATION_KEYWORDS, all_skill_terms

OUT = RESULTS_DIR / "practical9_ner"
PRETRAINED_KEEP = {"ORG", "GPE", "DATE", "PERSON", "NORP", "CARDINAL", "MONEY", "PERCENT", "PRODUCT",
                   "LAW", "FAC", "LOC", "WORK_OF_ART"}


@lru_cache(maxsize=1)
def rule_nlp():
    """A blank tokenizer + EntityRuler with resume-specific patterns."""
    import spacy
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns = []
    for term in all_skill_terms():
        if len(term) > 3:   # short acronyms handled by exact-case patterns below
            patterns.append({"label": "SKILL", "pattern": term})
        else:
            patterns.append({"label": "SKILL", "pattern": [{"TEXT": term.upper()}]})
    for q in QUALIFICATION_KEYWORDS:
        patterns.append({"label": "QUALIFICATION", "pattern": [{"TEXT": q.upper()}]})
    deg_words = ["bachelor", "master", "masters", "associate", "doctorate", "phd", "mba", "diploma", "b.s.", "m.s.",
                 "b.a.", "m.a.", "bsc", "msc", "b.e.", "b.tech", "m.tech"]
    patterns.append({"label": "DEGREE", "pattern": [{"LOWER": {"IN": deg_words}}, {"LOWER": {"IN": ["'s", "of", "in"]}, "OP": "*"},
                                                     {"LOWER": {"IN": ["science", "arts", "business", "engineering",
                                                                       "administration", "social", "work", "nursing",
                                                                       "accounting", "finance"]}, "OP": "*"}]})
    title_mods = {"senior", "sr", "junior", "jr", "assistant", "associate", "lead", "chief", "staff", "hr", "finance",
                  "financial", "project", "engineering", "mechanical", "electrical", "software", "sales", "marketing",
                  "operations", "office", "nurse", "clinical", "account", "accounts", "payroll", "benefits",
                  "research", "lab", "test", "quality", "maintenance", "administrative", "general", "regional",
                  "human", "resources", "resource", "equipment", "services", "service", "budget", "tax", "patient",
                  "access", "data", "entry", "employee", "relations", "plant", "protection", "education", "program"}
    patterns.append({"label": "JOB_TITLE", "pattern": [{"LOWER": {"IN": sorted(title_mods)}, "OP": "+"},
                                                        {"LOWER": {"IN": JOB_TITLE_HEADWORDS}}]})
    ruler.add_patterns(patterns)
    return nlp


def rule_based_entities(text: str) -> list[dict]:
    from spacy.util import filter_spans
    doc = rule_nlp()(text)
    ents = [{"text": e.text, "label": e.label_, "source": "rule-based (EntityRuler)",
             "start": e.start_char} for e in filter_spans(list(doc.ents))]
    for m in _RANGE_RE.finditer(text):
        ents.append({"text": m.group(0), "label": "EXPERIENCE_DATES", "source": "rule-based (regex)",
                     "start": m.start()})
    for p in DEGREE_PATTERNS[:4]:
        for m in re.finditer(p, text, re.I):
            if m.group(0)[:1].isupper():
                ents.append({"text": m.group(0).strip(), "label": "DEGREE", "source": "rule-based (regex)",
                             "start": m.start()})
    return ents


def pretrained_entities(text: str, model: str = "en_core_web_sm") -> list[dict]:
    nlp = get_spacy(model)
    out = []
    lines = [l for l in text.split("\n") if l.strip()]
    for doc in nlp.pipe(lines, batch_size=32):
        for e in doc.ents:
            if e.label_ in PRETRAINED_KEEP:
                out.append({"text": e.text, "label": e.label_, "source": f"pretrained ({model})"})
    return out


def hf_transformer_ner(text: str, model: str = "dslim/bert-base-NER") -> list[dict]:
    """Hugging Face token-classification pipeline (needs Hub access, e.g. Colab)."""
    from transformers import pipeline
    ner = pipeline("token-classification", model=model, aggregation_strategy="simple")
    return [{"text": r["word"], "label": r["entity_group"], "score": round(float(r["score"]), 3),
             "source": f"pretrained HF ({model})"} for r in ner(text[:3000])]


def analyze(text: str, use_trf: bool = False) -> dict:
    res = {"pretrained_sm": pretrained_entities(text), "rule_based": rule_based_entities(text)}
    if use_trf:
        try:
            import spacy
            spacy.load("en_core_web_trf")
            res["pretrained_trf"] = pretrained_entities(text, "en_core_web_trf")
        except Exception as ex:
            res["pretrained_trf_error"] = str(ex)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    counts = defaultdict(Counter)
    examples = {}
    rows = []
    try:
        import spacy
        spacy.load("en_core_web_trf")
        trf_ok = True
    except Exception:
        trf_ok = False
    for _, r in df.iterrows():
        res = analyze(r["cleaned_text"], use_trf=trf_ok and r["resume_id"] in ("ENG_05", "HR_03", "FIN_05"))
        for src, ents in res.items():
            if not isinstance(ents, list):
                continue
            for e in ents:
                counts[e["source"]][e["label"]] += 1
                rows.append({"resume_id": r["resume_id"], **{k: e[k] for k in ("source", "label", "text")}})
        if r["resume_id"] in ("ENG_05", "HR_03", "FIN_05", "HLT_01", "AGR_02"):
            examples[r["resume_id"]] = {k: [(e["label"], e["text"]) for e in v][:40]
                                        for k, v in res.items() if isinstance(v, list)}
    ent_df = pd.DataFrame(rows)
    ent_df.to_csv(OUT / "all_entities.csv", index=False)
    summary = pd.DataFrame(counts).fillna(0).astype(int)
    summary.to_csv(RESULTS_DIR / "ner_entity_counts.csv")
    report = {"entity_counts_by_source": {k: dict(v) for k, v in counts.items()},
              "examples": examples, "transformer_ner_run_on": ["ENG_05", "HR_03", "FIN_05"] if trf_ok else [],
              "notes": ["No NER model was trained in this project; all models are pretrained or rule-based.",
                        "No gold-standard entity annotations exist for these resumes, so precision/recall "
                        "of NER is not reported; outputs were inspected manually.",
                        "Pretrained ORG/PERSON labels are noisy on resumes (skills and section headers are often "
                        "tagged as ORG/PERSON), which is why rule-based SKILL/DEGREE/JOB_TITLE entities were added."]}
    (OUT / "ner_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(summary.to_string())
    for rid, ex in examples.items():
        print(f"\n=== {rid}")
        for src, ents in ex.items():
            print(f"  {src}: {ents[:14]}")


if __name__ == "__main__":
    main()
