"""Dataset validation: run before training.

    python -m src.validate_dataset

Checks (FAIL = must be fixed, WARN = report to a human):
  FAIL  each category folder data/raw/<Category>/ holds exactly 20 resume files, 100 in total
  FAIL  every file is readable and has a meaningful amount of text (>= 150 words)
  FAIL  no exact duplicates (same bytes or same normalised text)
  FAIL  no near-duplicates (character 5-gram TF-IDF cosine >= 0.85)
  FAIL  data/resume_sources.csv lists every file with the right category, a source type,
        provenance and licence; synthetic files are named SYN_* and marked in their PDF metadata
  WARN  content check: the lexicon evidence of a resume points to a different domain than its folder

Writes results/dataset_validation.json and exits with code 1 if any FAIL occurs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys

import numpy as np

from .config import CATEGORIES, DATA_DIR, RAW_DIR, RESULTS_DIR, TARGET_PER_CATEGORY
from .pdf_extraction import SUPPORTED_EXTENSIONS, extract_text
from .skill_extraction import content_label_check, extract_skills

SOURCES_CSV = DATA_DIR / "resume_sources.csv"
MIN_WORDS = 150
NEAR_DUP = 0.85


def _pdf_subject(path) -> str:
    try:
        import fitz
        with fitz.open(str(path)) as d:
            return (d.metadata or {}).get("subject", "") or ""
    except Exception:
        return ""


def validate(per_category: int = TARGET_PER_CATEGORY) -> dict:
    fails, warns, files = [], [], []
    for cat in CATEGORIES:
        folder = RAW_DIR / cat
        if not folder.is_dir():
            fails.append(f"missing folder {folder}")
            continue
        found = sorted(p for p in folder.iterdir() if p.is_file())
        bad_ext = [p.name for p in found if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
        if bad_ext:
            fails.append(f"{cat}: unsupported files {bad_ext}")
        docs = [p for p in found if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(docs) != per_category:
            fails.append(f"{cat}: {len(docs)} resumes (expected {per_category})")
        files += [(cat, p) for p in docs]

    texts, rows = [], []
    for cat, p in files:
        ex = extract_text(p)
        words = len(ex["text"].split())
        text = ex["text"]
        rows.append({"category": cat, "filename": p.name, "status": ex["status"], "words": words,
                     "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                     "norm_hash": hashlib.md5(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()})
        texts.append(text)
        if ex["status"] != "ok" or words < MIN_WORDS:
            fails.append(f"{cat}/{p.name}: unreadable or too short ({ex['status']}, {words} words)")
        check = content_label_check(extract_skills(text), cat)
        rows[-1]["content_domain"] = check["content_domain"]
        rows[-1]["label_check"] = check["label_check"]
        if check["review_required"]:
            warns.append(f"{cat}/{p.name}: {check['label_check']} (domain scores {check['domain_scores']})")

    for key in ("md5", "norm_hash"):
        seen = {}
        for r in rows:
            if r[key] in seen:
                fails.append(f"exact duplicate ({key}): {seen[r[key]]} == {r['category']}/{r['filename']}")
            seen.setdefault(r[key], f"{r['category']}/{r['filename']}")

    max_sim, top_pairs = 0.0, []
    if len(texts) > 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        S = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5)).fit_transform(texts))
        np.fill_diagonal(S, 0)
        max_sim = float(S.max())
        iu = np.dstack(np.unravel_index(np.argsort(S.ravel())[::-1], S.shape))[0]
        for i, j in iu[:10:2]:
            top_pairs.append({"a": rows[i]["filename"], "b": rows[j]["filename"], "cosine": round(float(S[i, j]), 3)})
        for i, j in zip(*np.where(np.triu(S) >= NEAR_DUP)):
            fails.append(f"near-duplicate (cosine {S[i, j]:.2f}): {rows[i]['filename']} ~ {rows[j]['filename']}")

    # ---- provenance manifest
    if not SOURCES_CSV.exists():
        fails.append(f"missing {SOURCES_CSV}")
        src = {}
    else:
        src = {(r["category"], r["filename"]): r for r in csv.DictReader(SOURCES_CSV.open())
               if r.get("status") == "included"}
        listed = set(src)
        actual = {(r["category"], r["filename"]) for r in rows}
        for k in sorted(actual - listed):
            fails.append(f"not in resume_sources.csv: {k}")
        for k in sorted(listed - actual):
            fails.append(f"listed in resume_sources.csv but missing on disk: {k}")
        for (cat, name), r in src.items():
            if r["source_type"] not in ("supplied_original", "synthetic", "public"):
                fails.append(f"{name}: unknown source_type {r['source_type']}")
            if not r.get("provenance") or not r.get("licence"):
                fails.append(f"{name}: provenance/licence missing")
            if r["source_type"] == "synthetic":
                if not name.startswith("SYN_"):
                    fails.append(f"{name}: synthetic file not named SYN_*")
                if "SYNTHETIC" not in _pdf_subject(RAW_DIR / cat / name).upper():
                    fails.append(f"{name}: synthetic PDF lacks SYNTHETIC metadata")
    for r in rows:
        r["source_type"] = src.get((r["category"], r["filename"]), {}).get("source_type", "?")

    counts = {c: sum(r["category"] == c for r in rows) for c in CATEGORIES}
    by_source = {c: {s: sum(r["category"] == c and r["source_type"] == s for r in rows)
                     for s in ("supplied_original", "synthetic")} for c in CATEGORIES}
    report = {"status": "FAIL" if fails else ("WARN" if warns else "PASS"), "total": len(rows),
              "per_category": counts, "per_category_by_source": by_source,
              "words": {"min": min(r["words"] for r in rows), "max": max(r["words"] for r in rows),
                        "mean": round(float(np.mean([r["words"] for r in rows])), 1)} if rows else {},
              "max_pairwise_similarity": round(max_sim, 3), "most_similar_pairs": top_pairs,
              "fails": fails, "warnings": warns}
    (RESULTS_DIR / "dataset_validation.json").write_text(json.dumps(report, indent=1))
    with (RESULTS_DIR / "dataset_validation_files.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return report


def main():
    rep = validate()
    print(f"DATASET VALIDATION: {rep['status']}")
    print(f"  total resumes: {rep['total']}")
    for c in CATEGORIES:
        s = rep["per_category_by_source"][c]
        print(f"  {c:12s} {rep['per_category'][c]:3d}  (original {s['supplied_original']}, synthetic {s['synthetic']})")
    print(f"  words per resume: {rep['words']}")
    print(f"  max pairwise similarity: {rep['max_pairwise_similarity']} (near-duplicate threshold {NEAR_DUP})")
    for f in rep["fails"]:
        print("  FAIL", f)
    for w in rep["warnings"]:
        print("  WARN", w)
    print("  details: results/dataset_validation.json, results/dataset_validation_files.csv")
    sys.exit(1 if rep["fails"] else 0)


if __name__ == "__main__":
    main()
