"""Stages A-E of the professor's data workflow.

A. Inventory every file under data/raw/<Category>/ -> data/dataset_manifest.csv
B. Select up to 5 valid resumes per category       -> pilot sample (25)
C. Preprocess the pilot, save intermediate outputs  -> data/interim/pilot/*
D. Process ALL available resumes                    -> data/resume_dataset.csv/.json
E. Check counts against the 20/category, 100 total target -> results/dataset_report.json

Run:  python -m src.build_dataset
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from .config import (CATEGORIES, DATASET_CSV, DATASET_JSON, INTERIM_DIR, MANIFEST_CSV,
                     PILOT_CSV, PILOT_PER_CATEGORY, PROCESSED_DIR, RAW_DIR, RESULTS_DIR,
                     TARGET_PER_CATEGORY, canonical_category)
from .pdf_extraction import SUPPORTED_EXTENSIONS, extract_text, file_md5
from .preprocessing import (anonymize, clean_text, preprocess, segment_sections,
                            validate_script_language)
from .skill_extraction import analyze_resume

PREFIX = {"Engineering": "ENG", "HR": "HR", "Finance": "FIN", "Healthcare": "HLT", "Agriculture": "AGR"}


def load_sources() -> dict:
    """(category, filename) -> provenance row from data/resume_sources.csv."""
    path = RAW_DIR.parent / "resume_sources.csv"
    if not path.exists():
        return {}
    src = pd.read_csv(path)
    return {(r.category, r.filename): r._asdict() for r in src.itertuples(index=False)}


# ----------------------------------------------------------------- Stage A
def stage_a_inventory() -> pd.DataFrame:
    rows = []
    sources = load_sources()
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        cat = canonical_category(folder.name)
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            rec = {"source_filename": f.name, "source_path": str(f.relative_to(RAW_DIR.parent.parent)),
                   "folder": folder.name, "category": cat, "extension": f.suffix.lower(),
                   "size_bytes": f.stat().st_size,
                   "source_type": sources.get((cat, f.name), {}).get("source_type", "unknown"),
                   "licence": sources.get((cat, f.name), {}).get("licence", "unknown")}
            if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rec.update(extraction_status="unsupported")
                rows.append(rec)
                continue
            ex = extract_text(f)
            rec.update(file_md5=file_md5(f), n_pages=ex["n_pages"], extraction_method=ex["method"],
                       extraction_status=ex["status"], extraction_error=ex["error"],
                       n_chars=len(ex["text"]), n_words=len(ex["text"].split()),
                       removed_header_lines=len(ex["removed_header_lines"]),
                       title_line=(ex["text"].split("\n")[0][:80] if ex["text"] else ""),
                       _text=ex["text"])
            rows.append(rec)
    df = pd.DataFrame(rows)
    # resume ids: <PREFIX>_<nn> in filename order inside each category
    df["resume_id"] = ""
    for cat, g in df.groupby("category"):
        for i, idx in enumerate(g.index, 1):
            df.at[idx, "resume_id"] = f"{PREFIX.get(cat, 'UNK')}_{i:02d}"
    df["exact_duplicate_of"] = ""
    first = {}
    for idx, r in df.iterrows():
        h = r.get("file_md5")
        if isinstance(h, str):
            if h in first:
                df.at[idx, "exact_duplicate_of"] = first[h]
            else:
                first[h] = r["resume_id"]
    df["valid"] = (df["extraction_status"] == "ok") & (df["exact_duplicate_of"] == "") & df["category"].notna()
    cols = ["resume_id"] + [c for c in df.columns if c not in ("resume_id", "_text")]
    df[cols].to_csv(MANIFEST_CSV, index=False)
    return df


# ----------------------------------------------------------------- Stage B
def stage_b_pilot(manifest: pd.DataFrame) -> pd.DataFrame:
    valid = manifest[manifest["valid"]]
    pilot = valid.groupby("category").head(PILOT_PER_CATEGORY)
    return pilot


# ------------------------------------------------- Stage C / D (per resume)
def process_record(r: pd.Series) -> dict:
    raw = r["_text"]
    anon_raw, anon_counts = anonymize(raw)
    cleaned = clean_text(anon_raw)
    clf = preprocess(anon_raw, "classification")
    trf = preprocess(anon_raw, "transformer")
    info = analyze_resume(cleaned, folder_label=r["category"])
    lang = validate_script_language(cleaned)
    sections = list(segment_sections(cleaned).keys())
    return {
        "resume_id": r["resume_id"], "source_filename": r["source_filename"],
        "source_path": r["source_path"], "category": r["category"], "source_type": r["source_type"],
        "extracted_text": raw, "anonymized_text": anon_raw, "cleaned_text": cleaned,
        "classification_text": clf["text"], "transformer_text": trf["text"],
        "extracted_skills": "; ".join(s["skill"] for s in info["skills"]),
        "skill_count": len(info["skills"]),
        "education": json.dumps(info["education"]),
        "job_titles": "; ".join(info["job_titles"]),
        "organizations": "; ".join(info["organizations"]),
        "experience_information": json.dumps(info["experience"]),
        "language_or_script_validation": json.dumps(lang),
        "valid_english_latin": lang["valid_english_latin"],
        "sections_found": "; ".join(sections),
        "anonymization_counts": json.dumps(anon_counts),
        "domain_scores": json.dumps(info["label_check"]["domain_scores"]),
        "content_domain": info["label_check"]["content_domain"],
        "label_check": info["label_check"]["label_check"],
        "review_required": info["label_check"]["review_required"],
        "n_pages": r["n_pages"], "n_chars_raw": len(raw), "n_chars_cleaned": len(cleaned),
        "n_tokens_classification": len(clf["tokens"]),
    }


def near_duplicates(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Character 5-gram TF-IDF cosine similarity between all resumes."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5)).fit_transform(df["cleaned_text"])
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    ids = df["resume_id"].tolist()
    df["max_similarity"] = S.max(axis=1).round(4)
    df["most_similar_to"] = [ids[i] for i in S.argmax(axis=1)]
    df["near_duplicate_of"] = [ids[j] if S[i, j] >= threshold and j < i else ""
                               for i, j in enumerate(S.argmax(axis=1))]
    return df


def save_pilot_intermediates(pilot_df: pd.DataFrame):
    d = INTERIM_DIR / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    for _, r in pilot_df.iterrows():
        rid = r["resume_id"]
        (d / f"{rid}_1_raw.txt").write_text(r["extracted_text"], encoding="utf-8")
        (d / f"{rid}_2_anonymized.txt").write_text(r["anonymized_text"], encoding="utf-8")
        (d / f"{rid}_3_cleaned.txt").write_text(r["cleaned_text"], encoding="utf-8")
        (d / f"{rid}_4_classification_tokens.txt").write_text(r["classification_text"], encoding="utf-8")


def main():
    print("Stage A: inventory ...")
    manifest = stage_a_inventory()
    print(manifest.groupby("category")["resume_id"].count().to_string())
    bad = manifest[~manifest["valid"]]
    if len(bad):
        print("Invalid / skipped files:\n", bad[["source_path", "extraction_status"]].to_string())

    print("Stage B: pilot selection ...")
    pilot = stage_b_pilot(manifest)

    print("Stage C: preprocess pilot ...")
    pilot_rows = pd.DataFrame([process_record(r) for _, r in pilot.iterrows()])
    pilot_rows["is_pilot"] = True
    pilot_rows.to_csv(PILOT_CSV, index=False)
    save_pilot_intermediates(pilot_rows)

    print("Stage D: process all available resumes ...")
    rest = manifest[manifest["valid"] & ~manifest["resume_id"].isin(pilot["resume_id"])]
    rest_rows = pd.DataFrame([process_record(r) for _, r in rest.iterrows()]) if len(rest) else pd.DataFrame()
    if len(rest_rows):
        rest_rows["is_pilot"] = False
    full = pd.concat([pilot_rows, rest_rows], ignore_index=True)
    full = near_duplicates(full)
    full["review_required"] = full["review_required"] | (full["near_duplicate_of"] != "") | ~full["valid_english_latin"]
    full.to_csv(DATASET_CSV, index=False)
    full.to_json(DATASET_JSON, orient="records", indent=1, force_ascii=False)
    full[["resume_id", "source_filename", "source_path", "category", "source_type", "is_pilot"]].to_csv(
        PROCESSED_DIR / "record_mapping.csv", index=False)

    print("Stage E: target check ...")
    counts = full.groupby("category")["resume_id"].count().reindex(CATEGORIES, fill_value=0)
    flagged = full.groupby("category")["review_required"].sum().reindex(CATEGORIES, fill_value=0)
    table = pd.DataFrame({"available_valid": counts, "target": TARGET_PER_CATEGORY,
                          "shortfall": (TARGET_PER_CATEGORY - counts).clip(lower=0),
                          "pilot": pilot_rows.groupby("category")["resume_id"].count()
                          .reindex(CATEGORIES, fill_value=0),
                          "flagged_for_review": flagged.astype(int)})
    table.index.name = "category"
    table.to_csv(RESULTS_DIR / "category_counts.csv")
    report = {
        "files_found": int(len(manifest)),
        "valid_resumes": int(manifest["valid"].sum()),
        "pilot_size": int(len(pilot_rows)),
        "full_dataset_size": int(len(full)),
        "target_total": TARGET_PER_CATEGORY * len(CATEGORIES),
        "target_met": bool(len(full) >= TARGET_PER_CATEGORY * len(CATEGORIES)),
        "pilot_equals_full_dataset": bool(len(rest_rows) == 0),
        "exact_duplicates": int((manifest["exact_duplicate_of"] != "").sum()),
        "near_duplicates": int((full["near_duplicate_of"] != "").sum()),
        "max_pairwise_similarity": float(full["max_similarity"].max()),
        "review_required": full.loc[full["review_required"], ["resume_id", "source_filename", "category",
                                                              "content_domain", "label_check"]]
        .to_dict(orient="records"),
        "per_category": table.reset_index().to_dict(orient="records"),
    }
    (RESULTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(table.to_string())
    print(json.dumps({k: v for k, v in report.items() if k not in ("review_required", "per_category")}, indent=2))
    print("Flagged for manual review:")
    for r in report["review_required"]:
        print("  ", r)


if __name__ == "__main__":
    main()
