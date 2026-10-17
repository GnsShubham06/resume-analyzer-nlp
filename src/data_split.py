"""Leakage-safe, stratified train/validation/test split shared by ALL models.

* Split is done at RESUME level (never at chunk level), so chunks of one
  resume can never appear in both training and test data.
* Near-duplicate resumes (see build_dataset.near_duplicates) are kept in the
  same split as the resume they duplicate.
* 60 / 20 / 20 % per category: with 20 resumes per category this is
  12 train / 4 validation / 4 test (60 / 20 / 20 resumes in total).
* The split is written to data/processed/splits.csv and reused by BOTH the
  baseline and the transformer, so they are compared on the same resumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CATEGORIES, DATASET_CSV, RANDOM_SEED, SPLITS_CSV


def load_dataset(path=DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["category"].isin(CATEGORIES)].reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, seed: int = RANDOM_SEED, save: bool = True) -> pd.DataFrame:
    """Resume-level split, stratified by category AND by source type.

    Inside each category the resumes are shuffled (fixed seed), supplied
    originals are placed first and the split pattern test, val, train, train,
    train is repeated.  With 20 resumes per category this gives exactly
    12 train / 4 validation / 4 test, and the few real (supplied) resumes are
    spread over the three splits instead of all landing in training.
    """
    rng = np.random.RandomState(seed)
    pattern = ["test", "val", "train", "train", "train"]
    split = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if "source_type" in sub.columns:
            real = sub.loc[sub["source_type"] != "synthetic", "resume_id"].tolist()
            syn = sub.loc[sub["source_type"] == "synthetic", "resume_id"].tolist()
        else:
            real, syn = sub["resume_id"].tolist(), []
        rng.shuffle(real)
        rng.shuffle(syn)
        for i, rid in enumerate(real + syn):
            split[rid] = pattern[i % len(pattern)]
    out = df[["resume_id", "source_filename", "category"] + (["source_type"] if "source_type" in df.columns else [])].copy()
    out["split"] = out["resume_id"].map(split)
    if "near_duplicate_of" in df.columns:  # keep near-duplicates together
        for i, dup in df["near_duplicate_of"].fillna("").items():
            if dup:
                out.loc[i, "split"] = out.loc[out["resume_id"] == dup, "split"].iloc[0]
    if save:
        out.to_csv(SPLITS_CSV, index=False)
    return out


def get_splits(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load the saved split; (re)create it if it is missing or out of date."""
    df = df if df is not None else load_dataset()
    if SPLITS_CSV.exists():
        saved = pd.read_csv(SPLITS_CSV)
        if set(saved["resume_id"]) == set(df["resume_id"]) and \
                dict(zip(saved["resume_id"], saved["source_filename"])) == dict(zip(df["resume_id"], df["source_filename"])):
            return saved
    return make_splits(df)


def cv_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED):
    """Stratified K-fold over resumes (each fold = 1 resume per class here)."""
    k = min(n_splits, df["category"].value_counts().min())
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(df, df["category"]))
