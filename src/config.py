"""Central configuration: paths, categories and shared constants.

Every module imports paths from here so that the notebook, the CLI scripts
and the Streamlit app all read/write the same files.
"""
from pathlib import Path
import os

ROOT = Path(os.environ.get("RESUME_NLP_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MANIFEST_CSV = DATA_DIR / "dataset_manifest.csv"
DATASET_CSV = DATA_DIR / "resume_dataset.csv"
DATASET_JSON = DATA_DIR / "resume_dataset.json"
PILOT_CSV = PROCESSED_DIR / "pilot_dataset.csv"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"

# The five target domains (fixed order = label ids 0..4)
CATEGORIES = ["Engineering", "HR", "Finance", "Healthcare", "Agriculture"]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

# Folder names in the supplied zip were inconsistent ("Health care", "Hr",
# trailing spaces).  Map every variant onto the canonical category name.
FOLDER_TO_CATEGORY = {
    "engineering": "Engineering",
    "hr": "HR",
    "human resources": "HR",
    "finance": "Finance",
    "health care": "Healthcare",
    "healthcare": "Healthcare",
    "agriculture": "Agriculture",
}

TARGET_PER_CATEGORY = 20
PILOT_PER_CATEGORY = 5
RANDOM_SEED = 42

# Transformer used for fine-tuning (src/train_transformer.py).  The default is the
# official Hugging Face checkpoint "roberta-base".  When the Hugging Face Hub is
# unreachable, train_transformer.py falls back to LOCAL_ROBERTA_DIR, a RoBERTa-base
# converted from spaCy's en_core_web_trf by scripts/convert_spacy_roberta.py.
DEFAULT_TRANSFORMER = os.environ.get("RESUME_NLP_MODEL", "roberta-base")
LOCAL_ROBERTA_DIR = MODELS_DIR / "pretrained" / "roberta-base-spacy"
# The fine-tuned checkpoint (best validation epoch) - loaded by predict.py and app.py
FINETUNED_MODEL_DIR = MODELS_DIR / "transformer_finetuned"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def canonical_category(folder_name: str):
    """Return the canonical category for a folder name, or None."""
    return FOLDER_TO_CATEGORY.get(folder_name.strip().lower())
