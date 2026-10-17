"""Shared lazy loaders for NLTK resources and spaCy pipelines."""
from __future__ import annotations

from functools import lru_cache

NLTK_PACKAGES = {  # package -> resource path used to test if it is present
    "punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords", "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "treebank": "corpora/treebank", "universal_tagset": "taggers/universal_tagset",
}


@lru_cache(maxsize=1)
def ensure_nltk() -> bool:
    import nltk
    for pkg, resource in NLTK_PACKAGES.items():
        try:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.data.find(resource + ".zip")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True, raise_on_error=False)
            except Exception:
                pass
    return True


@lru_cache(maxsize=4)
def get_spacy(model: str = "en_core_web_sm"):
    """Load a spaCy pipeline; fall back to a blank English pipeline."""
    import spacy
    try:
        return spacy.load(model)
    except OSError:
        print(f"[warn] spaCy model {model!r} not installed -> "
              f"run: python -m spacy download {model}. Using blank 'en'.")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp
