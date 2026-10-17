"""Resume-aware text preprocessing (Practicals 3 and 4).

All steps are small pure functions so that each experiment can call exactly
the steps it needs.  `preprocess()` chains them according to a named
configuration, because different tasks need different cleaning:

* "classification" - lowercased, stopwords removed, lemmatized, skill phrases
  kept as single tokens (used by the TF-IDF baseline and Word2Vec).
* "transformer"    - light cleaning only (transformers need natural text,
  casing/stopwords/numbers are kept).
* "extraction"     - noise fixed but casing, numbers and dates preserved
  (used for NER, dates and skill extraction).
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

from .nlp_utils import ensure_nltk, get_spacy
from .skill_lexicon import SECTION_HEADERS, all_skill_terms

# --------------------------------------------------------------------------
# 1. HTML tags
# --------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def remove_html_tags(text: str) -> str:
    text = _SCRIPT_RE.sub(" ", text)
    text = re.sub(r"<\s*(br|/p|/div|/li|/h\d)\s*/?>", "\n", text, flags=re.I)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


# --------------------------------------------------------------------------
# 2. Whitespace
# --------------------------------------------------------------------------
def normalize_whitespace(text: str, keep_newlines: bool = True) -> str:
    text = text.replace("\r", "\n").replace("\t", " ")
    if keep_newlines:
        text = re.sub(r"[  ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# 3. Special characters (keep the ones that carry meaning in skills:
#    C++, C#, .NET, R&D, GD&T, P&L, 3D, i-9, 24/7 ...)
# --------------------------------------------------------------------------
def handle_special_characters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[•▪●◦■□►✓✔·\*]+", " ", text)          # bullets
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’`]", "'", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"[^\w\s\.\,\;\:\-\+\#\&\/\(\)\[\]\'\"%\$@!?]", " ", text)
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)                # "!!!" -> "!"
    return text


# --------------------------------------------------------------------------
# 4. Sentence boundary detection (resume aware)
# --------------------------------------------------------------------------
def split_sentences(text: str, method: str = "nltk") -> list[str]:
    """Resumes use line breaks and bullets as boundaries, so we first split
    on lines, then run a sentence tokenizer inside each line."""
    out = []
    for line in text.split("\n"):
        line = line.strip(" -|")
        if not line:
            continue
        if method == "spacy":
            out.extend(s.text.strip() for s in get_spacy()(line).sents if s.text.strip())
        elif method == "regex":
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip())
        else:
            ensure_nltk()
            from nltk.tokenize import sent_tokenize
            out.extend(s.strip() for s in sent_tokenize(line) if s.strip())
    return out


# --------------------------------------------------------------------------
# 5. Contractions
# --------------------------------------------------------------------------
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "shan't": "shall not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
    "let's": "let us", "y'all": "you all",
}


def expand_contractions(text: str) -> str:
    try:  # use the `contractions` package when installed
        import contractions
        # slang=False: otherwise "RN" -> "right now", "U.S." -> "you.s"
        return contractions.fix(text, slang=False)
    except Exception:
        pass
    for k in ("won't", "can't", "shan't", "let's", "y'all"):
        text = re.sub(k, CONTRACTIONS[k], text, flags=re.I)
    text = re.sub(r"(\w)n't\b", r"\1 not", text)
    for k in ("'re", "'d", "'ll", "'ve", "'m"):
        text = re.sub(rf"(\w){k}\b", rf"\1{CONTRACTIONS[k]}", text)
    return text


# --------------------------------------------------------------------------
# 6. Lowercasing
# --------------------------------------------------------------------------
def lowercase(text: str) -> str:
    return text.lower()


# --------------------------------------------------------------------------
# 7. Number words -> digits ("twenty years" -> "20 years")
# --------------------------------------------------------------------------
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_NUMWORDS = {w: i for i, w in enumerate(_UNITS)}
_NUMWORDS.update({w: 10 * i for i, w in enumerate(_TENS) if w})
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
_NUM_TOKEN = "|".join(sorted(list(_NUMWORDS) + list(_SCALES), key=len, reverse=True))
_NUM_SPAN_RE = re.compile(rf"\b(?:{_NUM_TOKEN})(?:(?:[\s-]+|\s+and\s+)(?:{_NUM_TOKEN}))*\b", re.I)


def _words_to_int(span: str) -> int:
    total, current = 0, 0
    for w in re.split(r"[\s-]+", span.lower()):
        if w == "and":
            continue
        if w in _NUMWORDS:
            current += _NUMWORDS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def words_to_numbers(text: str) -> str:
    """Convert spelled-out numbers to digits.  The single word "one" is left
    alone when it is used as a pronoun ("one of the ...")."""
    def repl(m):
        span = m.group(0)
        if span.lower() == "one" and re.match(r"\s+of\b", text[m.end():m.end() + 4]):
            return span
        return str(_words_to_int(span))
    return _NUM_SPAN_RE.sub(repl, text)


# --------------------------------------------------------------------------
# 8. Number removal (separate, configurable experiment)
# --------------------------------------------------------------------------
def remove_numbers(text: str, keep_years: bool = False) -> str:
    if keep_years:
        return re.sub(r"\b(?!(19|20)\d{2}\b)\d+(?:[.,]\d+)*\b", " ", text)
    return re.sub(r"\b\d+(?:[.,]\d+)*\b", " ", text)


# --------------------------------------------------------------------------
# 9. Stopwords (skill-phrase-aware)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    ensure_nltk()
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    # words that change meaning in resumes are kept
    sw -= {"no", "not", "nor", "it", "own"}
    return frozenset(sw)


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = get_stopwords()
    # tokens joined by "_" are multi-word skills -> never removed
    return [t for t in tokens if "_" in t or t.lower() not in sw]


# --------------------------------------------------------------------------
# 10/11. Tokenization and phrase extraction
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _mwe_tokenizer():
    ensure_nltk()
    from nltk.tokenize import MWETokenizer
    phrases = [tuple(t.split()) for t in all_skill_terms() if " " in t]
    return MWETokenizer(phrases, separator="_")


def tokenize(text: str, method: str = "nltk") -> list[str]:
    ensure_nltk()
    if method == "whitespace":
        return text.split()
    if method == "regex":
        return re.findall(r"[A-Za-z][A-Za-z\+\#\.&]*[A-Za-z\+\#]|[A-Za-z]|\d+(?:\.\d+)?", text)
    if method == "spacy":
        return [t.text for t in get_spacy()(text) if not t.is_space]
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)


def skill_aware_tokenize(text: str) -> list[str]:
    """Word tokenization that keeps multi-word skills ("machine learning",
    "human resource management") together as one token (machine_learning)."""
    toks = tokenize(text.lower())
    return _mwe_tokenizer().tokenize(toks)


def extract_noun_phrases(text: str, max_len: int = 5) -> list[str]:
    doc = get_spacy()(text)
    if not doc.has_annotation("DEP"):
        return []
    phrases = []
    for nc in doc.noun_chunks:
        toks = [t for t in nc if not (t.is_stop or t.is_punct)]
        if 1 < len(toks) <= max_len:
            phrases.append(" ".join(t.text.lower() for t in toks))
    return phrases


# --------------------------------------------------------------------------
# 12. Script and language validation
# --------------------------------------------------------------------------
def validate_script_language(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    n = max(len(letters), 1)
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    scripts = {}
    for c in letters:
        name = unicodedata.name(c, "UNKNOWN").split(" ")[0]
        scripts[name] = scripts.get(name, 0) + 1
    lang, prob = "unknown", 0.0
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        best = detect_langs(text[:5000])[0]
        lang, prob = best.lang, round(best.prob, 3)
    except Exception:
        pass
    # line-level detection finds mixed-language resumes (e.g. an English
    # resume with an Indonesian section) that a whole-document guess hides
    other_lines = {}
    try:
        from langdetect import detect_langs
        for line in text.split("\n"):
            letters_ = [c for c in line if c.isalpha()]
            mostly_caps = letters_ and sum(c.isupper() for c in letters_) / len(letters_) > 0.5
            if len(line.split()) >= 5 and not mostly_caps:
                best = detect_langs(line)[0]
                if best.lang != "en" and best.prob > 0.9:
                    other_lines[best.lang] = other_lines.get(best.lang, 0) + 1
    except Exception:
        pass
    latin_ratio = latin / n
    ok = latin_ratio > 0.95 and lang in ("en", "unknown")
    return {"latin_ratio": round(latin_ratio, 4), "scripts": scripts, "language": lang,
            "language_prob": prob, "valid_english_latin": ok,
            "non_english_lines": other_lines, "mixed_language": sum(other_lines.values()) >= 3}


# --------------------------------------------------------------------------
# 13. Handling different textual formats
# --------------------------------------------------------------------------
def normalize_format(text: str, fmt: str = "pdf") -> str:
    """Bring text from different sources into one plain-text form."""
    if fmt in ("html", "htm") or re.search(r"</?(p|div|br|li|span|html)\b", text, re.I):
        text = remove_html_tags(text)
    if fmt == "markdown":
        text = re.sub(r"[#*_>`]+", " ", text)
    text = text.replace(" ", "\n").replace("\f", "\n")
    return text


# --------------------------------------------------------------------------
# Section segmentation (process the resume meaningfully, not as one blob)
# --------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse=True)) + r")\s*:?\s*$",
    re.I)


def segment_sections(text: str) -> dict[str, str]:
    sections, current = {"header": []}, "header"
    for line in text.split("\n"):
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


# --------------------------------------------------------------------------
# Practical 4: stemming, lemmatization, noise, anonymization
# --------------------------------------------------------------------------
def stem_tokens(tokens: list[str], algorithm: str = "porter") -> list[str]:
    ensure_nltk()
    from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
    stemmer = {"porter": PorterStemmer(), "lancaster": LancasterStemmer(),
               "snowball": SnowballStemmer("english")}[algorithm]
    return [t if "_" in t else stemmer.stem(t) for t in tokens]


def _wn_pos(tag: str):
    from nltk.corpus import wordnet as wn
    return {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}.get(tag[:1], wn.NOUN)


def lemmatize_tokens(tokens: list[str], method: str = "wordnet") -> list[str]:
    if method == "spacy":
        doc = get_spacy()(" ".join(tokens))
        return [t.lemma_ if "_" not in t.text else t.text for t in doc]
    ensure_nltk()
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    return [t if "_" in t else wnl.lemmatize(t, _wn_pos(tag)) for t, tag in pos_tag(tokens)]


# placeholders and template noise seen in the supplied (already partly
# anonymised) resumes: "Company Name , City , State", "[Number] years" ...
_NOISE_PATTERNS = [
    (re.compile(r"\bCompany Name\s*[,|]?\s*City\s*,\s*State\b", re.I), " "),
    (re.compile(r"\b(Company Name|City\s*,\s*State)\b", re.I), " "),
    # template placeholders such as [Number], [level of management]; our own
    # anonymization masks ([NAME], [EMAIL] ...) are kept
    (re.compile(r"\[(?!(?:NAME|EMAIL|PHONE|URL|ADDRESS)\])[^\]]{1,40}\]"), " "),
    # provenance stamp printed on synthetic resumes (scripts/prepare_dataset_100.py):
    # removed so that no model can learn "synthetic" as a feature
    (re.compile(r"^.*Synthetic resume - fictional person generated for the Resume Analyzer.*$", re.M), " "),
    (re.compile(r"(.)\1{3,}"), r"\1"),                         # "!!!!!", "-----"
    (re.compile(r"https?://\S+|www\.\S+"), " "),
]


def handle_noise(text: str) -> str:
    for pat, rep in _NOISE_PATTERNS:
        text = pat.sub(rep, text)
    # drop consecutive duplicate lines (PDF column artefacts)
    lines, prev = [], None
    for line in text.split("\n"):
        key = line.strip().lower()
        if key and key == prev:
            continue
        lines.append(line)
        prev = key
    return "\n".join(lines)


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+", re.I)
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Blvd|Boulevard|Court|Ct|Way)\b\.?",
)
_ZIP_RE = re.compile(r"\b[A-Z]{2}\s\d{5}(?:-\d{4})?\b")


_INITIAL_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,} [A-Z]\.)(?=[,;\s]|$)")   # "Parekh R." author lists


def _is_plausible_name(span_text: str) -> bool:
    """Filter spaCy PERSON false positives (very common on resumes: skills,
    product names, section headers).  A plausible name is 1-3 capitalised
    tokens of which at least one is NOT an ordinary English word in WordNet
    ("Joseph Sirotnak" -> yes, "Delineation Knowledgeable" -> no)."""
    ensure_nltk()
    from nltk.corpus import wordnet as wn
    toks = span_text.split()
    if not 1 <= len(toks) <= 3 or span_text.lower() in all_skill_terms():
        return False
    unknown = 0
    for t in toks:
        if re.fullmatch(r"[A-Z]\.", t):     # initials are allowed but do not count
            continue
        if not re.fullmatch(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?", t):
            return False
        if t.lower() in _SECTION_WORDS or t.lower() in _ORG_WORDS:
            return False
        if not wn.synsets(t.lower()):
            unknown += 1
    return unknown >= 1


# allow-list: product / company / law names that spaCy-sm often mistakes for people
NOT_NAMES = {"matplotlib", "visio", "gripper", "sarbanes oxley", "pfizer", "abbott", "leica",
             "acta", "pandas", "numpy", "oracle", "java", "excel", "peoplesoft", "cum laude",
             "adobe dreamweaver", "aquafresh white strips", "emc xiv", "lindgren funnel", "greenie tots"}
_ORG_WORDS = {"college", "university", "institute", "school", "city", "inc", "corp", "assoc", "laude"}

_SECTION_WORDS = {w for h in SECTION_HEADERS for w in h.split()} | {"focus", "career", "resume"}


def find_person_names(text: str) -> set[str]:
    names = set(_INITIAL_NAME_RE.findall(text))
    nlp = get_spacy()
    if nlp.has_pipe("ner"):
        lines = [l for l in text.split("\n") if l.strip()]
        for i, doc in enumerate(nlp.pipe(lines, batch_size=64)):
            for e in doc.ents:
                t = e.text.strip(" '\".,;:-")
                # the candidate's own name normally sits in the first lines:
                # there we trust spaCy; elsewhere we also apply the filter
                header_name = (i < 3 and re.fullmatch(r"[A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+){1,2}", t)
                               and not set(t.lower().split()) & _SECTION_WORDS)
                if e.label_ == "PERSON" and t.lower() not in NOT_NAMES and (header_name or _is_plausible_name(t)):
                    names.add(t)
    return names


def anonymize(text: str, use_ner: bool = True) -> tuple[str, dict]:
    """Mask e-mails, phone numbers, URLs, street addresses and person names.
    Returns the masked text and a count of each replacement."""
    counts = {}

    def sub(pattern, token, s):
        new, n = pattern.subn(token, s)
        counts[token] = counts.get(token, 0) + n
        return new

    text = sub(_EMAIL_RE, "[EMAIL]", text)
    text = sub(_URL_RE, "[URL]", text)
    text = sub(_PHONE_RE, "[PHONE]", text)
    text = sub(_ADDRESS_RE, "[ADDRESS]", text)
    text = sub(_ZIP_RE, "[ADDRESS]", text)
    if use_ner:
        names = find_person_names(text)
        for n in sorted(names, key=len, reverse=True):
            text = re.sub(rf"(?<![\w\[]){re.escape(n)}(?!\w)", "[NAME]", text)
        counts["[NAME]"] = len(names)
    return text, counts


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
CONFIGS = {
    "classification": dict(lower=True, numbers="remove", stopwords=True, lemma=True, stem=False),
    "transformer": dict(lower=False, numbers="keep", stopwords=False, lemma=False, stem=False),
    "extraction": dict(lower=False, numbers="keep", stopwords=False, lemma=False, stem=False),
    "stemmed": dict(lower=True, numbers="remove", stopwords=True, lemma=False, stem=True),
}


def clean_text(raw: str, fmt: str = "pdf") -> str:
    """Shared light cleaning used before every task-specific pipeline."""
    t = normalize_format(raw, fmt)
    t = remove_html_tags(t)
    t = handle_noise(t)
    t = handle_special_characters(t)
    t = expand_contractions(t)
    t = words_to_numbers(t)
    return normalize_whitespace(t, keep_newlines=True)


def preprocess(raw: str, config: str = "classification", fmt: str = "pdf") -> dict:
    cfg = CONFIGS[config]
    cleaned = clean_text(raw, fmt)
    text = cleaned.lower() if cfg["lower"] else cleaned
    if config not in ("transformer", "extraction"):   # masks carry no domain information
        text = re.sub(r"\[(name|email|phone|url|address)\]", " ", text, flags=re.I)
    if cfg["numbers"] == "remove":
        text = remove_numbers(text)
    if config in ("transformer", "extraction"):
        return {"cleaned": cleaned, "text": normalize_whitespace(text, keep_newlines=False), "tokens": []}
    toks = skill_aware_tokenize(text)
    toks = [t for t in toks if re.search(r"[a-z0-9]", t) and len(t) > 1]
    if cfg["stopwords"]:
        toks = remove_stopwords(toks)
    if cfg["lemma"]:
        toks = lemmatize_tokens(toks)
    if cfg["stem"]:
        toks = stem_tokens(toks)
    return {"cleaned": cleaned, "text": " ".join(toks), "tokens": toks}
