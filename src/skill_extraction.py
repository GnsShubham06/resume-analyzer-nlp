"""Task B: information & skill extraction (separate from classification).

Methods
-------
* Skills         - dictionary lookup with spaCy PhraseMatcher (case-insensitive,
                   multi-word aware) over the lexicons in skill_lexicon.py.
* Education      - regular-expression patterns for degrees + institution names.
* Job titles     - resume title line + lines whose last word is a job headword.
* Organizations  - spaCy pretrained NER (ORG) + institution regex.
* Experience     - date-range regexes and "N years of experience" patterns.
* Domain check   - counts domain-specific lexicon hits to *flag* resumes whose
                   content disagrees with their folder label (manual review).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from .nlp_utils import get_spacy
from .preprocessing import segment_sections
from .skill_lexicon import (DEGREE_PATTERNS, DOMAIN_SKILLS, JOB_TITLE_HEADWORDS,
                            QUALIFICATION_KEYWORDS, all_skill_terms)


@lru_cache(maxsize=1)
def _phrase_matcher():
    from spacy.matcher import PhraseMatcher
    nlp = get_spacy()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for term, dom in all_skill_terms().items():
        matcher.add(f"{dom}::{term}", [nlp.make_doc(term)])
    return matcher


def extract_skills(text: str) -> list[dict]:
    """Return unique skills with domain and frequency (longest match wins)."""
    nlp = get_spacy()
    doc = nlp.make_doc(text)
    from spacy.util import filter_spans
    matches = _phrase_matcher()(doc)
    spans = filter_spans([doc[s:e] for _, s, e in matches])
    label_of = {}
    for mid, s, e in matches:
        label_of[(s, e)] = nlp.vocab.strings[mid]
    counter = Counter()
    for sp in spans:
        dom, term = label_of[(sp.start, sp.end)].split("::", 1)
        # short ambiguous acronyms must be written in capitals (RN, CAD ...)
        if len(term) <= 3 and sp.text != sp.text.upper():
            continue
        counter[(term, dom)] += 1
    return [{"skill": t, "domain": d, "count": c} for (t, d), c in counter.most_common()]


def domain_scores(skills: list[dict]) -> dict[str, int]:
    scores = {d: 0 for d in DOMAIN_SKILLS}
    for s in skills:
        if s["domain"] in scores:
            scores[s["domain"]] += s["count"]
    return scores


def content_label_check(skills: list[dict], folder_label: str, margin: float = 1.5) -> dict:
    """Rule-based check of the folder label against resume content.

    review_required=True when the folder label's lexicon score is not the
    highest, or when the best domain beats it by `margin`x.  This never
    changes the label automatically; it only flags it for a human.
    """
    sc = domain_scores(skills)
    best = max(sc, key=sc.get)
    lab = sc.get(folder_label, 0)
    if sc[best] == 0:
        status, review = "no_domain_evidence", True
    elif best == folder_label:
        status, review = "consistent", False
    elif sc[best] >= margin * max(lab, 1):
        status, review = f"content_suggests_{best}", True
    else:
        status, review = f"ambiguous_{best}_vs_{folder_label}", True
    return {"domain_scores": sc, "content_domain": best, "label_check": status,
            "review_required": review}


_DEGREE_RES = [re.compile(p, re.I) for p in DEGREE_PATTERNS]
_INSTITUTION_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z&'.-]+ ){0,5}(?:University|College|Institute|School|Academy|Polytechnic)"
    r"(?: (?:of|for|at) (?:[A-Z][A-Za-z&'.-]+ ?){1,4})?)")


def extract_education(text: str) -> dict:
    sections = segment_sections(text)
    edu_text = sections.get("education") or sections.get("education and training") or text
    degrees = []
    for line in edu_text.split("\n"):
        for rx in _DEGREE_RES:
            for m in rx.finditer(line):
                d = m.group(0).strip(" ,.:")
                if len(d) >= 2 and d.lower() not in ("ma", "ms", "ba", "be") or d.isupper():
                    degrees.append(d)
    institutions = [m.group(1).strip() for m in _INSTITUTION_RE.finditer(edu_text)]
    quals = sorted({q.upper() for q in QUALIFICATION_KEYWORDS
                    if re.search(rf"\b{re.escape(q)}\b", text, re.I if len(q) > 3 else 0)
                    or re.search(rf"\b{re.escape(q.upper())}\b", text)})
    return {"degrees": list(dict.fromkeys(degrees))[:8],
            "institutions": list(dict.fromkeys(institutions))[:6],
            "qualifications": quals}


_TITLE_LINE_RE = re.compile(
    r"^([A-Z][A-Za-z/&,.' -]{2,80}?\b(?:" + "|".join(JOB_TITLE_HEADWORDS) + r")s?)\b", re.I)


_SMALL = {"and", "of", "the", "for", "in", "to", "at", "a", "an", "&"}


def smart_title(line: str) -> str:
    """'HR BENEFITS/LEAVE COORDINATOR' -> 'HR Benefits/Leave Coordinator'
    (keeps short acronyms such as HR, RN, VP upper-case; no "Women'S")."""
    out = []
    for i, w in enumerate(line.split()):
        core = re.sub(r"[^A-Za-z]", "", w)
        if len(core) <= 3 and core.isalpha() and core.lower() not in _SMALL:
            out.append(w)
        elif w.lower() in _SMALL and i:
            out.append(w.lower())
        else:
            out.append("/".join(p[:1].upper() + p[1:].lower() for p in w.split("/")))
    return " ".join(out)


def extract_job_titles(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    titles = []
    if lines:
        titles.append(smart_title(lines[0]) if lines[0].isupper() else lines[0])
    exp = segment_sections(text)
    exp_text = "\n".join(v for k, v in exp.items() if "experience" in k or "history" in k)
    for line in exp_text.split("\n"):
        line = re.sub(r"\b(Company Name|City\s*,\s*State)\b.*$", "", line).strip(" ,|-")
        line = re.sub(r"\b(\d{1,2}/\d{4}|\w+ \d{4}|\d{4})\b.*$", "", line).strip(" ,|-")
        m = _TITLE_LINE_RE.match(line)
        # whole (short) line must be the title: avoids sentences such as
        # "Work closely with Design Engineers"
        if m and m.group(1).strip() == line and len(line.split()) <= 6 \
                and not re.match(r"(?i)(i|we|my|work|assist|provide|responsible|managed|consult)\b", line):
            titles.append(m.group(1).strip())
    seen, out = set(), []
    for t in titles:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:10]


def extract_organizations(text: str) -> list[str]:
    nlp = get_spacy()
    orgs = []
    if nlp.has_pipe("ner"):
        skills = set(all_skill_terms())
        lines = [l for l in text.split("\n") if l.strip()]   # entities never cross lines
        for doc in nlp.pipe(lines, batch_size=64):
            for e in doc.ents:
                t = e.text.strip(" ,;:")
                if e.label_ == "ORG" and t.lower() not in skills and len(t) > 2 and len(t.split()) <= 6 \
                        and not re.search(r"company name|city|state|skills?|summary|qualifications", t, re.I):
                    orgs.append(t)
    orgs += [m.group(1).strip() for m in _INSTITUTION_RE.finditer(text)]
    return list(dict.fromkeys(orgs))[:15]


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
_RANGE_RE = re.compile(rf"({_DATE})\s*(?:to|-|–|until)\s*({_DATE}|current|present|now)", re.I)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+\w+){0,3}\s+experience", re.I)


def extract_experience(text: str) -> dict:
    ranges = [f"{a} to {b}" for a, b in _RANGE_RE.findall(text)]
    years = [float(y) for y in _YEARS_RE.findall(text)]
    yrs = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", " ".join(ranges))]
    has_current = any(re.search(r"current|present|now", r, re.I) for r in ranges)
    span = None
    if yrs:
        span = (2026 if has_current else max(yrs)) - min(yrs)
    return {"date_ranges": ranges[:12], "stated_years_experience": years[:5],
            "n_positions_with_dates": len(ranges), "career_span_years_estimate": span}


def analyze_resume(text: str, folder_label: str | None = None) -> dict:
    skills = extract_skills(text)
    out = {
        "skills": skills,
        "education": extract_education(text),
        "job_titles": extract_job_titles(text),
        "organizations": extract_organizations(text),
        "experience": extract_experience(text),
    }
    if folder_label:
        out["label_check"] = content_label_check(skills, folder_label)
    else:
        sc = domain_scores(skills)
        out["label_check"] = {"domain_scores": sc, "content_domain": max(sc, key=sc.get)}
    return out
