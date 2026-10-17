"""Resume text extraction for PDF / DOCX / HTML / TXT inputs.

* pdfplumber is the primary PDF engine; PyMuPDF (fitz) is the fallback.
* Multi-page PDFs are joined page by page.
* Lines repeated at the top/bottom of several pages (running headers and
  footers) are removed.
* Common UTF-8-read-as-cp1252 "mojibake" (e.g. "Â", "ï¼​") is repaired.
* Empty, encrypted or malformed files never crash the pipeline: they return
  status != "ok" and an error message, so they can be listed in the manifest.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm", ".docx"}

# Mojibake fragments observed in the supplied PDFs plus the usual suspects.
MOJIBAKE_MAP = {
    "ï¼​": ",",      # full-width comma U+FF0C read as cp1252
    "ï¼Œ": ",",
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€": '"',
    "â€“": "-", "â€”": "-", "â€¢": "•", "â„¢": "™",
    "Â·": "•", "Â®": "®", "Â©": "©", "Â": " ",
    "​": "", "﻿": "", "\xa0": " ",
}


def fix_mojibake(text: str) -> str:
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    # glyphs pdfplumber cannot map are returned as "(cid:NNN)"; 127 is a bullet
    text = text.replace("(cid:127)", "•")
    return re.sub(r"\(cid:\d+\)", " ", text)


def _remove_repeated_headers(pages: list[str], edge_lines: int = 2) -> tuple[list[str], list[str]]:
    """Remove lines that occur at the top/bottom edge of 2+ pages."""
    if len(pages) < 2:
        return pages, []
    counts = Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        edges = set(lines[:edge_lines] + lines[-edge_lines:])
        counts.update(edges)
    repeated = {l for l, c in counts.items() if c >= 2 and len(l) < 120}
    cleaned = []
    for p in pages:
        lines = p.splitlines()
        stripped = [l.strip() for l in lines]
        keep = []
        for i, (raw, s) in enumerate(zip(lines, stripped)):
            at_edge = i < edge_lines or i >= len(lines) - edge_lines
            if at_edge and s in repeated:
                continue
            keep.append(raw)
        cleaned.append("\n".join(keep))
    return cleaned, sorted(repeated)


def _pdf_pages_pdfplumber(path: Path) -> list[str]:
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        return [(pg.extract_text() or "") for pg in pdf.pages]


def _pdf_pages_pymupdf(path: Path) -> list[str]:
    import fitz  # PyMuPDF
    with fitz.open(str(path)) as doc:
        return [pg.get_text("text") or "" for pg in doc]


def _read_docx(path: Path) -> list[str]:
    import docx
    d = docx.Document(str(path))
    return ["\n".join(p.text for p in d.paragraphs)]


def _read_text(path: Path) -> list[str]:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return [raw.decode(enc)]
        except UnicodeDecodeError:
            continue
    return [raw.decode("utf-8", errors="replace")]


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def extract_text(path: str | Path) -> dict:
    """Extract text from one resume file.

    Returns a dict with keys: text, pages, n_pages, method, status, error,
    removed_header_lines.
    """
    path = Path(path)
    out = {"text": "", "pages": [], "n_pages": 0, "method": None,
           "status": "ok", "error": "", "removed_header_lines": []}
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        out.update(status="unsupported", error=f"unsupported extension {ext}")
        return out
    if not path.exists() or path.stat().st_size == 0:
        out.update(status="empty_file", error="file missing or 0 bytes")
        return out
    pages: list[str] = []
    try:
        if ext == ".pdf":
            try:
                pages = _pdf_pages_pdfplumber(path)
                out["method"] = "pdfplumber"
            except Exception as e:  # malformed for pdfplumber -> try fitz
                out["error"] = f"pdfplumber failed: {e}"
            if not "".join(pages).strip():
                try:
                    pages = _pdf_pages_pymupdf(path)
                    out["method"] = "pymupdf"
                except Exception as e:
                    out["error"] += f" | pymupdf failed: {e}"
        elif ext == ".docx":
            pages, out["method"] = _read_docx(path), "python-docx"
        else:  # .txt / .html (HTML tags are removed later in preprocessing)
            pages, out["method"] = _read_text(path), "text"
    except Exception as e:
        out.update(status="error", error=str(e))
        return out

    pages = [fix_mojibake(p) for p in pages]
    pages, removed = _remove_repeated_headers(pages)
    text = "\n".join(pages).strip()
    out.update(pages=pages, n_pages=len(pages), text=text, removed_header_lines=removed)
    if not text:
        out["status"] = "no_text"  # e.g. scanned image PDF -> would need OCR
        out["error"] = out["error"] or "no extractable text (scanned PDF? needs OCR)"
    return out


if __name__ == "__main__":
    import sys
    r = extract_text(sys.argv[1])
    print(r["status"], r["method"], r["n_pages"], "pages")
    print(r["text"][:1500])
