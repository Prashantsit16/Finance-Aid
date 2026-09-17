"""
cleaner.py — Make raw extracted text actually usable.

Why do we need cleaning?
PDFs are formatted for HUMANS reading on screen. When you extract text you get:
  - Page numbers: "Page 47 of 213"
  - Repeated headers: "ANNUAL REPORT 2023 | CONFIDENTIAL"
  - Repeated footers: "Infosys Limited | CIN: L85110KA1981PLC013115"
  - Weird hyphena-   tion across lines
  - Multiple blank lines where there was a visual gap
  - Non-breaking spaces, weird unicode characters

If we feed this dirty text to the embedding model, we get dirty embeddings.
Garbage in, garbage out.

What we do NOT do:
  - We don't strip all punctuation — it carries meaning
  - We don't lowercase everything — "EBITDA" vs "ebitda" matters for BM25 later
  - We don't remove numbers — financial docs ARE numbers
"""

import re
from loguru import logger


# Common patterns in annual reports and legal docs that are noise
_HEADER_FOOTER_PATTERNS = [
    r"Page\s+\d+\s+of\s+\d+",          # "Page 47 of 213"
    r"^\d+\s*$",                         # lone page numbers
    r"CONFIDENTIAL",                     # boilerplate
    r"Annual Report \d{4}",             # repeated title
    r"©\s*\d{4}.*",                     # copyright lines
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                      for p in _HEADER_FOOTER_PATTERNS]


def remove_headers_footers(text: str) -> str:
    """
    Remove common header/footer patterns.
    These repeat every page and add noise without adding information.
    """
    for pattern in _COMPILED_PATTERNS:
        text = pattern.sub("", text)
    return text


def fix_hyphenation(text: str) -> str:
    """
    PDF line wrapping sometimes splits words with a hyphen at the line break.
    Example: "opera-\ntional" → "operational"

    We fix this by joining word- followed by newline.
    """
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple blank lines into one.
    Strip trailing whitespace.
    Replace non-breaking spaces with regular spaces.

    We keep single newlines — they often signal paragraph breaks
    which matter for meaning.
    """
    # non-breaking space and other unicode spaces -> regular space
    text = text.replace("\xa0", " ").replace("\u2009", " ")

    # more than 2 newlines -> 2 newlines (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Remove lines that are too short to be meaningful.
    Things like lone letters, stray punctuation from table borders, etc.

    min_length=3 means we keep "AI" but drop "—" or "."
    """
    lines = text.splitlines()
    cleaned = [line for line in lines if len(line.strip()) >= min_length or line.strip() == ""]
    return "\n".join(cleaned)


def clean_page(page: dict) -> dict:
    """
    Apply all cleaning steps to a single page dict.
    Returns the same dict structure with cleaned text.

    The cleaning is non-destructive — we return a new dict,
    not modifying the original. Good practice.
    """
    text = page["text"]
    original_len = len(text)

    text = remove_headers_footers(text)
    text = fix_hyphenation(text)
    text = normalize_whitespace(text)
    text = remove_short_lines(text)

    cleaned_len = len(text)
    reduction = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)

    if reduction > 30:
        logger.debug(
            f"Page {page['page']} of {page['source']}: "
            f"removed {reduction}% of text during cleaning. "
            f"Check if too aggressive."
        )

    return {**page, "text": text}


def clean_document(pages: list[dict]) -> list[dict]:
    """
    Clean all pages. Filter out pages that are empty after cleaning.

    Why filter empty pages?
    They might be blank separator pages, image-only pages (charts, graphs).
    They contribute nothing to retrieval but add noise.
    """
    cleaned = [clean_page(p) for p in pages]
    non_empty = [p for p in cleaned if len(p["text"].strip()) > 20]

    dropped = len(pages) - len(non_empty)
    if dropped > 0:
        logger.info(f"Dropped {dropped} empty/near-empty pages after cleaning")

    return non_empty
