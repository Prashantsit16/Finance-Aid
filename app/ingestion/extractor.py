"""
extractor.py — Pull raw text out of a PDF, page by page.

Why PyMuPDF over pdfplumber?
  - PyMuPDF (fitz) is faster and handles most PDFs cleanly.
  - pdfplumber is better at tables but slower.
  - We use PyMuPDF first. If a page comes back empty, we try pdfplumber.
  - This "fallback" pattern is common in real pipelines.

Key concept: PDFs are NOT just text files. They're like a canvas where
text is drawn at specific coordinates. Extraction is reconstructing that
text from those coordinates — which is why it sometimes breaks.
"""

import fitz  # this is PyMuPDF. the package is called pymupdf but you import it as fitz
import pdfplumber
from pathlib import Path
from loguru import logger


def extract_pages_pymupdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page using PyMuPDF.
    Returns a list of dicts: [{page: 1, text: "..."}, ...]

    Why page-by-page?
    Because we want to keep track of WHERE in the document each piece
    of text came from. This becomes the citation later.
    """
    pages = []
    pdf_path = Path(pdf_path)

    # fitz.open() loads the PDF into memory
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text("text") extracts plain text.
        # "blocks" gives you text with position, "dict" gives even more detail.
        # Plain text is fine for our use case.
        text = page.get_text("text")

        pages.append({
            "page": page_num + 1,  # 1-indexed so it matches the actual PDF page number
            "text": text,
            "source": pdf_path.name
        })

    doc.close()
    logger.info(f"PyMuPDF extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def extract_pages_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Fallback extractor using pdfplumber.
    Slower but sometimes handles complex layouts better.
    """
    pages = []
    pdf_path = Path(pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": pdf_path.name
            })

    logger.info(f"pdfplumber extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def extract_pdf(pdf_path: str) -> list[dict]:
    """
    Main extraction function.
    Tries PyMuPDF first. Falls back to pdfplumber for empty pages.

    After this function: you have raw text per page.
    Next step: clean it (remove headers, footers, garbage).
    """
    primary = extract_pages_pymupdf(pdf_path)

    # Check if any pages came back empty — common with scanned PDFs or weird layouts
    empty_pages = [p for p in primary if len(p["text"].strip()) < 50]

    if empty_pages:
        logger.warning(
            f"{len(empty_pages)} pages returned very little text via PyMuPDF. "
            f"Running pdfplumber fallback on those pages."
        )
        # Wrap in try/except — pdfplumber can fail on corrupt or password-protected PDFs.
        # If it does, we just keep whatever PyMuPDF gave us rather than crashing.
        try:
            fallback = extract_pages_pdfplumber(pdf_path)
            fallback_map = {p["page"]: p for p in fallback}
            for page in primary:
                if len(page["text"].strip()) < 50 and page["page"] in fallback_map:
                    page["text"] = fallback_map[page["page"]]["text"]
        except Exception as e:
            logger.warning(
                f"pdfplumber fallback also failed: {e}. "
                f"Continuing with PyMuPDF output only. "
                f"If the PDF is scanned/image-based, you'll need OCR (e.g. pytesseract)."
            )

    return primary
