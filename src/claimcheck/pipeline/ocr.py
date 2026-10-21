"""OCR stage: PDF/PNG/JPEG bytes in, text with per-word confidence and offsets out.

The upload limits cap bytes, not work: a small PDF can hold thousands of pages. A document
over MAX_PAGES is refused before anything is rendered, PDF pages are rendered and read one at
a time so only one page image is in memory, and each Tesseract run gets OCR_TIMEOUT_SECONDS.
A refused or timed-out document is unsupported, which sends it to a person as `other`.
"""

import io
from collections.abc import Iterator

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageOps
from pypdfium2 import PdfiumError

from claimcheck.pipeline.types import OcrPage, OcrResult, OcrWord, UnsupportedDocumentError

LOW_CONFIDENCE_THRESHOLD = 0.60
PDF_DPI = 300
MAX_PAGES = 20
OCR_TIMEOUT_SECONDS = 60

_PDF_MAGIC = b"%PDF"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

_LineKey = tuple[int, int, int]


def run_ocr(data: bytes, filename: str) -> OcrResult:
    """Recognise the text of a PDF, PNG or JPEG.

    The format is taken from the leading bytes; `filename` only appears in error messages.
    """
    if not data:
        raise UnsupportedDocumentError(f"{filename}: empty file")

    images: Iterator[Image.Image]
    if data.startswith(_PDF_MAGIC):
        images = _pdf_pages(data, filename)
    elif data.startswith((_PNG_MAGIC, _JPEG_MAGIC)):
        images = iter([_load_image(data, filename)])
    else:
        raise UnsupportedDocumentError(f"{filename}: not a PDF, PNG or JPEG")

    pages: list[OcrPage] = []
    words: list[OcrWord] = []
    # Offsets are recorded while the text is built, so text[w.start:w.end] == w.text holds
    # without searching for the word afterwards.
    offset = 0
    for page_no, image in enumerate(images, start=1):
        if page_no > 1:
            offset += 1  # the "\f" between pages
        try:
            page_text, page_words = _ocr_page(image, page_no, offset)
        except RuntimeError as exc:
            # pytesseract kills the process and raises a plain RuntimeError on its timeout.
            if "timeout" not in str(exc).lower():
                raise
            raise UnsupportedDocumentError(
                f"{filename}: ocr timeout on page {page_no} after {OCR_TIMEOUT_SECONDS} s"
            ) from exc
        pages.append(OcrPage(page_no=page_no, text=page_text))
        words.extend(page_words)
        offset += len(page_text)

    mean = sum(w.confidence for w in words) / len(words) if words else 0.0
    return OcrResult(
        pages=pages,
        words=words,
        text="\f".join(p.text for p in pages),
        mean_confidence=mean,
        low_confidence=mean < LOW_CONFIDENCE_THRESHOLD,
    )


def span_confidence(ocr: OcrResult, start: int, end: int) -> float:
    """Mean confidence of the words overlapping [start, end); 0.0 if no word does."""
    if end <= start:
        return 0.0
    hits = [w.confidence for w in ocr.words if w.start < end and w.end > start]
    return sum(hits) / len(hits) if hits else 0.0


def _pdf_pages(data: bytes, filename: str) -> Iterator[Image.Image]:
    """The pages as images, rendered one at a time as the caller asks for them.

    The page count is checked before the first yield, so an oversized PDF is refused as soon
    as the caller starts reading it.
    """
    try:
        pdf = pdfium.PdfDocument(data)
    except PdfiumError as exc:
        raise UnsupportedDocumentError(f"{filename}: cannot open PDF ({exc})") from exc
    try:
        count = len(pdf)
        if count == 0:
            raise UnsupportedDocumentError(f"{filename}: PDF has no pages")
        if count > MAX_PAGES:
            raise UnsupportedDocumentError(
                f"{filename}: too many pages ({count}; at most {MAX_PAGES} are read)"
            )
        for index in range(count):
            try:
                image = _render_page(pdf, index)
            except PdfiumError as exc:
                raise UnsupportedDocumentError(f"{filename}: cannot render PDF ({exc})") from exc
            yield image
    finally:
        pdf.close()


def _render_page(pdf: pdfium.PdfDocument, index: int) -> Image.Image:
    image: Image.Image = pdf[index].render(scale=PDF_DPI / 72).to_pil()
    return image


def _load_image(data: bytes, filename: str) -> Image.Image:
    try:
        opened = Image.open(io.BytesIO(data))
        opened.load()
        img = ImageOps.exif_transpose(opened)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UnsupportedDocumentError(f"{filename}: cannot decode image ({exc})") from exc
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    if img.mode not in ("L", "RGB"):
        # CMYK (common in scanner JPEGs), I;16, 1-bit and the like: pytesseract re-saves the
        # image as PNG, which cannot hold every mode.
        return img.convert("RGB")
    return img


def _ocr_page(image: Image.Image, page_no: int, offset: int) -> tuple[str, list[OcrWord]]:
    """Return the page text and its words, with offsets shifted by `offset`."""
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, timeout=OCR_TIMEOUT_SECONDS
    )

    parts: list[str] = []
    words: list[OcrWord] = []
    length = 0
    prev_line: _LineKey | None = None
    rows = zip(
        data["text"],
        data["conf"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
        strict=True,
    )
    for raw_text, raw_conf, block, par, line in rows:
        conf = float(raw_conf)
        text = str(raw_text).strip()
        if conf < 0 or not text:
            continue  # -1 marks layout boxes (page/block/line), not words
        key = (int(block), int(par), int(line))
        if prev_line is not None:
            sep = " " if key == prev_line else "\n"
            parts.append(sep)
            length += 1
        start = length
        parts.append(text)
        length += len(text)
        words.append(
            OcrWord(
                text=text,
                confidence=conf / 100.0,
                page=page_no,
                start=offset + start,
                end=offset + length,
            )
        )
        prev_line = key
    return "".join(parts), words
