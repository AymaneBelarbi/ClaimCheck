import io
import shutil
from typing import Any

import pytesseract
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.pdfgen.canvas import Canvas

import pipeline.ocr
from pipeline.ocr import MAX_PAGES, OCR_TIMEOUT_SECONDS, run_ocr, span_confidence
from pipeline.types import OcrResult, OcrWord, UnsupportedDocumentError

pytestmark = [
    pytest.mark.ocr,
    pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed"),
]


def _pdf(pages: list[list[str]]) -> bytes:
    buf = io.BytesIO()
    canvas = Canvas(buf, pagesize=(612, 792))
    for lines in pages:
        canvas.setFont("Helvetica", 18)
        y = 700
        for line in lines:
            canvas.drawString(72, y, line)
            y -= 36
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _png(text: str, blur: float = 0.0, contrast: int = 0) -> bytes:
    img = Image.new("L", (1000, 300), 255)
    draw = ImageDraw.Draw(img)
    draw.text((40, 100), text, font=ImageFont.load_default(size=48), fill=contrast)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clean_pdf_text_recovered() -> None:
    result = run_ocr(_pdf([["Invoice Total 1,234.56"]]), "invoice.pdf")
    for token in ("Invoice", "Total", "1,234.56"):
        assert token in result.text
    assert result.low_confidence is False
    assert result.mean_confidence >= 0.60


def test_multipage_pdf_pages() -> None:
    result = run_ocr(_pdf([["First page alpha"], ["Second page bravo"]]), "two.pdf")
    assert [p.page_no for p in result.pages] == [1, 2]
    assert "alpha" in result.pages[0].text
    assert "bravo" in result.pages[1].text
    assert "bravo" not in result.pages[0].text
    assert result.text == "\f".join(p.text for p in result.pages)
    assert {w.page for w in result.words} == {1, 2}


def test_blurred_image_flags_low_confidence() -> None:
    clean = run_ocr(_png("Policy AUT-104233 Claimant"), "scan.png")
    assert clean.low_confidence is False
    blurred = run_ocr(_png("Policy AUT-104233 Claimant", blur=9.0, contrast=170), "scan.png")
    assert blurred.low_confidence is True
    assert blurred.mean_confidence < 0.60


def test_blank_image_has_zero_confidence() -> None:
    result = run_ocr(_png(""), "blank.png")
    assert result.words == []
    assert result.mean_confidence == 0.0
    assert result.low_confidence is True


def test_jpeg_is_read_by_signature_not_extension() -> None:
    img = Image.open(io.BytesIO(_png("Invoice Total 1,234.56"))).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    result = run_ocr(buf.getvalue(), "upload.bin")
    assert "Total" in result.text


@pytest.mark.parametrize("mode", ["CMYK", "1", "I;16"])
def test_uncommon_image_modes_are_read(mode: str) -> None:
    img = Image.open(io.BytesIO(_png("Invoice Total 1,234.56"))).convert(mode)
    buf = io.BytesIO()
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    img.save(buf, format=fmt)
    result = run_ocr(buf.getvalue(), "scan")
    assert "Total" in result.text


def test_word_offsets_index_text() -> None:
    result = run_ocr(_pdf([["Claimant Jane Doe", "Policy AUT-104233"], ["Second page"]]), "a.pdf")
    assert result.words
    for w in result.words:
        assert result.text[w.start : w.end] == w.text
        assert 0.0 <= w.confidence <= 1.0
    # Lines are separated by newlines, so line structure survives in the text.
    assert "Claimant Jane Doe\nPolicy AUT-104233" in result.text


def test_page_text_is_slice_of_document_text() -> None:
    result = run_ocr(_pdf([["First page alpha"], ["Second page bravo"]]), "two.pdf")
    first, second = result.pages
    assert result.text.startswith(first.text)
    assert result.text.endswith(second.text)


@pytest.mark.parametrize("data", [b"", b"PK\x03\x04docx-ish bytes", b"just some text"])
def test_unsupported_bytes_raise(data: bytes) -> None:
    with pytest.raises(UnsupportedDocumentError):
        run_ocr(data, "claim.docx")


def test_empty_bytes_raises() -> None:
    with pytest.raises(UnsupportedDocumentError, match="empty"):
        run_ocr(b"", "claim.pdf")


def test_docx_raises() -> None:
    with pytest.raises(UnsupportedDocumentError, match=r"claim\.docx"):
        run_ocr(b"PK\x03\x04" + b"\x00" * 64, "claim.docx")


def test_corrupt_pdf_raises_with_cause() -> None:
    with pytest.raises(UnsupportedDocumentError) as exc:
        run_ocr(b"%PDF-1.7\nthis is not a pdf", "broken.pdf")
    assert exc.value.__cause__ is not None


def test_corrupt_png_raises_with_cause() -> None:
    with pytest.raises(UnsupportedDocumentError) as exc:
        run_ocr(b"\x89PNG\r\n\x1a\n" + b"garbage", "broken.png")
    assert exc.value.__cause__ is not None


def _result() -> OcrResult:
    words = [
        OcrWord("alpha", 0.9, 1, 0, 5),
        OcrWord("beta", 0.5, 1, 6, 10),
        OcrWord("gamma", 0.1, 1, 11, 16),
    ]
    return OcrResult(
        pages=[],
        words=words,
        text="alpha beta gamma",
        mean_confidence=0.5,
        low_confidence=True,
    )


def test_a_pdf_over_the_page_limit_is_refused_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_ocr(*args: object, **kwargs: object) -> None:
        raise AssertionError("tesseract must not run")

    monkeypatch.setattr(pytesseract, "image_to_data", no_ocr)
    pages = [[f"page {n}"] for n in range(MAX_PAGES + 1)]

    with pytest.raises(UnsupportedDocumentError, match="too many pages"):
        run_ocr(_pdf(pages), "long.pdf")


def test_a_pdf_at_the_page_limit_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.ocr, "MAX_PAGES", 2)

    result = run_ocr(_pdf([["First page alpha"], ["Second page bravo"]]), "two.pdf")

    assert len(result.pages) == 2


def test_pdf_pages_are_rendered_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    render = pipeline.ocr._render_page
    ocr_page = pipeline.ocr._ocr_page

    def spy_render(*args: Any) -> Any:
        events.append("render")
        return render(*args)

    def spy_ocr(*args: Any) -> Any:
        events.append("ocr")
        return ocr_page(*args)

    monkeypatch.setattr(pipeline.ocr, "_render_page", spy_render)
    monkeypatch.setattr(pipeline.ocr, "_ocr_page", spy_ocr)

    run_ocr(_pdf([["one"], ["two"], ["three"]]), "three.pdf")

    assert events == ["render", "ocr"] * 3


def test_tesseract_gets_a_timeout_and_a_timeout_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def slow(*args: object, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_data", slow)

    with pytest.raises(UnsupportedDocumentError, match="ocr timeout") as exc:
        run_ocr(_png("Total 12.00"), "scan.png")

    assert seen["timeout"] == OCR_TIMEOUT_SECONDS
    assert "scan.png" in str(exc.value)


def test_span_confidence_averages_overlapping_words() -> None:
    r = _result()
    assert span_confidence(r, 0, 5) == pytest.approx(0.9)
    assert span_confidence(r, 3, 8) == pytest.approx(0.7)  # partial overlap counts
    assert span_confidence(r, 0, 16) == pytest.approx(0.5)


def test_span_confidence_without_words_is_zero() -> None:
    r = _result()
    assert span_confidence(r, 5, 6) == 0.0  # only whitespace
    assert span_confidence(r, 100, 120) == 0.0
    assert span_confidence(r, 4, 4) == 0.0
