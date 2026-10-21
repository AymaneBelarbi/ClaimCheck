"""Typed inputs and outputs of the pipeline stages."""

from dataclasses import dataclass


class UnsupportedDocumentError(Exception):
    """The bytes are empty, not a PDF/PNG/JPEG, or cannot be decoded as one."""


@dataclass(frozen=True)
class OcrWord:
    """One recognised word. `start`/`end` index into `OcrResult.text`."""

    text: str
    confidence: float  # 0..1
    page: int  # 1-based
    start: int
    end: int


@dataclass(frozen=True)
class OcrPage:
    page_no: int  # 1-based
    text: str


@dataclass(frozen=True)
class OcrResult:
    pages: list[OcrPage]
    words: list[OcrWord]
    text: str  # pages joined with "\f"
    mean_confidence: float
    low_confidence: bool
