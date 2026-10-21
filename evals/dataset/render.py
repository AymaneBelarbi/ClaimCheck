"""What each synthetic document says, and how it is drawn.

Claim forms, invoices and incident reports are PDFs drawn with reportlab (two layouts each). ID
cards are PNG "scans": drawn with Pillow, then rotated slightly, blurred and given sensor noise.
A document chosen to be unreadable is drawn as a low-contrast, heavily blurred PNG instead.

All randomness is passed in. PDFs use reportlab's invariant mode so the same input gives the
same bytes; PNGs carry no metadata.
"""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import numpy as np
import reportlab
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas

from claimcheck.core.types import ClaimType, DocType
from evals.dataset.policies import INSURER

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# "dmy" prints DD/MM/YYYY, which is only unambiguous when the day cannot be a month.
DATE_STYLES: tuple[str, ...] = ("iso", "long", "us_long", "dmy")


def format_date(day: date, style: str) -> str:
    month = _MONTHS[day.month - 1]
    if style == "iso":
        return day.isoformat()
    if style == "us_long":
        return f"{month} {day.day}, {day.year}"
    if style == "dmy" and day.day > 12:
        return f"{day.day:02d}/{day.month:02d}/{day.year}"
    if style in ("long", "dmy"):
        return f"{day.day} {month} {day.year}"
    raise ValueError(f"unknown date style {style!r}")


def format_money(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def amount_str(amount: Decimal) -> str:
    """Normalized gold form: plain 2-dp string, no separators or symbol."""
    return f"{amount.quantize(Decimal('0.01'))}"


# --- document content --------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimFormDoc:
    doc_type: ClassVar[DocType] = DocType.CLAIM_FORM
    claimant_name: str
    policy_number: str
    incident_date: date
    incident_type: ClaimType
    vehicle_plate: str
    vehicle: str
    phone: str
    description: str
    signed_on: date
    layout: int
    date_style: str

    def gold(self) -> dict[str, str]:
        return {
            "claimant_name": self.claimant_name,
            "policy_number": self.policy_number,
            "incident_date": self.incident_date.isoformat(),
            "incident_type": self.incident_type.value,
            "vehicle_plate": self.vehicle_plate,
            "description": self.description,
        }

    def lines(self) -> list[str]:
        d = self.date_style
        return [
            f"{INSURER} - Motor claim form",
            f"Claimant name: {self.claimant_name}",
            f"Policy number: {self.policy_number}",
            f"Vehicle registration: {self.vehicle_plate}",
            f"Date of incident: {format_date(self.incident_date, d)}",
            f"Type of incident: {self.incident_type.value.capitalize()}",
            "What happened:",
            self.description,
            f"Signed: {self.claimant_name}   Date: {format_date(self.signed_on, d)}",
        ]


@dataclass(frozen=True)
class InvoiceDoc:
    doc_type: ClassVar[DocType] = DocType.INVOICE
    vendor: str
    vendor_address: str
    vendor_phone: str
    invoice_number: str
    invoice_date: date
    bill_to: str
    vehicle_plate: str
    items: tuple[tuple[str, Decimal], ...]
    layout: int
    date_style: str

    @property
    def total(self) -> Decimal:
        return sum((amount for _, amount in self.items), Decimal("0"))

    def gold(self) -> dict[str, str]:
        items = [{"description": desc, "amount": amount_str(amt)} for desc, amt in self.items]
        return {
            "vendor": self.vendor,
            "invoice_date": self.invoice_date.isoformat(),
            "line_items": json.dumps(items, ensure_ascii=False),
            "total": amount_str(self.total),
        }

    def lines(self) -> list[str]:
        out = [
            self.vendor,
            self.vendor_address,
            f"Invoice {self.invoice_number}",
            f"Date: {format_date(self.invoice_date, self.date_style)}",
            f"Bill to: {self.bill_to}  Vehicle: {self.vehicle_plate}",
        ]
        out += [f"{desc}    {format_money(amt)}" for desc, amt in self.items]
        out.append(f"Total due: {format_money(self.total)}")
        return out


@dataclass(frozen=True)
class IdCardDoc:
    doc_type: ClassVar[DocType] = DocType.ID_CARD
    full_name: str
    id_number: str
    date_of_birth: date
    expiry_date: date
    issue_date: date
    address: str
    licence_class: str
    layout: int
    date_style: str

    def gold(self) -> dict[str, str]:
        return {
            "full_name": self.full_name,
            "id_number": self.id_number,
            "date_of_birth": self.date_of_birth.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
        }

    def lines(self) -> list[str]:
        d = self.date_style
        return [
            "DRIVER LICENCE",
            f"DL No. {self.id_number}",
            f"Name {self.full_name}",
            f"Date of birth {format_date(self.date_of_birth, d)}",
            f"Issued {format_date(self.issue_date, d)}",
            f"Expires {format_date(self.expiry_date, d)}",
            f"Class {self.licence_class}",
            self.address,
        ]


@dataclass(frozen=True)
class IncidentReportDoc:
    doc_type: ClassVar[DocType] = DocType.INCIDENT_REPORT
    agency: str
    report_number: str
    incident_date: date
    report_date: date
    location: str
    officer: str
    vehicle_plate: str
    description: str
    layout: int
    date_style: str

    def gold(self) -> dict[str, str]:
        return {
            "report_number": self.report_number,
            "incident_date": self.incident_date.isoformat(),
            "location": self.location,
            "description": self.description,
        }

    def lines(self) -> list[str]:
        d = self.date_style
        return [
            self.agency,
            f"Incident report {self.report_number}",
            f"Date of incident: {format_date(self.incident_date, d)}",
            f"Date reported: {format_date(self.report_date, d)}",
            f"Location: {self.location}",
            f"Vehicle: {self.vehicle_plate}",
            "Narrative:",
            self.description,
            f"Reporting officer: {self.officer}",
        ]


Document = ClaimFormDoc | InvoiceDoc | IdCardDoc | IncidentReportDoc

# --- PDF ---------------------------------------------------------------------------------

_W, _H = letter
_MARGIN = 54.0


def _wrap(c: Canvas, text: str, x: float, y: float, width: float, size: float = 10.5) -> float:
    c.setFont("Helvetica", size)
    for line in simpleSplit(text, "Helvetica", size, width):
        c.drawString(x, y, line)
        y -= size * 1.4
    return y


def _labelled(c: Canvas, label: str, value: str, x: float, y: float, gap: float = 150) -> None:
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(x, y, label)
    c.setFont("Helvetica", 10.5)
    c.drawString(x + gap, y, value)


def _box(c: Canvas, label: str, value: str, x: float, y: float, w: float, h: float = 34) -> None:
    c.rect(x, y, w, h)
    c.setFont("Helvetica", 7.5)
    c.drawString(x + 4, y + h - 10, label.upper())
    c.setFont("Helvetica", 11)
    c.drawString(x + 4, y + 7, value)


def _claim_form(c: Canvas, doc: ClaimFormDoc) -> None:
    d = doc.date_style
    kind = doc.incident_type.value.capitalize()
    if doc.layout == 0:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(_MARGIN, _H - 60, INSURER)
        c.setFont("Helvetica", 13)
        c.drawString(_MARGIN, _H - 80, "Motor claim form")
        c.line(_MARGIN, _H - 90, _W - _MARGIN, _H - 90)
        y = _H - 120
        rows = [
            ("Claimant name", doc.claimant_name),
            ("Policy number", doc.policy_number),
            ("Contact phone", doc.phone),
            ("Vehicle", doc.vehicle),
            ("Vehicle registration", doc.vehicle_plate),
            ("Date of incident", format_date(doc.incident_date, d)),
            ("Type of incident", kind),
        ]
        for label, value in rows:
            _labelled(c, label, value, _MARGIN, y)
            y -= 22
        y -= 10
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(_MARGIN, y, "What happened")
        y = _wrap(c, doc.description, _MARGIN, y - 18, _W - 2 * _MARGIN)
        y -= 30
        _labelled(c, "Signature", doc.claimant_name, _MARGIN, y, gap=80)
        _labelled(c, "Date signed", format_date(doc.signed_on, d), _MARGIN + 300, y, gap=80)
        return

    c.setFillGray(0.92)
    c.rect(0, _H - 70, _W, 70, stroke=0, fill=1)
    c.setFillGray(0)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(_MARGIN, _H - 45, "CLAIM NOTIFICATION")
    c.setFont("Helvetica", 10)
    c.drawRightString(_W - _MARGIN, _H - 45, f"{INSURER} motor claims")
    full = _W - 2 * _MARGIN
    half = full / 2 - 5
    y = _H - 130
    _box(c, "Policy no.", doc.policy_number, _MARGIN, y, half)
    _box(c, "Incident type", kind.upper(), _MARGIN + half + 10, y, half)
    y -= 44
    _box(c, "Policyholder / claimant", doc.claimant_name, _MARGIN, y, full)
    y -= 44
    _box(c, "Date of loss", format_date(doc.incident_date, d), _MARGIN, y, half)
    _box(c, "Telephone", doc.phone, _MARGIN + half + 10, y, half)
    y -= 44
    _box(c, "Registration (plate)", doc.vehicle_plate, _MARGIN, y, half)
    _box(c, "Make and model", doc.vehicle, _MARGIN + half + 10, y, half)
    y -= 30
    c.setFont("Helvetica", 7.5)
    c.drawString(_MARGIN + 4, y, "DESCRIPTION OF THE INCIDENT")
    end = _wrap(c, doc.description, _MARGIN + 4, y - 16, full - 8, size=11)
    c.rect(_MARGIN, end - 6, full, y - end + 16)
    y = end - 50
    c.setFont("Helvetica", 9)
    c.drawString(
        _MARGIN,
        y,
        f"I confirm the above is true. Signed {doc.claimant_name}, "
        f"{format_date(doc.signed_on, d)}.",
    )


def _invoice(c: Canvas, doc: InvoiceDoc) -> None:
    d = doc.date_style
    if doc.layout == 0:
        c.setFont("Helvetica-Bold", 15)
        c.drawString(_MARGIN, _H - 60, doc.vendor)
        c.setFont("Helvetica", 9.5)
        c.drawString(_MARGIN, _H - 76, doc.vendor_address)
        c.drawString(_MARGIN, _H - 89, f"Tel. {doc.vendor_phone}")
        c.setFont("Helvetica-Bold", 20)
        c.drawRightString(_W - _MARGIN, _H - 60, "INVOICE")
        c.setFont("Helvetica", 10)
        c.drawRightString(_W - _MARGIN, _H - 78, f"Invoice no. {doc.invoice_number}")
        c.drawRightString(_W - _MARGIN, _H - 92, f"Date: {format_date(doc.invoice_date, d)}")
        y = _H - 135
        _labelled(c, "Bill to", doc.bill_to, _MARGIN, y, gap=70)
        _labelled(c, "Vehicle", doc.vehicle_plate, _MARGIN, y - 16, gap=70)
        y -= 50
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(_MARGIN, y, "Description")
        c.drawRightString(_W - _MARGIN, y, "Amount")
        c.line(_MARGIN, y - 5, _W - _MARGIN, y - 5)
        y -= 22
        c.setFont("Helvetica", 10.5)
        for desc, amount in doc.items:
            c.drawString(_MARGIN, y, desc)
            c.drawRightString(_W - _MARGIN, y, format_money(amount))
            y -= 18
        c.line(_W - _MARGIN - 180, y + 6, _W - _MARGIN, y + 6)
        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(_W - _MARGIN - 180, y - 10, "Total due")
        c.drawRightString(_W - _MARGIN, y - 10, format_money(doc.total))
        c.setFont("Helvetica", 8.5)
        c.drawString(_MARGIN, 60, "Payment within 30 days. Thank you for your business.")
        return

    c.setFont("Helvetica-Bold", 17)
    c.drawCentredString(_W / 2, _H - 60, doc.vendor.upper())
    c.setFont("Helvetica", 9.5)
    c.drawCentredString(_W / 2, _H - 76, f"{doc.vendor_address}  |  {doc.vendor_phone}")
    c.line(_MARGIN, _H - 88, _W - _MARGIN, _H - 88)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(_MARGIN, _H - 115, "Tax invoice")
    c.setFont("Helvetica", 10)
    c.drawString(_MARGIN, _H - 133, f"Number: {doc.invoice_number}")
    c.drawString(_MARGIN, _H - 147, f"Issued: {format_date(doc.invoice_date, d)}")
    c.drawString(_W / 2, _H - 133, f"Customer: {doc.bill_to}")
    c.drawString(_W / 2, _H - 147, f"Registration: {doc.vehicle_plate}")
    y = _H - 185
    c.setFillGray(0.9)
    c.rect(_MARGIN, y - 5, _W - 2 * _MARGIN, 18, stroke=0, fill=1)
    c.setFillGray(0)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(_MARGIN + 6, y, "Item")
    c.drawString(_MARGIN + 40, y, "Work carried out")
    c.drawRightString(_W - _MARGIN - 6, y, "Line total")
    y -= 22
    c.setFont("Helvetica", 10)
    for i, (desc, amount) in enumerate(doc.items, start=1):
        c.drawString(_MARGIN + 6, y, str(i))
        c.drawString(_MARGIN + 40, y, desc)
        c.drawRightString(_W - _MARGIN - 6, y, format_money(amount))
        y -= 18
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(_W - _MARGIN - 110, y, "Amount due")
    c.drawRightString(_W - _MARGIN - 6, y, format_money(doc.total))


def _incident_report(c: Canvas, doc: IncidentReportDoc) -> None:
    d = doc.date_style
    if doc.layout == 0:
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(_W / 2, _H - 60, doc.agency.upper())
        c.setFont("Helvetica", 12)
        c.drawCentredString(_W / 2, _H - 78, "Traffic incident report")
        c.line(_MARGIN, _H - 90, _W - _MARGIN, _H - 90)
        y = _H - 120
        rows = [
            ("Report number", doc.report_number),
            ("Date of incident", format_date(doc.incident_date, d)),
            ("Date reported", format_date(doc.report_date, d)),
            ("Location", doc.location),
            ("Vehicle", doc.vehicle_plate),
            ("Reporting officer", doc.officer),
        ]
        for label, value in rows:
            _labelled(c, label, value, _MARGIN, y, gap=130)
            y -= 20
        y -= 14
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(_MARGIN, y, "Narrative")
        _wrap(c, doc.description, _MARGIN, y - 18, _W - 2 * _MARGIN)
        return

    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN, _H - 60, doc.agency)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(_MARGIN, _H - 90, f"INCIDENT REPORT  {doc.report_number}")
    c.setFont("Helvetica", 10.5)
    y = _H - 125
    for line in (
        f"Occurred on: {format_date(doc.incident_date, d)}",
        f"Place of occurrence: {doc.location}",
        f"Vehicle involved: {doc.vehicle_plate}",
        f"Report taken: {format_date(doc.report_date, d)} by {doc.officer}",
    ):
        c.drawString(_MARGIN, y, line)
        y -= 18
    y -= 12
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(_MARGIN, y, "Summary of events")
    _wrap(c, doc.description, _MARGIN, y - 18, _W - 2 * _MARGIN)


def render_pdf(doc: ClaimFormDoc | InvoiceDoc | IncidentReportDoc, path: Path) -> None:
    c = Canvas(str(path), pagesize=letter, invariant=1, pageCompression=1)
    c.setTitle(doc.doc_type.value.replace("_", " "))
    if isinstance(doc, ClaimFormDoc):
        _claim_form(c, doc)
    elif isinstance(doc, InvoiceDoc):
        _invoice(c, doc)
    else:
        _incident_report(c, doc)
    c.showPage()
    c.save()


# --- PNG scans ---------------------------------------------------------------------------

_FONT_DIR = Path(reportlab.__file__).parent / "fonts"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_FONT_DIR / ("VeraBd.ttf" if bold else "Vera.ttf")), size)


def _id_card_image(doc: IdCardDoc) -> Image.Image:
    w, h = 1012, 638
    img = Image.new("L", (w, h), 246)
    draw = ImageDraw.Draw(img)
    d = doc.date_style
    draw.rounded_rectangle((2, 2, w - 3, h - 3), radius=28, outline=120, width=3)
    draw.rectangle((3, 30, w - 4, 110), fill=205)
    draw.text((40, 45), "DRIVER LICENCE", font=_font(40, bold=True), fill=20)
    draw.text((w - 330, 58), "Dept. of Motor Vehicles", font=_font(22), fill=40)
    # Photo placeholder.
    draw.rectangle((40, 150, 270, 440), fill=190, outline=110, width=2)
    draw.ellipse((105, 190, 205, 290), fill=150)
    draw.rounded_rectangle((75, 300, 235, 440), radius=40, fill=150)
    label, value = _font(20), _font(30)
    if doc.layout == 0:
        rows = [
            ("DL NO.", doc.id_number),
            ("NAME", doc.full_name),
            ("DATE OF BIRTH", format_date(doc.date_of_birth, d)),
            ("EXPIRES", format_date(doc.expiry_date, d)),
        ]
        y = 145
        for lab, val in rows:
            draw.text((310, y), lab, font=label, fill=70)
            draw.text((310, y + 24), val, font=value, fill=15)
            y += 74
        draw.text((310, y + 4), f"ISSUED {format_date(doc.issue_date, d)}", font=label, fill=50)
        draw.text((700, y + 4), f"CLASS {doc.licence_class}", font=label, fill=50)
        draw.text((40, 560), doc.address, font=_font(22), fill=40)
    else:
        draw.text((310, 145), doc.full_name, font=_font(34, bold=True), fill=15)
        rows = [
            ("Licence no.", doc.id_number),
            ("Born", format_date(doc.date_of_birth, d)),
            ("Valid until", format_date(doc.expiry_date, d)),
            ("Issued", format_date(doc.issue_date, d)),
            ("Class", doc.licence_class),
        ]
        y = 210
        for lab, val in rows:
            draw.text((310, y + 6), lab, font=label, fill=70)
            draw.text((500, y), val, font=value, fill=15)
            y += 52
        draw.text((310, y + 20), doc.address, font=_font(22), fill=40)
    return img


def _text_page(lines: list[str]) -> Image.Image:
    """A plain page image of a document's text, used for the unreadable copies."""
    w, h = 1275, 1650
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = _font(28)
    y = 110
    for line in lines:
        words, current = line.split(" "), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) > w - 220:
                draw.text((110, y), current, font=font, fill=0)
                y += 44
                current = word
            else:
                current = candidate
        draw.text((110, y), current, font=font, fill=0)
        y += 58
    return img


def _noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def _scan(img: Image.Image, angle: float, seed: int) -> Image.Image:
    """Mild scanner artefacts: card on a scanner bed, slightly rotated, noisy, a little soft."""
    bed = Image.new("L", (img.width + 160, img.height + 160), 232)
    bed.paste(img, (80, 80))
    bed = bed.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=232)
    bed = _noise(bed, sigma=7.0, seed=seed)
    return bed.filter(ImageFilter.GaussianBlur(radius=0.7))


def _ruin(img: Image.Image, seed: int) -> Image.Image:
    """Heavy degradation: washed-out contrast, low resolution, strong blur and noise."""
    arr = np.asarray(img, dtype=np.float32)
    # Squeeze 0..255 into a narrow light band so ink and paper are barely distinguishable.
    arr = 185.0 + arr * (35.0 / 255.0)
    faded = Image.fromarray(arr.astype(np.uint8), mode="L")
    small = faded.resize((faded.width // 4, faded.height // 4), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius=1.6))
    big = small.resize(faded.size, Image.Resampling.BILINEAR)
    return _noise(big, sigma=22.0, seed=seed).filter(ImageFilter.GaussianBlur(radius=1.5))


def _save_png(img: Image.Image, path: Path) -> None:
    img.save(path, format="PNG", optimize=False, compress_level=6)


def render_id_card(doc: IdCardDoc, path: Path, angle: float, seed: int) -> None:
    _save_png(_scan(_id_card_image(doc), angle, seed), path)


def render_unreadable(doc: Document, path: Path, angle: float, seed: int) -> None:
    base = _id_card_image(doc) if isinstance(doc, IdCardDoc) else _text_page(doc.lines())
    _save_png(_ruin(base.rotate(angle, fillcolor=255, expand=False), seed), path)
