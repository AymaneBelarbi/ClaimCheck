"""What the extraction call must return: one output model per document type.

Every field is an `LLMField`, so the model has to say how sure it is and quote where it read
the value. Dates and amounts are plain strings here, copied as printed; `pipeline.extract`
normalizes them afterwards and refuses anything ambiguous. The field descriptions are sent to
the model in the prompt, so they are written as instructions.
"""

from pydantic import BaseModel, Field

from core.types import DocType


class LLMField[T](BaseModel):
    value: T | None = Field(
        description=(
            "The value as printed in the document, or null if the document does not show it."
        )
    )
    confidence: float = Field(
        description="Probability from 0 to 1 that value is exactly what the document says."
    )
    source_text: str | None = Field(
        description=(
            "A short quote copied character for character from the document text that contains "
            "the value; null when value is null."
        )
    )


class LineItem(BaseModel):
    description: str = Field(description="The part or work on this line, as printed.")
    amount: str = Field(description="The amount on this line exactly as printed, e.g. $1,234.50.")


class ClaimFormOut(BaseModel):
    claimant_name: LLMField[str] = Field(
        description="Full name of the policyholder making the claim, as printed on the form."
    )
    policy_number: LLMField[str] = Field(
        description="The insurance policy number, e.g. AUT-123456."
    )
    incident_date: LLMField[str] = Field(
        description=(
            "Date the incident happened (not the date the form was signed), exactly as printed."
        )
    )
    incident_type: LLMField[str] = Field(
        description="The type of incident as stated on the form: collision, glass or theft."
    )
    vehicle_plate: LLMField[str] = Field(
        description="The vehicle registration (licence plate) number."
    )
    description: LLMField[str] = Field(
        description="The claimant's account of what happened, copied in full."
    )


class IdCardOut(BaseModel):
    full_name: LLMField[str] = Field(
        description="The holder's name exactly as printed, in the order printed."
    )
    id_number: LLMField[str] = Field(
        description="The licence or ID card number (DL NO., licence no. and the like)."
    )
    date_of_birth: LLMField[str] = Field(
        description="The holder's date of birth, exactly as printed."
    )
    expiry_date: LLMField[str] = Field(
        description="The date the card expires (not the issue date), exactly as printed."
    )


class InvoiceOut(BaseModel):
    vendor: LLMField[str] = Field(
        description="Name of the garage or business that issued the invoice (not the customer)."
    )
    invoice_date: LLMField[str] = Field(
        description="The date the invoice was issued, exactly as printed."
    )
    line_items: LLMField[list[LineItem]] = Field(
        description=(
            "Every charged line in table order, each with its description and amount. Pair "
            "descriptions and amounts by their order in the table. Exclude the total. "
            "source_text quotes the first line's description."
        )
    )
    total: LLMField[str] = Field(
        description="The total amount due on the invoice, exactly as printed."
    )


class IncidentReportOut(BaseModel):
    report_number: LLMField[str] = Field(
        description="The incident or police report reference number, e.g. IR-2024-123456."
    )
    incident_date: LLMField[str] = Field(
        description=(
            "Date the incident occurred (not the date it was reported), exactly as printed."
        )
    )
    location: LLMField[str] = Field(
        description="Where the incident happened: the address or place, as printed."
    )
    description: LLMField[str] = Field(
        description="The officer's summary of events, copied in full."
    )


# No entry for DocType.OTHER: those documents go to a person, never to extraction.
SCHEMAS: dict[DocType, type[BaseModel]] = {
    DocType.CLAIM_FORM: ClaimFormOut,
    DocType.ID_CARD: IdCardOut,
    DocType.INVOICE: InvoiceOut,
    DocType.INCIDENT_REPORT: IncidentReportOut,
}
