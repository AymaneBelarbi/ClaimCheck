"""Exemplar texts for the embedding classifier, one list per document type.

These describe each kind of document in generic words, the way a form template or a
letterhead would, so the centroid sits near any realistic document of the type. They are
written independently of the synthetic dataset: nothing here is a copy of a generated file.

`other` has no prototype. A document only becomes `other` through the fallback path or
because its OCR was too poor to classify.
"""

from claimcheck.core.types import DocType

PROTOTYPES: dict[DocType, list[str]] = {
    DocType.CLAIM_FORM: [
        "Motor insurance claim form. Policyholder name, policy number, contact telephone. "
        "Vehicle make and model, registration. Date and type of incident. "
        "Describe what happened. Signed and dated by the policyholder.",
        "Vehicle damage claim submitted to the insurer. Please complete all sections. "
        "Insured person, policy reference, date of loss, kind of loss, "
        "a short account of the event, declaration and signature.",
        "Auto insurance notification of loss. Name of the person making the claim, "
        "policy no., phone number, car registration number, when the incident took place, "
        "whether it was an accident, a break-in or broken glass, and a written description.",
        "Claimant statement for an auto policy. I confirm the details above are true and "
        "complete to the best of my knowledge. Claimant signature and date signed.",
    ],
    DocType.ID_CARD: [
        "Driving licence. Licence number, surname and given names, date of birth, "
        "date of issue, expiry date, address of the holder. Issued by the motor vehicle "
        "licensing authority.",
        "Government photo identification card. Holder's full name, ID number, "
        "born, valid until, residential address, issuing department.",
        "Driver license card, class B. Licence no., name, DOB, issued, expires, "
        "licensing office, holder address.",
        "Identity card of the license holder: document number, last name first name, "
        "date of birth, expiration date, issuing authority.",
    ],
    DocType.INVOICE: [
        "Tax invoice from a vehicle repair garage. Invoice number, invoice date, customer, "
        "vehicle registration. Parts and labour itemised with line totals. Total amount due.",
        "Workshop repair bill. Description of work carried out, quantity, price per item, "
        "labour hours, subtotal, tax, grand total payable to the body shop.",
        "Invoice for auto glass replacement and bodywork. Bill to the customer, "
        "itemised charges for parts and fitting, amount due, payment terms within thirty days.",
        "Garage estimate and invoice: windscreen, bumper, paint and refinish, "
        "labour charge, total including tax, supplier address and telephone.",
    ],
    DocType.INCIDENT_REPORT: [
        "Police incident report. Report number, date and place of occurrence, "
        "vehicles involved, reporting officer and badge number, summary of events.",
        "Traffic accident report taken by the local police department. Time and location, "
        "road and weather conditions, injuries, statements from the parties, officer name.",
        "Crime report for a stolen vehicle. Reported by the owner, last seen location, "
        "registration of the vehicle, date reported, investigating officer.",
        "Law enforcement incident record: case number, offence or event type, "
        "address where it occurred, narrative of what the officer found on arrival.",
    ],
}
