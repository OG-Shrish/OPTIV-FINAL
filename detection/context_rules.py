"""Context-based PII detection rules.

Identifies PII when preceded by business evidence label indicators (e.g., 'Employee ID:',
'Contact:', 'DOB:', etc.) and boosts detection confidence.
"""
import re
from typing import List

from models.schemas import PIIFinding, PIIType, Severity, ConfidenceLevel
from utils.helpers import mask_sensitive_value
from detection.ner_detector import is_valid_person_entity

PLACEHOLDER_REGEX = re.compile(
    r"^\[(PERSON|EMAIL|PHONE|EMPLOYEE_ID|GOVERNMENT_ID|ADDRESS|DATE_OF_BIRTH|CREDIT_CARD|IP_ADDRESS|OTHER_PII)_\d+\]$"
)

NON_ID_WORDS = {
    "none", "null", "nil", "na", "n/a", "nan", "unknown", "required", "mandatory",
    "optional", "pending", "applicable", "available", "unavailable", "valid",
    "invalid", "verified", "true", "false", "yes", "no", "standard", "basic",
    "original", "copy", "attached", "provided", "check", "see", "below", "above",
    "indian", "national", "domestic", "international", "issued", "holder"
}


def is_valid_id_token(val: str) -> bool:
    """Validate that an ID candidate token represents a real identifier and not an English status word."""
    cleaned = val.strip().strip(":=-,. ")
    if len(cleaned) < 3 or len(cleaned) > 25:
        return False
    if cleaned.lower() in NON_ID_WORDS:
        return False
    # Genuine identifiers (Employee ID, Govt ID, Passport) must have at least one digit
    # or contain a hyphen/underscore with alphanumeric characters
    if not re.search(r"\d", cleaned) and not re.search(r"[-_]", cleaned):
        return False
    return True


# Rule definitions: (Regex Pattern, PII Type, Confidence, Severity, Explanation)
CONTEXT_RULES = [
    # Employee ID Cues
    (
        re.compile(r"(?i)\b(?:Employee\s+(?:ID|Id|Number|No\.?)|Emp\s*#|Staff\s+ID)\s*[:=-]?\s*([A-Za-z0-9_-]{3,15})"),
        PIIType.EMPLOYEE_ID,
        0.98,
        Severity.HIGH,
        "Preceded by explicit employee identifier keyword label.",
    ),
    # Government ID Cues
    (
        re.compile(r"(?i)\b(?:Government\s+ID|Govt\s+ID|National\s+ID|Passport\s*(?:No\.?|Number|#|ID)|Passport\s*[:=-]|Tax\s+ID)\s*[:=-]?\s*([A-Za-z0-9_-]{4,20})"),
        PIIType.GOVERNMENT_ID,
        0.98,
        Severity.CRITICAL,
        "Preceded by explicit government/national ID label.",
    ),
    # Date of Birth Cues
    (
        re.compile(r"(?i)\b(?:Date\s+of\s+Birth|DOB|Birth\s+Date|Born\s+on)\s*[:=-]?\s*(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"),
        PIIType.DATE_OF_BIRTH,
        0.96,
        Severity.MEDIUM,
        "Preceded by explicit Date of Birth keyword label.",
    ),
    # Contact / Passenger / Candidate / Person Name Cues (e.g. 'Passenger: Shrish', 'Contact: John Smith', 'Candidate: Vikas Ahankari')
    (
        re.compile(r"(?i)\b(?:Candidate(?:\s+Name)?|Applicant(?:\s+Name)?|Passenger(?:\s+Name)?|Name|Full\s+Name|First\s+Name|Last\s+Name|Traveler(?:\s+Name)?|Traveller(?:\s+Name)?|Customer(?:\s+Name)?|Guest(?:\s+Name)?|Contact(?:\s+Person)?|Assigned\s+To|Account\s+Owner|Manager|Employee|Owner|Investigator|Doctor|Patient|Attendee|Author|Consultant|Developer|Engineer|Auditor|Reporter|Prepared\s+By|Submitted\s+By|Authored\s+By|Reported\s+By|Created\s+By|Reviewed\s+By|Lead|By)\s*[:=-][ \t]*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,3})"),
        PIIType.PERSON,
        0.96,
        Severity.HIGH,
        "Person name preceded by organizational role, ticket header, passenger, or candidate label.",
    ),
    (
        re.compile(r"(?i)\b(?:Candidate(?:\s+Name)?|Applicant(?:\s+Name)?|Passenger(?:\s+Name)?|Name|Full\s+Name|First\s+Name|Last\s+Name|Traveler(?:\s+Name)?|Traveller(?:\s+Name)?|Customer(?:\s+Name)?|Guest(?:\s+Name)?|Contact(?:\s+Person)?|Assigned\s+To|Account\s+Owner|Manager|Employee|Owner|Investigator|Doctor|Patient|Attendee|Author|Consultant|Developer|Engineer|Auditor|Reporter|Prepared\s+By|Submitted\s+By|Authored\s+By|Reported\s+By|Created\s+By|Reviewed\s+By|Lead|By)\s*[:=-][ \t]*([A-Z]{2,}(?:[ \t]+[A-Z]{2,}){0,3})"),
        PIIType.PERSON,
        0.95,
        Severity.HIGH,
        "All-caps person name preceded by candidate, passenger, or role label.",
    ),
    # Honorific / Salutation Person Name Cues (e.g. 'Mr. Shrish', 'Dr. Watson', 'Shri Shrish')
    (
        re.compile(r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Shri|Smt\.?)[ \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,3})\b"),
        PIIType.PERSON,
        0.96,
        Severity.HIGH,
        "Person name preceded by honorific title or personal salutation.",
    ),
    # Email with Label
    (
        re.compile(r"(?i)\b(?:Email(?:\s+Address)?|E-mail|Mail|Mailbox|Contact\s+Email)\s*[:=-]?[ \t]*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,})"),
        PIIType.EMAIL,
        0.99,
        Severity.HIGH,
        "Email address with explicit contextual label header.",
    ),
    # Phone with Label
    (
        re.compile(r"(?i)\b(?:Phone|Mobile|Mob|Tel|Telephone|Cell(?:\s+Number|\s+No\.?)?|Contact(?:\s+No\.?|\s+Number)?|Ph\.?)\s*[:=-]?[ \t]*(\+?[\d\s().-]{7,20})"),
        PIIType.PHONE,
        0.98,
        Severity.HIGH,
        "Phone number with explicit telephone/mobile keyword label.",
    ),
    # Address with Label
    (
        re.compile(r"(?i)\b(?:Address|Billing\s+Address|Home\s+Address|Residence)\s*[:=-]\s*([0-9]{1,5}\s+[A-Za-z0-9\s,.-]{10,60})"),
        PIIType.ADDRESS,
        0.88,
        Severity.MEDIUM,
        "Postal or residential location preceded by address label.",
    ),
]


class ContextDetector:
    """Detects PII entities based on contextual prefix keywords and cues."""

    def detect(
        self,
        text: str,
        document_id: str,
        file_name: str,
        location: str,
        block_order: int
    ) -> List[PIIFinding]:
        findings: List[PIIFinding] = []
        if not text or not text.strip():
            return findings

        for regex, pii_type, confidence, severity, explanation in CONTEXT_RULES:
            for match in regex.finditer(text):
                # group(1) contains the target PII entity
                try:
                    val = match.group(1).strip()
                    start_char = match.start(1)
                    end_char = match.end(1)
                except IndexError:
                    val = match.group(0).strip()
                    start_char = match.start(0)
                    end_char = match.end(0)

                if not val or PLACEHOLDER_REGEX.match(val):
                    continue

                # Validation per PII type to prevent false positive pattern matches
                if pii_type in (PIIType.EMPLOYEE_ID, PIIType.GOVERNMENT_ID):
                    if not is_valid_id_token(val):
                        continue
                elif pii_type == PIIType.PERSON:
                    if not is_valid_person_entity(val):
                        continue
                elif pii_type == PIIType.PHONE:
                    digits = re.sub(r"\D", "", val)
                    if len(digits) < 7 or len(digits) > 15:
                        continue
                    if len(set(digits)) <= 2 or re.search(r"0{5,}", digits):
                        continue
                elif pii_type == PIIType.EMAIL:
                    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$", val):
                        continue

                conf_level = (
                    ConfidenceLevel.HIGH if confidence >= 0.90
                    else (ConfidenceLevel.MEDIUM if confidence >= 0.70 else ConfidenceLevel.LOW)
                )

                findings.append(
                    PIIFinding(
                        finding_id=f"ctx_{pii_type.value.lower()}_{block_order}_{start_char}",
                        document_id=document_id,
                        file_name=file_name,
                        location=location,
                        block_order=block_order,
                        start=start_char,
                        end=end_char,
                        text=val,
                        pii_type=pii_type,
                        confidence=confidence,
                        confidence_level=conf_level,
                        detection_method=["CONTEXT"],
                        severity=severity,
                        explanation=explanation,
                        masked_text=mask_sensitive_value(val, pii_type.value),
                    )
                )

        # Document header candidate name context: Primary subject in header context
        if block_order <= 1 or ("page_1" in location and "block_1" in location):
            header_name_pattern = re.compile(r"(?m)^[ \t]*([A-Z][a-z]{1,25}(?:\s+[A-Z][a-z]{1,25}){0,3})[ \t\r]*$")
            for match in header_name_pattern.finditer(text):
                val = match.group(1).strip()
                if not val or PLACEHOLDER_REGEX.match(val):
                    continue
                if is_valid_person_entity(val):
                    start_char = match.start(1)
                    end_char = match.end(1)
                    if not any(f.start <= start_char and f.end >= end_char for f in findings if f.pii_type == PIIType.PERSON):
                        findings.append(
                            PIIFinding(
                                finding_id=f"ctx_candidate_header_{block_order}_{start_char}",
                                document_id=document_id,
                                file_name=file_name,
                                location=location,
                                block_order=block_order,
                                start=start_char,
                                end=end_char,
                                text=val,
                                pii_type=PIIType.PERSON,
                                confidence=0.94,
                                confidence_level=ConfidenceLevel.HIGH,
                                detection_method=["CONTEXT"],
                                severity=Severity.HIGH,
                                explanation="Candidate subject name identified in primary document header context.",
                                masked_text=mask_sensitive_value(val, "PERSON"),
                            )
                        )

        return findings
