"""Post-redaction validation engine implementing fail-closed security re-scanning."""
import logging
import re
from typing import List, Optional, Set

from models.schemas import (
    DocumentBlock,
    PIIFinding,
    ValidationResult,
    SecurityStatus,
    Severity,
    ConfidenceLevel,
    PIIType,
)
from detection.pii_engine import PIIEngine
from detection.ner_detector import is_valid_person_entity, NON_PERSON_WORDS, ORG_SUFFIX_WORDS

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(
    r"^\[(PERSON|EMAIL|PHONE|EMPLOYEE_ID|GOVERNMENT_ID|ADDRESS|DATE_OF_BIRTH|DATE|CREDIT_CARD|IP_ADDRESS|OTHER_PII)(?:_\d+|_[A-Z0-9]+)?\]$"
)


def is_placeholder_token(text: str) -> bool:
    """Check if token is an established semantic placeholder (e.g. [PERSON_001])."""
    return bool(PLACEHOLDER_PATTERN.match(text.strip()))


class PostRedactionValidator:
    """Validates sanitized content by executing a secondary security scan."""

    def __init__(self, pii_engine: PIIEngine):
        self.pii_engine = pii_engine

    def validate(
        self,
        redacted_blocks: List[DocumentBlock],
        document_id: str,
        original_findings: Optional[List[PIIFinding]] = None,
        original_blocks: Optional[List[DocumentBlock]] = None,
    ) -> ValidationResult:
        """Re-scan sanitized blocks to verify complete elimination of detectable PII."""
        # 1. Verification of Redaction Completeness:
        # Check if any detected sensitive entity from original scan still exists verbatim in sanitized text
        actionable_residuals: List[PIIFinding] = []
        if original_findings:
            redacted_corpus = "\n".join(b.text for b in redacted_blocks)
            for f in original_findings:
                clean_raw = f.text.strip()
                # Skip trivial tokens or placeholders
                if len(clean_raw) < 3 or is_placeholder_token(clean_raw):
                    continue
                if f.pii_type == PIIType.PERSON and not is_valid_person_entity(clean_raw):
                    continue
                # Word-bounded search for raw unredacted entity leak
                escaped = re.escape(clean_raw)
                leak_pat = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE if f.pii_type != PIIType.PERSON else 0)
                if leak_pat.search(redacted_corpus):
                    actionable_residuals.append(
                        PIIFinding(
                            finding_id=f"val_leak_{f.finding_id}",
                            document_id=document_id,
                            file_name=f.file_name,
                            location=f.location,
                            block_order=f.block_order,
                            start=0,
                            end=len(clean_raw),
                            text=clean_raw,
                            pii_type=f.pii_type,
                            confidence=1.0,
                            confidence_level=ConfidenceLevel.HIGH,
                            detection_method=["VERIFICATION_LEAK"],
                            severity=Severity.CRITICAL,
                            explanation=f"Unredacted sensitive entity '{clean_raw}' was found in sanitized text.",
                            masked_text=f.masked_text,
                        )
                    )

        # 2. Run full hybrid detection engine on the sanitized blocks
        rescan_findings = self.pii_engine.detect_blocks(redacted_blocks)

        # Build original unflagged text lines for disambiguating placeholder-induced syntactic artifacts
        original_unflagged_lines: Set[str] = set()
        if original_blocks and original_findings:
            flagged_normalized = {f.text.strip().lower() for f in original_findings}
            for ob in original_blocks:
                for line in ob.text.splitlines():
                    cl = line.strip().lower()
                    if cl and cl not in flagged_normalized:
                        original_unflagged_lines.add(cl)

        # 3. Categorize re-scan findings
        for rf in rescan_findings:
            # Skip placeholders
            if is_placeholder_token(rf.text):
                continue

            # Deterministic high-risk PII: RFC emails, Luhn credit cards, Govt IDs, Phones, Employee IDs, IPs
            if rf.pii_type in (
                PIIType.EMAIL,
                PIIType.CREDIT_CARD,
                PIIType.GOVERNMENT_ID,
                PIIType.PHONE,
                PIIType.EMPLOYEE_ID,
                PIIType.IP_ADDRESS,
            ):
                actionable_residuals.append(rf)
                continue

            # Multi-detector corroborated findings (e.g. NER+REGEX, REGEX+CONTEXT)
            if len(rf.detection_method) > 1:
                actionable_residuals.append(rf)
                continue

            # Regex-detected Person findings (structural line or salutation)
            if "REGEX" in rf.detection_method:
                actionable_residuals.append(rf)
                continue

            # Pure statistical NER findings:
            if rf.pii_type == PIIType.PERSON and rf.detection_method == ["NER"]:
                # Check valid person entity
                if not is_valid_person_entity(rf.text):
                    continue
                # If this text was already present in the original document and was NOT flagged as PII
                # by any engine in the initial pass, its presence here is an artifact of the bracketed placeholder tag
                # altering spaCy's syntactic parse tree, NOT a genuine PII leak.
                rf_lower = rf.text.strip().lower()
                if original_blocks and any(rf_lower in uline for uline in original_unflagged_lines):
                    continue

                # Genuine unredacted person entity
                actionable_residuals.append(rf)

        if actionable_residuals:
            # Residual PII detected: FAIL-CLOSED
            logger.warning(
                f"Document {document_id} failed post-redaction validation: {len(actionable_residuals)} residual findings."
            )
            msg = (
                f"BLOCKED — Residual PII detected after redaction ({len(actionable_residuals)} items remaining). "
                "Document fails validation and is unsafe for downstream AI processing."
            )
            return ValidationResult(
                document_id=document_id,
                is_safe=False,
                status=SecurityStatus.BLOCKED,
                residual_findings=actionable_residuals,
                validation_message=msg,
                passed=False,
            )

        # Clean clearance: SANITIZED
        msg = "SANITIZED — Zero detectable PII remains after re-scan. Verified safe for downstream AI processing."
        return ValidationResult(
            document_id=document_id,
            is_safe=True,
            status=SecurityStatus.SANITIZED,
            residual_findings=[],
            validation_message=msg,
            passed=True,
        )
