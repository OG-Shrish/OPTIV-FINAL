"""Tests for post-redaction validation, fail-closed security model, and end-to-end pipeline."""
from pipeline import GatewayPipeline
from models.schemas import SecurityStatus, DocumentBlock, PIIFinding, PIIType, Severity, ConfidenceLevel
from redaction.validator import PostRedactionValidator
from detection.pii_engine import PIIEngine


def test_end_to_end_pipeline_success():
    """Verify that an artifact with PII is cleanly redacted, re-scanned, and marked SANITIZED."""
    pipeline = GatewayPipeline()
    sample_text = (
        "CONFIDENTIAL AUDIT REPORT\n\n"
        "Lead Investigator: John Smith\n"
        "Official Email: john.smith@secure-corp.com\n"
        "Contact Phone: +1 212 555 1234\n"
        "Employee ID: EMP-10293\n\n"
        "All procedures followed standard operating guideline #8410."
    )
    file_bytes = sample_text.encode("utf-8")

    result = pipeline.process_document(file_bytes, "investigation_audit.txt")

    assert result.security_status == SecurityStatus.SANITIZED
    assert result.validation_result is not None
    assert result.validation_result.passed is True
    assert len(result.validation_result.residual_findings) == 0
    assert len(result.pii_findings) >= 3

    # Sanitized output must not contain original values
    sanitized_text = result.redaction_result.redacted_full_text
    assert "john.smith@secure-corp.com" not in sanitized_text
    assert "+1 212 555 1234" not in sanitized_text
    assert "[EMAIL_001]" in sanitized_text or "[EMAIL_" in sanitized_text


def test_fail_closed_on_empty_file():
    """Verify pipeline fails closed on empty files."""
    pipeline = GatewayPipeline()
    result = pipeline.process_document(b"", "empty_document.txt")

    assert result.security_status == SecurityStatus.BLOCKED
    assert "empty" in result.error_message.lower()


def test_fail_closed_on_residual_pii():
    """Verify validator BLOCKS a document if any residual PII is detected after redaction."""
    pii_engine = PIIEngine()
    validator = PostRedactionValidator(pii_engine=pii_engine)

    # Simulate a buggy or incomplete redaction where an email was missed
    leaked_blocks = [
        DocumentBlock(
            document_id="d_leak",
            file_name="leak.txt",
            file_type="txt",
            location="p1",
            block_type="paragraph",
            text="Sanitized header [PERSON_001], but leaked email remaining: leak@vulnerable.org",
            order=0
        )
    ]

    val_result = validator.validate(leaked_blocks, "d_leak")

    assert val_result.is_safe is False
    assert val_result.passed is False
    assert val_result.status == SecurityStatus.BLOCKED
    assert len(val_result.residual_findings) > 0
    assert any("leak@vulnerable.org" in f.text for f in val_result.residual_findings)


def test_clean_document_passes():
    """Verify clean document with zero PII is validated as SANITIZED."""
    pipeline = GatewayPipeline()
    clean_text = (
        "SYSTEM HEALTH SUMMARY\n\n"
        "All cloud clusters report nominal health status.\n"
        "Database latency is 14ms and replication lag is zero seconds.\n"
        "Quarterly maintenance scheduled for Q4 2026."
    )
    result = pipeline.process_document(clean_text.encode("utf-8"), "clean_status.txt")

    assert result.security_status == SecurityStatus.SANITIZED
    assert len(result.pii_findings) == 0
    assert result.validation_result.is_safe is True


def test_validation_raw_entity_leak_blocked():
    """Verify that if an entity detected in the initial pass is not redacted and leaks into sanitized text, validator blocks."""
    pii_engine = PIIEngine()
    validator = PostRedactionValidator(pii_engine=pii_engine)

    orig_finding = PIIFinding(
        finding_id="f_orig",
        document_id="d_leak2",
        file_name="leak2.txt",
        location="p1",
        block_order=0,
        start=0,
        end=10,
        text="Alice Smith",
        pii_type=PIIType.PERSON,
        confidence=0.95,
        confidence_level=ConfidenceLevel.HIGH,
        detection_method=["NER"],
        severity=Severity.HIGH,
    )
    # Block still contains "Alice Smith" unredacted
    b = DocumentBlock("d_leak2", "leak2.txt", "txt", "p1", "text", "Hello Alice Smith how are you", 0)

    val_res = validator.validate(
        redacted_blocks=[b],
        document_id="d_leak2",
        original_findings=[orig_finding],
        original_blocks=[b],
    )
    assert val_res.status == SecurityStatus.BLOCKED
    assert val_res.passed is False
    assert any("Alice Smith" in rf.text for rf in val_res.residual_findings)

