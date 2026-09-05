"""I test that application diagnostics can be persisted to a chosen log file."""

from __future__ import annotations

from kneescope12.utils.logging import configure_logging


def test_configure_logging_writes_file(tmp_path):
    """I write a named message to a caller-provided log destination."""

    log_file = tmp_path / "logs" / "audit.log"
    logger = configure_logging("INFO", log_file=log_file)
    logger.info("I recorded a test diagnostic.")
    for handler in logger.handlers:
        handler.flush()
    assert "I recorded a test diagnostic." in log_file.read_text(encoding="utf-8")
