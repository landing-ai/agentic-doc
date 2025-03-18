from agentic_doc.config import Settings


def test_default_config():
    settings = Settings()
    assert settings.retry_logging_style == "log_msg"
    assert settings.batch_size > 0
    assert settings.max_workers > 0
    assert settings.max_retries > 0
    assert settings.max_retry_wait_time > 0
