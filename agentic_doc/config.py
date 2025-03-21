import json
import logging
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER = structlog.get_logger(__name__)
_MAX_PARALLEL_TASKS = 100


class Settings(BaseSettings):
    vision_agent_api_key: str = Field()
    batch_size: int = Field(default=4)
    max_workers: int = Field(default=5)
    max_retries: int = Field(default=100)
    max_retry_wait_time: int = Field(default=60)
    retry_logging_style: Literal["none", "log_msg", "inline_block"] = Field(
        default="log_msg"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    def __str__(self) -> str:
        # Create a copy of dict with redacted API key
        settings_dict = self.model_dump()
        if "vision_agent_api_key" in settings_dict:
            settings_dict["vision_agent_api_key"] = (
                settings_dict["vision_agent_api_key"][:5] + "[REDACTED]"
            )
        return f"{json.dumps(settings_dict, indent=2)}"


settings = Settings()
_LOGGER.info(f"Settings loaded: {settings}")

if settings.batch_size * settings.max_workers > _MAX_PARALLEL_TASKS:
    raise ValueError(
        f"Batch size * max workers must be less than {_MAX_PARALLEL_TASKS}."
        " Please reduce the batch size or max workers."
        " Current settings: batch_size={settings.batch_size}, max_workers={settings.max_workers}"
    )

if settings.retry_logging_style == "inline_block":
    logging.getLogger("httpx").setLevel(logging.WARNING)
