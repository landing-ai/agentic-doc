import json
from typing import Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER = structlog.get_logger(__name__)


class Settings(BaseSettings):
    vision_agent_api_key: str = Field()
    batch_size: int = Field(default=4)
    max_workers: int = Field(default=5)
    max_retries: int = Field(default=100)
    max_retry_wait_time: int = Field(default=60)
    retry_logging_style: Literal["none", "log_msg", "inline_blobk"] = Field(
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
