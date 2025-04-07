import json
import logging
from typing import Literal

import cv2
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc.common import ChunkType

_LOGGER = structlog.get_logger(__name__)
_MAX_PARALLEL_TASKS = 100
# Colors in BGR format (OpenCV uses BGR)
_COLOR_MAP = {
    ChunkType.title: (0, 0, 125),  # Dark red for titles
    ChunkType.page_header: (0, 200, 200),  # Yellow-ish for headers
    ChunkType.page_footer: (200, 200, 0),  # Cyan-ish for footers
    ChunkType.page_number: (128, 128, 128),  # Gray for page numbers
    ChunkType.key_value: (255, 0, 255),  # Magenta for key-value pairs
    ChunkType.form: (128, 0, 255),  # Purple for forms
    ChunkType.table: (139, 69, 19),  # Brown for tables
    ChunkType.figure: (50, 205, 50),  # Lime green for figures
    ChunkType.text: (255, 0, 0),  # Blue for regular text
}


class Settings(BaseSettings):
    vision_agent_api_key: str = Field(
        description="API key for the vision agent",
        default="",
    )
    batch_size: int = Field(
        default=4,
        description="Number of documents to process in parallel",
        ge=1,
    )
    max_workers: int = Field(
        default=5,
        description="Maximum number of workers to use for parallel processing for each document",
        ge=1,
    )
    max_retries: int = Field(
        default=100,
        description="Maximum number of retries for a failed request",
        ge=0,
    )
    max_retry_wait_time: int = Field(
        default=60,
        description="Maximum wait time for a retry",
        ge=0,
    )
    retry_logging_style: Literal["none", "log_msg", "inline_block"] = Field(
        default="log_msg",
        description="Logging style for retries",
    )
    pdf_to_image_dpi: int = Field(
        default=96,
        description="DPI for converting PDF pages to images",
        ge=1,
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


class VisualizationConfig(BaseSettings):
    thickness: int = Field(
        default=1,
        description="Thickness of the bounding box and text",
        ge=0,
    )
    text_bg_color: tuple[int, int, int] = Field(
        default=(211, 211, 211),  # Light gray
        description="Background color of the text, in BGR format",
    )
    text_bg_opacity: float = Field(
        default=0.7,
        description="Opacity of the text background",
        ge=0.0,
        le=1.0,
    )
    padding: int = Field(
        default=1,
        description="Padding of the text background box",
        ge=0,
    )
    font_scale: float = Field(
        default=0.5,
        description="Font scale of the text",
        ge=0.0,
    )
    font: int = Field(
        default=cv2.FONT_HERSHEY_SIMPLEX,
        description="Font of the text",
    )
    color_map: dict[ChunkType, tuple[int, int, int]] = Field(
        default=_COLOR_MAP,
        description="Color map for each chunk type",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
