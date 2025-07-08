import json
import logging
from typing import Literal, Any, Optional, Iterator
import cv2
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from agentic_doc.common import ChunkType, T
import warnings

_LOGGER = structlog.get_logger(__name__)
_MAX_PARALLEL_TASKS = 200
# Colors in BGR format (OpenCV uses BGR)
_COLOR_MAP = {
    ChunkType.marginalia: (128, 0, 255),  # Purple for marginalia
    ChunkType.table: (139, 69, 19),  # Brown for tables
    ChunkType.figure: (50, 205, 50),  # Lime green for figures
    ChunkType.text: (255, 0, 0),  # Blue for regular text
}


class ParseConfig:
    """
    Configuration class for the parse function.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        include_marginalia: Optional[bool] = None,
        include_metadata_in_markdown: Optional[bool] = None,
        extraction_model: Optional[type[T]] = None,
        extraction_schema: Optional[dict[str, Any]] = None,
        split_size: Optional[int] = None,
        extraction_split_size: Optional[int] = None,
    ) -> None:
        self.api_key = api_key
        self.include_marginalia = include_marginalia
        self.include_metadata_in_markdown = include_metadata_in_markdown
        self.extraction_model = extraction_model
        self.extraction_schema = extraction_schema
        self.split_size = split_size
        self.extraction_split_size = extraction_split_size


class SettingsOverrides:
    def __init__(self) -> None:
        object.__setattr__(self, "_overrides", {})

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_overrides":
            object.__setattr__(self, name, value)
            return

        warnings.warn(
            (
                "Setting values directly on agentic_doc.config.settings will be "
                "deprecated in a future release. Please, call "
                "parse(..., config=ParseConfig(api_key='xxx')) instead."
            ),
            DeprecationWarning,
        )
        self._overrides[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._overrides.items())

    def __str__(self) -> str:
        # Create a copy of dict with redacted API key
        settings_dict = get_settings().model_dump()
        if "vision_agent_api_key" in settings_dict:
            settings_dict["vision_agent_api_key"] = (
                settings_dict["vision_agent_api_key"][:5] + "[REDACTED]"
            )
        return f"{json.dumps(settings_dict, indent=2)}"


class Settings(BaseSettings):
    endpoint_host: str = Field(
        default="https://api.va.landing.ai",
        description="The host of the endpoint to use",
    )
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
    split_size: int = Field(
        default=10,
        description="Pages per chunk for splitting the document",
        ge=1,
        le=100,
    )
    extraction_split_size: int = Field(
        default=50,
        description="Pages per chunk for splitting the document when field extraction is enabled",
        ge=1,
        le=50,
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


# Global settings instance to hold overrides
# This is a temporary solution to avoid breaking changes in the API, since there
# are users doing `agentic_doc.config.settings.vision_agent_api_key = 'xxx'` today.
# The internal code should use `get_settings()` to retrieve the default settings instance,
# and end users should call `parse(..., settings=Settings(api_key='xxx'))` to override
# the global "env-vars-based" settings.
settings = SettingsOverrides()


def get_settings() -> Settings:
    """
    Get the settings instance, applying any overrides set on the settings global object.
    """
    new_settings = Settings()
    for k, v in settings:
        setattr(new_settings, k, v)
    return new_settings


_LOGGER.info(f"Settings loaded: {settings}")

if get_settings().batch_size * get_settings().max_workers > _MAX_PARALLEL_TASKS:
    raise ValueError(
        f"Batch size * max workers must be less than {_MAX_PARALLEL_TASKS}."
        " Please reduce the batch size or max workers."
        " Current settings: batch_size={settings.batch_size}, max_workers={settings.max_workers}"
    )
if get_settings().retry_logging_style == "inline_block":
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
