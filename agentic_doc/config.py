import json
import logging
from typing import Literal, Any, Optional
import cv2
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from agentic_doc.common import ChunkType, T
import os
from dotenv import load_dotenv

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
        include_marginalia: Optional[bool] = None,
        include_metadata_in_markdown: Optional[bool] = None,
        extraction_model: Optional[T] = None,
        extraction_schema: Optional[dict[str, Any]] = None,
        split_size: Optional[int] = None,
        extraction_split_size: Optional[int] = None,
    ):
        self.include_marginalia = include_marginalia
        self.include_metadata_in_markdown = include_metadata_in_markdown
        self.extraction_model = extraction_model
        self.extraction_schema = extraction_schema
        self.split_size = split_size
        self.extraction_split_size = extraction_split_size


class Settings:
    """
    Settings class that automatically pulls from environment variables with va_ prefix
    or returns user-set values. Falls back to defaults if user has not set and env variable not set.
    """

    def __init__(self):
        self._user_values = {}

        self._defaults = {
            "endpoint_host": "https://api.va.landing.ai",
            "vision_agent_api_key": "",
            "batch_size": 4,
            "max_workers": 5,
            "max_retries": 100,
            "max_retry_wait_time": 60,
            "retry_logging_style": "log_msg",
            "pdf_to_image_dpi": 96,
            "split_size": 10,
            "extraction_split_size": 50,
        }

        self._env_mappings = {
            "vision_agent_api_key": "VISION_AGENT_API_KEY",
            "endpoint_host": "VA_ENDPOINT_HOST",
            "batch_size": "VA_BATCH_SIZE",
            "max_workers": "VA_MAX_WORKERS",
            "max_retries": "VA_MAX_RETRIES",
            "max_retry_wait_time": "VA_MAX_RETRY_WAIT_TIME",
            "retry_logging_style": "VA_RETRY_LOGGING_STYLE",
            "pdf_to_image_dpi": "VA_PDF_TO_IMAGE_DPI",
            "split_size": "VA_SPLIT_SIZE",
            "extraction_split_size": "VA_EXTRACTION_SPLIT_SIZE",
        }

        self._validation_rules = {
            "batch_size": {"type": int, "min": 1},
            "max_workers": {"type": int, "min": 1},
            "max_retries": {"type": int, "min": 0},
            "max_retry_wait_time": {"type": int, "min": 0},
            "pdf_to_image_dpi": {"type": int, "min": 1},
            "split_size": {"type": int, "min": 1, "max": 100},
            "extraction_split_size": {"type": int, "min": 1, "max": 50},
            "retry_logging_style": {
                "type": str,
                "choices": ["none", "log_msg", "inline_block"],
            },
        }

        load_dotenv()

    def _validate_value(self, key: str, value: Any) -> Any:
        """
        Validate the given value against the rules defined for the key,
        which are defined in self._validation_rules. This may need to be extended in the future.
        """
        if key not in self._validation_rules:
            return value

        rules = self._validation_rules[key]

        if rules["type"] == int:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {key}: must be an integer")
        elif rules["type"] == str:
            value = str(value)

        if "min" in rules and value < rules["min"]:
            raise ValueError(f"Invalid value for {key}: must be >= {rules['min']}")
        if "max" in rules and value > rules["max"]:
            raise ValueError(f"Invalid value for {key}: must be <= {rules['max']}")

        if "choices" in rules and value not in rules["choices"]:
            raise ValueError(
                f"Invalid value for {key}: must be one of {rules['choices']}"
            )

        return value

    def _get_value(self, key: str) -> Any:
        """
        Get value of a key with precedence: user-set > environment > default.
        """
        # First check if the user explicitly set a value (settings.key = value)
        if key in self._user_values:
            return self._user_values[key]

        # Then check environment variable
        env_var = self._env_mappings.get(key)
        if env_var and env_var in os.environ:
            env_value = os.environ.get(env_var, None)
            if env_value is not None:
                return self._validate_value(key, env_value)

        # Finally use default
        return self._defaults.get(key)

    def _set_value(self, key: str, value: Any) -> None:
        validated_value = self._validate_value(key, value)
        self._user_values[key] = validated_value
        self._run_validation_checks()

    def _run_validation_checks(self) -> None:
        if self.va_batch_size * self.va_max_workers > _MAX_PARALLEL_TASKS:
            raise ValueError(
                f"Batch size * max workers must be less than {_MAX_PARALLEL_TASKS}."
                " Please reduce the batch size or max workers."
                f" Current settings: batch_size={self.va_batch_size}, max_workers={self.va_max_workers}"
            )

        if self.va_retry_logging_style == "inline_block":
            logging.getLogger("httpx").setLevel(logging.WARNING)

    # properties for each setting
    @property
    def vision_agent_api_key(self) -> str:
        return self._get_value("vision_agent_api_key")

    @vision_agent_api_key.setter
    def vision_agent_api_key(self, value: str) -> None:
        self._set_value("vision_agent_api_key", value)

    @property
    def va_endpoint_host(self) -> str:
        return self._get_value("endpoint_host")

    @va_endpoint_host.setter
    def va_endpoint_host(self, value: str) -> None:
        self._set_value("endpoint_host", value)

    @property
    def va_batch_size(self) -> int:
        return self._get_value("batch_size")

    @va_batch_size.setter
    def va_batch_size(self, value: int) -> None:
        self._set_value("batch_size", value)

    @property
    def va_max_workers(self) -> int:
        return self._get_value("max_workers")

    @va_max_workers.setter
    def va_max_workers(self, value: int) -> None:
        self._set_value("max_workers", value)

    @property
    def va_max_retries(self) -> int:
        return self._get_value("max_retries")

    @va_max_retries.setter
    def va_max_retries(self, value: int) -> None:
        self._set_value("max_retries", value)

    @property
    def va_max_retry_wait_time(self) -> int:
        return self._get_value("max_retry_wait_time")

    @va_max_retry_wait_time.setter
    def va_max_retry_wait_time(self, value: int) -> None:
        self._set_value("max_retry_wait_time", value)

    @property
    def va_retry_logging_style(self) -> Literal["none", "log_msg", "inline_block"]:
        return self._get_value("retry_logging_style")

    @va_retry_logging_style.setter
    def va_retry_logging_style(
        self, value: Literal["none", "log_msg", "inline_block"]
    ) -> None:
        self._set_value("retry_logging_style", value)

    @property
    def va_pdf_to_image_dpi(self) -> int:
        return self._get_value("pdf_to_image_dpi")

    @va_pdf_to_image_dpi.setter
    def va_pdf_to_image_dpi(self, value: int) -> None:
        self._set_value("pdf_to_image_dpi", value)

    @property
    def va_split_size(self) -> int:
        return self._get_value("split_size")

    @va_split_size.setter
    def va_split_size(self, value: int) -> None:
        self._set_value("split_size", value)

    @property
    def va_extraction_split_size(self) -> int:
        return self._get_value("extraction_split_size")

    @va_extraction_split_size.setter
    def va_extraction_split_size(self, value: int) -> None:
        self._set_value("extraction_split_size", value)

    # Legacy property aliases for backward compatibility
    @property
    def endpoint_host(self) -> str:
        return self.va_endpoint_host

    @endpoint_host.setter
    def endpoint_host(self, value: str) -> None:
        self.va_endpoint_host = value

    @property
    def max_retries(self) -> str:
        return self.va_max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self.va_max_retries = value

    @property
    def max_retry_wait_time(self) -> int:
        return self.va_max_retry_wait_time

    @max_retry_wait_time.setter
    def max_retry_wait_time(self, value: int) -> None:
        self.va_max_retry_wait_time = value

    @property
    def pdf_to_image_dpi(self) -> int:
        return self.va_pdf_to_image_dpi

    @pdf_to_image_dpi.setter
    def pdf_to_image_dpi(self, value: int) -> None:
        self.va_pdf_to_image_dpi = value

    @property
    def split_size(self) -> int:
        return self.va_split_size

    @split_size.setter
    def split_size(self, value: int) -> None:
        self.va_split_size = value

    @property
    def extraction_split_size(self) -> int:
        return self.va_extraction_split_size

    @extraction_split_size.setter
    def extraction_split_size(self, value: int) -> None:
        self.va_extraction_split_size = value

    @property
    def batch_size(self) -> int:
        return self.va_batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.va_batch_size = value

    @property
    def max_workers(self) -> int:
        return self.va_max_workers

    @max_workers.setter
    def max_workers(self, value: int) -> None:
        self.va_max_workers = value

    @property
    def retry_logging_style(self) -> Literal["none", "log_msg", "inline_block"]:
        return self.va_retry_logging_style

    @retry_logging_style.setter
    def retry_logging_style(
        self, value: Literal["none", "log_msg", "inline_block"]
    ) -> None:
        self.va_retry_logging_style = value

    def __str__(self) -> str:
        settings_dict = {}

        for key in self._defaults.keys():
            # Use the original key for vision_agent_api_key, otherwise prefix with va_
            if key.startswith("vision_agent_api_key"):
                prop_name = key
            else:
                prop_name = f"va_{key}"

            value = self._get_value(key)

            # Redact API key
            if key == "vision_agent_api_key" and value:
                value = value[:5] + "[REDACTED]" if len(value) > 5 else "[REDACTED]"

            settings_dict[prop_name] = value

        return f"{json.dumps(settings_dict, indent=2)}"


settings = Settings()


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
