import os
import json
from unittest.mock import patch, MagicMock

import pytest
import cv2

from agentic_doc.config import (
    Settings,
    VisualizationConfig,
    _COLOR_MAP,
    _MAX_PARALLEL_TASKS,
)
from agentic_doc.common import ChunkType


def test_default_config():
    settings = Settings()
    assert settings.retry_logging_style == "log_msg"
    assert settings.batch_size > 0
    assert settings.max_workers > 0
    assert settings.max_retries > 0
    assert settings.max_retry_wait_time > 0
    assert settings.endpoint_host == "https://api.va.landing.ai"
    assert settings.pdf_to_image_dpi == 96


def test_custom_config(monkeypatch):
    # Set environment variables
    monkeypatch.setenv("BATCH_SIZE", "10")
    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("MAX_RETRIES", "50")
    monkeypatch.setenv("MAX_RETRY_WAIT_TIME", "30")
    monkeypatch.setenv("RETRY_LOGGING_STYLE", "inline_block")
    monkeypatch.setenv("ENDPOINT_HOST", "https://custom-endpoint.example.com")
    monkeypatch.setenv("PDF_TO_IMAGE_DPI", "150")

    settings = Settings()

    # Verify settings were loaded from environment variables
    assert settings.batch_size == 10
    assert settings.max_workers == 8
    assert settings.max_retries == 50
    assert settings.max_retry_wait_time == 30
    assert settings.retry_logging_style == "inline_block"
    assert settings.endpoint_host == "https://custom-endpoint.example.com"
    assert settings.pdf_to_image_dpi == 150


def test_settings_validation():
    # Test that max_retries can't be negative
    with pytest.raises(ValueError):
        Settings(max_retries=-1)

    # Test that batch_size can't be less than 1
    with pytest.raises(ValueError):
        Settings(batch_size=0)

    # Test that max_workers can't be less than 1
    with pytest.raises(ValueError):
        Settings(max_workers=0)

    # Test that max_retry_wait_time can't be negative
    with pytest.raises(ValueError):
        Settings(max_retry_wait_time=-1)

    # Test pdf_to_image_dpi can't be less than 1
    with pytest.raises(ValueError):
        Settings(pdf_to_image_dpi=0)


def test_settings_str_method():
    # Create settings with an API key
    settings = Settings(vision_agent_api_key="abcde12345")

    # Convert to string and verify API key is redacted
    settings_str = str(settings)
    assert "vision_agent_api_key" in settings_str
    assert "abcde[REDACTED]" in settings_str
    assert "12345" not in settings_str

    # Verify other settings are included
    assert "batch_size" in settings_str
    assert "max_workers" in settings_str
    assert "max_retries" in settings_str
    assert "max_retry_wait_time" in settings_str
    assert "retry_logging_style" in settings_str

def test_visualization_config_defaults():
    # Test default visualization config
    viz_config = VisualizationConfig()

    # Check defaults
    assert viz_config.thickness == 1
    assert viz_config.text_bg_opacity == 0.7
    assert viz_config.padding == 1
    assert viz_config.font_scale == 0.5
    assert viz_config.font == cv2.FONT_HERSHEY_SIMPLEX

    # Check that the color map contains all relevant chunk types
    expected_chunk_types = set(ChunkType)
    for chunk_type in expected_chunk_types:
        assert chunk_type in viz_config.color_map, f"Missing chunk type: {chunk_type}"


def test_visualization_config_custom():
    # Test custom visualization config
    custom_viz_config = VisualizationConfig(
        thickness=2,
        text_bg_opacity=0.5,
        padding=3,
        font_scale=0.8,
        font=cv2.FONT_HERSHEY_PLAIN,
        color_map={ChunkType.text: (255, 0, 0), ChunkType.table: (0, 255, 0)},
    )

    # Check custom values
    assert custom_viz_config.thickness == 2
    assert custom_viz_config.text_bg_opacity == 0.5
    assert custom_viz_config.padding == 3
    assert custom_viz_config.font_scale == 0.8
    assert custom_viz_config.font == cv2.FONT_HERSHEY_PLAIN

    # Check that the custom color map contains only the specified chunk types
    assert custom_viz_config.color_map[ChunkType.text] == (255, 0, 0)
    assert custom_viz_config.color_map[ChunkType.table] == (0, 255, 0)


def test_visualization_config_validation():
    # Test that text_bg_opacity must be between 0 and 1
    with pytest.raises(ValueError):
        VisualizationConfig(text_bg_opacity=-0.1)

    with pytest.raises(ValueError):
        VisualizationConfig(text_bg_opacity=1.1)

    # Test that thickness can't be negative
    with pytest.raises(ValueError):
        VisualizationConfig(thickness=-1)

    # Test that padding can't be negative
    with pytest.raises(ValueError):
        VisualizationConfig(padding=-1)

    # Test that font_scale can't be negative
    with pytest.raises(ValueError):
        VisualizationConfig(font_scale=-0.1)
