import json
import os
from unittest.mock import MagicMock, patch

import cv2
import pytest

from agentic_doc.common import ChunkType
from agentic_doc.config import (
    _COLOR_MAP,
    _MAX_PARALLEL_TASKS,
    get_settings,
    Settings,
    VisualizationConfig,
)


def test_default_config():
    settings = get_settings()
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

    settings = get_settings()

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


# ParseConfig Tests
def test_parse_config_default_instantiation():
    from agentic_doc.config import ParseConfig
    
    config = ParseConfig()
    
    assert config.api_key is None
    assert config.include_marginalia is None
    assert config.include_metadata_in_markdown is None
    assert config.extraction_model is None
    assert config.extraction_schema is None
    assert config.split_size is None
    assert config.extraction_split_size is None


def test_parse_config_custom_instantiation():
    from agentic_doc.config import ParseConfig
    from pydantic import BaseModel
    
    class TestModel(BaseModel):
        field1: str
        field2: int
    
    test_schema = {"type": "object", "properties": {"test": {"type": "string"}}}
    
    config = ParseConfig(
        api_key="test_key_123",
        include_marginalia=False,
        include_metadata_in_markdown=True,
        extraction_model=TestModel,
        extraction_schema=test_schema,
        split_size=5,
        extraction_split_size=25
    )
    
    assert config.api_key == "test_key_123"
    assert config.include_marginalia is False
    assert config.include_metadata_in_markdown is True
    assert config.extraction_model == TestModel
    assert config.extraction_schema == test_schema
    assert config.split_size == 5
    assert config.extraction_split_size == 25


def test_parse_config_partial_instantiation():
    from agentic_doc.config import ParseConfig
    
    config = ParseConfig(
        api_key="partial_key",
        include_marginalia=True,
        split_size=15
    )
    
    assert config.api_key == "partial_key"
    assert config.include_marginalia is True
    assert config.include_metadata_in_markdown is None
    assert config.extraction_model is None
    assert config.extraction_schema is None
    assert config.split_size == 15
    assert config.extraction_split_size is None


def test_parse_config_settings_integration():
    from agentic_doc.config import ParseConfig, Settings
    
    # Test that ParseConfig can hold settings-related values
    config = ParseConfig(
        api_key="config_api_key",
        split_size=20,
        extraction_split_size=30
    )
    
    # Test that settings still work independently
    custom_settings = Settings(
        vision_agent_api_key="settings_api_key",
        split_size=25,
        extraction_split_size=35
    )
    
    # Verify values are independent
    assert config.api_key == "config_api_key"
    assert config.split_size == 20
    assert config.extraction_split_size == 30
    assert custom_settings.vision_agent_api_key == "settings_api_key"
    assert custom_settings.split_size == 25
    assert custom_settings.extraction_split_size == 35


def test_parse_config_precedence_logic():
    from agentic_doc.config import ParseConfig
    
    # Test the logic used in parse function for precedence
    # config values should take precedence over settings when not None
    config = ParseConfig(
        include_marginalia=False,
        include_metadata_in_markdown=True,
        split_size=12,
        extraction_split_size=18
    )
    
    # Simulate the precedence logic from parse function
    include_marginalia = config.include_marginalia if config.include_marginalia is not None else True
    include_metadata_in_markdown = config.include_metadata_in_markdown if config.include_metadata_in_markdown is not None else True
    split_size = config.split_size if config.split_size is not None else 10
    extraction_split_size = config.extraction_split_size if config.extraction_split_size is not None else 50
    
    assert include_marginalia is False  # From config
    assert include_metadata_in_markdown is True  # From config
    assert split_size == 12  # From config
    assert extraction_split_size == 18  # From config
    
    # Test with None values - should use defaults
    config_none = ParseConfig()
    
    include_marginalia_none = config_none.include_marginalia if config_none.include_marginalia is not None else True
    include_metadata_in_markdown_none = config_none.include_metadata_in_markdown if config_none.include_metadata_in_markdown is not None else True
    split_size_none = config_none.split_size if config_none.split_size is not None else 10
    extraction_split_size_none = config_none.extraction_split_size if config_none.extraction_split_size is not None else 50
    
    assert include_marginalia_none is True  # Default
    assert include_metadata_in_markdown_none is True  # Default
    assert split_size_none == 10  # Default
    assert extraction_split_size_none == 50  # Default


def test_parse_config_env_var_interaction(monkeypatch):
    from agentic_doc.config import ParseConfig, get_settings
    
    # Set environment variables for settings
    monkeypatch.setenv("VISION_AGENT_API_KEY", "env_api_key")
    monkeypatch.setenv("SPLIT_SIZE", "8")
    monkeypatch.setenv("EXTRACTION_SPLIT_SIZE", "16")
    
    # Create a config that partially overrides env vars
    config = ParseConfig(
        api_key="config_api_key",  # Override env var
        split_size=22,  # Override env var
        # extraction_split_size left as None - should use env var
    )
    
    # Get settings to verify env vars are loaded
    settings = get_settings()
    assert settings.vision_agent_api_key == "env_api_key"
    assert settings.split_size == 8
    assert settings.extraction_split_size == 16
    
    # Test precedence: config overrides env vars, but None values fall back to settings
    assert config.api_key == "config_api_key"  # Config overrides env
    assert config.split_size == 22  # Config overrides env
    assert config.extraction_split_size is None  # Will use settings value (16)
    
    # Simulate the logic from parse function
    final_split_size = config.split_size if config.split_size is not None else settings.split_size
    final_extraction_split_size = config.extraction_split_size if config.extraction_split_size is not None else settings.extraction_split_size
    
    assert final_split_size == 22  # From config
    assert final_extraction_split_size == 16  # From env var via settings


def test_parse_config_behavior_with_defaults():
    from agentic_doc.config import ParseConfig, Settings
    
    # Create config with some values
    config = ParseConfig(
        api_key="config_only_key",
        split_size=33
    )
    
    # Create fresh settings instance - this will load from .env or defaults
    settings = Settings()
    
    # Test that config values are preserved and None values use defaults
    assert config.api_key == "config_only_key"
    assert config.split_size == 33
    assert config.extraction_split_size is None
    
    # Test the core behavior: config values override, None values fall back to settings
    final_api_key = config.api_key if config.api_key is not None else settings.vision_agent_api_key
    final_split_size = config.split_size if config.split_size is not None else settings.split_size
    final_extraction_split_size = config.extraction_split_size if config.extraction_split_size is not None else settings.extraction_split_size
    
    assert final_api_key == "config_only_key"  # From config
    assert final_split_size == 33  # From config
    assert final_extraction_split_size == settings.extraction_split_size  # From settings (could be env or default)


def test_parse_config_mixed_env_and_config_scenarios(monkeypatch):
    from agentic_doc.config import ParseConfig, get_settings
    
    # Set some env vars but not others
    monkeypatch.setenv("VISION_AGENT_API_KEY", "env_mixed_key")
    monkeypatch.setenv("SPLIT_SIZE", "7")
    # EXTRACTION_SPLIT_SIZE not set - should use default
    
    # Test various config scenarios
    
    # Scenario 1: Config overrides everything
    config1 = ParseConfig(
        api_key="override_key",
        split_size=11,
        extraction_split_size=21
    )
    
    settings = get_settings()
    assert settings.vision_agent_api_key == "env_mixed_key"
    assert settings.split_size == 7
    assert settings.extraction_split_size == 50  # Default
    
    # Config values should be independent
    assert config1.api_key == "override_key"
    assert config1.split_size == 11
    assert config1.extraction_split_size == 21
    
    # Scenario 2: Config partially overrides
    config2 = ParseConfig(
        split_size=99
        # api_key and extraction_split_size are None
    )
    
    assert config2.api_key is None
    assert config2.split_size == 99
    assert config2.extraction_split_size is None
    
    # Final values would be: config if not None, else settings
    final_api_key = config2.api_key if config2.api_key is not None else settings.vision_agent_api_key
    final_split_size = config2.split_size if config2.split_size is not None else settings.split_size
    final_extraction_split_size = config2.extraction_split_size if config2.extraction_split_size is not None else settings.extraction_split_size
    
    assert final_api_key == "env_mixed_key"  # From env var
    assert final_split_size == 99  # From config
    assert final_extraction_split_size == 50  # From settings default


def test_parse_config_api_key_precedence(monkeypatch):
    from agentic_doc.config import ParseConfig, get_settings, settings
    
    # Set environment API key
    monkeypatch.setenv("VISION_AGENT_API_KEY", "env_api_key_test")
    
    # Test 1: Config API key overrides environment
    config_with_key = ParseConfig(api_key="config_api_key_test")
    settings_instance = get_settings()
    
    assert settings_instance.vision_agent_api_key == "env_api_key_test"
    assert config_with_key.api_key == "config_api_key_test"
    
    # Config should take precedence
    final_api_key = config_with_key.api_key if config_with_key.api_key is not None else settings_instance.vision_agent_api_key
    assert final_api_key == "config_api_key_test"
    
    # Test 2: No config API key - should use environment
    config_no_key = ParseConfig()
    assert config_no_key.api_key is None
    
    final_api_key_no_config = config_no_key.api_key if config_no_key.api_key is not None else settings_instance.vision_agent_api_key
    assert final_api_key_no_config == "env_api_key_test"


def test_parse_config_api_key_with_settings_override(monkeypatch):
    from agentic_doc.config import ParseConfig, get_settings, settings
    
    # Clear environment API key
    monkeypatch.delenv("VISION_AGENT_API_KEY", raising=False)
    
    # Set API key through settings global object (legacy method)
    settings.vision_agent_api_key = "settings_override_key"
    
    # Test that config still takes precedence over settings override
    config_with_key = ParseConfig(api_key="config_precedence_key")
    settings_instance = get_settings()
    
    assert settings_instance.vision_agent_api_key == "settings_override_key"
    assert config_with_key.api_key == "config_precedence_key"
    
    # Config should take precedence
    final_api_key = config_with_key.api_key if config_with_key.api_key is not None else settings_instance.vision_agent_api_key
    assert final_api_key == "config_precedence_key"
    
    # Test with no config API key - should use settings override
    config_no_key = ParseConfig()
    assert config_no_key.api_key is None
    
    final_api_key_no_config = config_no_key.api_key if config_no_key.api_key is not None else settings_instance.vision_agent_api_key
    assert final_api_key_no_config == "settings_override_key"


def test_parse_config_api_key_empty_string():
    from agentic_doc.config import ParseConfig
    
    # Test with empty string API key
    config_empty = ParseConfig(api_key="")
    assert config_empty.api_key == ""
    
    # Test with None API key
    config_none = ParseConfig(api_key=None)
    assert config_none.api_key is None
    
    # Test with whitespace API key
    config_whitespace = ParseConfig(api_key="   ")
    assert config_whitespace.api_key == "   "


def test_parse_config_api_key_type_validation():
    from agentic_doc.config import ParseConfig
    
    # Test that non-string API keys are handled appropriately
    # (ParseConfig doesn't do type validation, but we test the behavior)
    
    # Valid string
    config_valid = ParseConfig(api_key="valid_key_123")
    assert config_valid.api_key == "valid_key_123"
    
    # None is valid
    config_none = ParseConfig(api_key=None)
    assert config_none.api_key is None
    
    # Empty string is valid
    config_empty = ParseConfig(api_key="")
    assert config_empty.api_key == ""
