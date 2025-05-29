import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from agentic_doc.connectors import (
    GoogleDriveConnector,
    GoogleDriveConnectorConfig,
    LocalConnector,
    LocalConnectorConfig,
    S3Connector,
    S3ConnectorConfig,
    URLConnector,
    URLConnectorConfig,
    create_connector,
)


class TestLocalConnector:
    """Test LocalConnector functionality."""

    def test_list_files_in_directory(self, temp_dir):
        """Test listing files in a directory."""
        # Create test files
        (temp_dir / "test1.pdf").touch()
        (temp_dir / "test2.png").touch()
        (temp_dir / "test3.txt").touch()

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        files = connector.list_files(str(temp_dir))

        # Should only return supported file types
        assert len(files) == 2
        assert str(temp_dir / "test1.pdf") in files
        assert str(temp_dir / "test2.png") in files
        assert str(temp_dir / "test3.txt") not in files

    def test_list_files_with_pattern(self, temp_dir):
        """Test listing files with a pattern."""
        (temp_dir / "doc1.pdf").touch()
        (temp_dir / "doc2.pdf").touch()
        (temp_dir / "image.png").touch()

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        files = connector.list_files(str(temp_dir), "*.pdf")

        assert len(files) == 2
        assert all(f.endswith(".pdf") for f in files)

    def test_list_files_single_file(self, temp_dir):
        """Test listing a single file."""
        test_file = temp_dir / "test.pdf"
        test_file.touch()

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        files = connector.list_files(str(test_file))

        assert len(files) == 1
        assert files[0] == str(test_file)

    def test_list_files_nonexistent_path(self):
        """Test listing files from non-existent path."""
        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        with pytest.raises(FileNotFoundError):
            connector.list_files("/nonexistent/path")

    def test_download_file(self, temp_dir):
        """Test downloading (returning) a local file."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        result_path = connector.download_file(str(test_file))

        assert result_path == test_file
        assert result_path.exists()

    def test_download_nonexistent_file(self):
        """Test downloading non-existent file."""
        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        with pytest.raises(FileNotFoundError):
            connector.download_file("/nonexistent/file.pdf")

    def test_get_file_info(self, temp_dir):
        """Test getting file metadata."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("test content")

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        info = connector.get_file_info(str(test_file))

        assert info["name"] == "test.pdf"
        assert info["path"] == str(test_file)
        assert info["size"] == len("test content")
        assert info["suffix"] == ".pdf"
        assert "modified" in info


class TestGoogleDriveConnector:
    """Test GoogleDriveConnector functionality."""

class TestS3Connector:
    """Test S3Connector functionality."""

    def test_init_with_credentials(self):
        """Test initialization with AWS credentials."""
        config = S3ConnectorConfig(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-west-2",
        )
        connector = S3Connector(config)

        assert connector.config.bucket_name == "test-bucket"
        assert connector.config.aws_access_key_id == "test-key"
        assert connector.config.region_name == "us-west-2"

class TestURLConnector:
    """Test URLConnector functionality."""

    def test_init_with_headers(self):
        """Test initialization with custom headers."""
        config = URLConnectorConfig(
            headers={"Authorization": "Bearer token"}, timeout=60
        )
        connector = URLConnector(config)

        assert connector.config.headers == {"Authorization": "Bearer token"}
        assert connector.config.timeout == 60

    def test_list_files(self):
        """Test listing files (should return the URL)."""
        config = URLConnectorConfig()
        connector = URLConnector(config)

        files = connector.list_files("https://example.com/document.pdf")

        assert len(files) == 1
        assert files[0] == "https://example.com/document.pdf"

class TestConnectorFactory:
    """Test the connector factory function."""

    def test_create_local_connector(self):
        """Test creating a local connector."""
        config = LocalConnectorConfig()
        connector = create_connector(config)

        assert isinstance(connector, LocalConnector)

    def test_create_google_drive_connector(self):
        """Test creating a Google Drive connector."""
        config = GoogleDriveConnectorConfig(
            client_secret_file="test"
        )
        connector = create_connector(config)

        assert isinstance(connector, GoogleDriveConnector)

    def test_create_s3_connector(self):
        """Test creating an S3 connector."""
        config = S3ConnectorConfig(bucket_name="test-bucket")
        connector = create_connector(config)

        assert isinstance(connector, S3Connector)

    def test_create_url_connector(self):
        """Test creating a URL connector."""
        config = URLConnectorConfig()
        connector = create_connector(config)

        assert isinstance(connector, URLConnector)

    def test_create_unknown_connector(self):
        """Test creating an unknown connector type."""
        config = LocalConnectorConfig()
        config.connector_type = "unknown"

        with pytest.raises(ValueError, match="Unknown connector type"):
            create_connector(config)


class TestConnectorConfigs:
    """Test connector configuration models."""

    def test_local_connector_config_defaults(self):
        """Test LocalConnectorConfig defaults."""
        config = LocalConnectorConfig()
        assert config.connector_type == "local"

    def test_google_drive_connector_config_defaults(self):
        """Test GoogleDriveConnectorConfig defaults."""
        config = GoogleDriveConnectorConfig()
        assert config.connector_type == "google_drive"
        assert config.folder_id is None

    def test_s3_connector_config_defaults(self):
        """Test S3ConnectorConfig defaults."""
        config = S3ConnectorConfig(bucket_name="test-bucket")
        assert config.connector_type == "s3"
        assert config.region_name == "us-east-1"
        assert config.bucket_name == "test-bucket"

    def test_url_connector_config_defaults(self):
        """Test URLConnectorConfig defaults."""
        config = URLConnectorConfig()
        assert config.connector_type == "url"
        assert config.headers is None
        assert config.timeout == 30
