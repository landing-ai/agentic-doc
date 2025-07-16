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

    def test_list_files_recursive(self, temp_dir):
        """Test listing files with a pattern."""
        (temp_dir / "doc1.pdf").touch()
        (temp_dir / "doc2.pdf").touch()
        (temp_dir / "image1.png").touch()
        (temp_dir / "subdir1").mkdir()
        (temp_dir / "subdir1" / "doc3.pdf").touch()
        (temp_dir / "subdir1" / "doc4.pdf").touch()
        (temp_dir / "subdir1" / "image2.png").touch()
        (temp_dir / "subdir1" / "subdir2").mkdir()
        (temp_dir / "subdir1" / "subdir2" / "image3.png").touch()
        (temp_dir / "subdir1" / "subdir2" / "doc5.pdf").touch()

        config = LocalConnectorConfig(recursive=True)
        connector = LocalConnector(config)

        files = connector.list_files(str(temp_dir))

        assert len(files) == 8

    def test_list_files_recursive_with_pattern(self, temp_dir):
        """Test listing files with a pattern."""
        (temp_dir / "doc1.pdf").touch()
        (temp_dir / "doc2.pdf").touch()
        (temp_dir / "image1.png").touch()
        (temp_dir / "subdir1").mkdir()
        (temp_dir / "subdir1" / "doc3.pdf").touch()
        (temp_dir / "subdir1" / "doc4.pdf").touch()
        (temp_dir / "subdir1" / "image2.png").touch()
        (temp_dir / "subdir1" / "subdir2").mkdir()
        (temp_dir / "subdir1" / "subdir2" / "image3.png").touch()
        (temp_dir / "subdir1" / "subdir2" / "doc5.pdf").touch()

        config = LocalConnectorConfig(recursive=True)
        connector = LocalConnector(config)

        files = connector.list_files(str(temp_dir), "*.pdf")

        assert len(files) == 5
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

    @pytest.fixture
    def config(self):
        """Basic configuration fixture."""
        return GoogleDriveConnectorConfig(client_secret_file="test.json")

    @pytest.fixture
    def config_with_folder(self):
        """Configuration with folder ID."""
        return GoogleDriveConnectorConfig(
            client_secret_file="test.json", 
            folder_id="test_folder_id"
        )

    @pytest.fixture
    def mock_credentials(self):
        """Mock credentials object."""
        credentials = MagicMock()
        credentials.valid = False
        credentials.expired = True
        credentials.refresh_token = True
        return credentials

    @pytest.fixture
    def mock_valid_credentials(self):
        """Mock valid credentials object."""
        credentials = MagicMock()
        credentials.valid = True
        return credentials

    @pytest.fixture
    def mock_service(self):
        """Mock Google Drive service."""
        return MagicMock()

    @pytest.fixture
    def mock_files_response(self):
        """Mock file listing response."""
        return {
            "files": [
                {"id": "file1", "name": "document1.pdf", "mimeType": "application/pdf", "size": "1024"},
                {"id": "file2", "name": "image1.png", "mimeType": "image/png", "size": "2048"},
                {"id": "file3", "name": "document2.pdf", "mimeType": "application/pdf", "size": "3072"},
            ]
        }

    @pytest.fixture
    def mock_file_metadata(self):
        """Mock file metadata response."""
        return {
            "id": "file_id_123",
            "name": "test_document.pdf",
            "mimeType": "application/pdf",
            "size": "1024",
            "createdTime": "2023-01-01T00:00:00Z",
            "modifiedTime": "2023-01-02T00:00:00Z"
        }

    @pytest.fixture
    def mock_auth_patches(self):
        """Common authentication patches."""
        with patch('agentic_doc.connectors.build') as mock_build, \
             patch('agentic_doc.connectors.InstalledAppFlow') as mock_flow, \
             patch('agentic_doc.connectors.Credentials') as mock_creds, \
             patch('agentic_doc.connectors.Request') as mock_request, \
             patch('builtins.open', new_callable=mock_open), \
             patch('os.path.exists') as mock_exists:
            
            yield {
                'build': mock_build,
                'flow': mock_flow,
                'credentials': mock_creds,
                'request': mock_request,
                'exists': mock_exists
            }

    def setup_auth_mocks(self, patches, credentials, service, token_exists=False, needs_refresh=False):
        """Helper to set up authentication mocks."""
        patches['exists'].return_value = token_exists
        patches['credentials'].from_authorized_user_file.return_value = credentials
        patches['build'].return_value = service
        
        if not token_exists or (not credentials.valid and not needs_refresh):
            # Setup flow for new authentication
            mock_flow_instance = MagicMock()
            mock_flow_instance.run_local_server.return_value = credentials
            patches['flow'].from_client_secrets_file.return_value = mock_flow_instance

    def test_init_with_config(self, config):
        """Test initialization with config."""
        connector = GoogleDriveConnector(config)
        assert connector.config.client_secret_file == "test.json"

    def test_init_with_folder_id(self, config_with_folder):
        """Test initialization with folder ID."""
        connector = GoogleDriveConnector(config_with_folder)
        assert connector.config.folder_id == "test_folder_id"

    def test_get_service_new_credentials(self, config, mock_credentials, mock_service, mock_auth_patches):
        """Test service initialization with new credentials."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        connector = GoogleDriveConnector(config)
        service = connector._get_service()
        
        assert service == mock_service
        mock_auth_patches['build'].assert_called_once_with("drive", "v3", credentials=mock_credentials)

    def test_get_service_existing_credentials(self, config, mock_valid_credentials, mock_service, mock_auth_patches):
        """Test service initialization with existing valid credentials."""
        self.setup_auth_mocks(mock_auth_patches, mock_valid_credentials, mock_service, token_exists=True)
        
        connector = GoogleDriveConnector(config)
        service = connector._get_service()
        
        assert service == mock_service
        mock_auth_patches['build'].assert_called_once_with("drive", "v3", credentials=mock_valid_credentials)

    def test_get_service_refresh_credentials(self, config, mock_credentials, mock_service, mock_auth_patches):
        """Test service initialization with expired credentials that need refresh."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=True, needs_refresh=True)
        
        connector = GoogleDriveConnector(config)
        service = connector._get_service()
        
        assert service == mock_service
        mock_credentials.refresh.assert_called_once_with(mock_auth_patches['request'].return_value)

    def test_get_service_missing_client_secret(self):
        """Test service initialization without client secret file."""
        config = GoogleDriveConnectorConfig()  # No client_secret_file
        
        with patch('os.path.exists', return_value=False):
            connector = GoogleDriveConnector(config)
            
            with pytest.raises(ValueError, match="client_secret_file must be provided"):
                connector._get_service()

    def setup_file_listing_mocks(self, patches, service, files_response):
        """Helper to set up file listing mocks."""
        mock_files_list = MagicMock()
        mock_files_list.list.return_value.execute.return_value = files_response
        service.files.return_value = mock_files_list
        return mock_files_list

    def test_list_files_with_folder_id(self, config_with_folder, mock_credentials, mock_service, 
                                       mock_files_response, mock_auth_patches):
        """Test listing files with folder ID."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        mock_files_list = self.setup_file_listing_mocks(mock_auth_patches, mock_service, mock_files_response)
        
        connector = GoogleDriveConnector(config_with_folder)
        files = connector.list_files()
        
        expected_query = "'test_folder_id' in parents and (mimeType='application/pdf' or mimeType contains 'image/')"
        mock_files_list.list.assert_called_once_with(
            q=expected_query, 
            fields="files(id, name, mimeType, size)"
        )
        assert files == ["file1", "file2", "file3"]

    def test_list_files_with_path(self, config, mock_credentials, mock_service, 
                                  mock_files_response, mock_auth_patches):
        """Test listing files with path parameter."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        # Use only first file for this test
        single_file_response = {"files": [mock_files_response["files"][0]]}
        mock_files_list = self.setup_file_listing_mocks(mock_auth_patches, mock_service, single_file_response)
        
        connector = GoogleDriveConnector(config)
        files = connector.list_files(path="path_folder_id")
        
        expected_query = "'path_folder_id' in parents and (mimeType='application/pdf' or mimeType contains 'image/')"
        mock_files_list.list.assert_called_once_with(
            q=expected_query, 
            fields="files(id, name, mimeType, size)"
        )
        assert files == ["file1"]

    def test_list_files_with_pattern(self, config, mock_credentials, mock_service, 
                                     mock_files_response, mock_auth_patches):
        """Test listing files with pattern filtering."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        mock_files_list = self.setup_file_listing_mocks(mock_auth_patches, mock_service, mock_files_response)
        
        connector = GoogleDriveConnector(config)
        files = connector.list_files(pattern="*.pdf")
        
        # Should only return PDF files
        assert files == ["file1", "file3"]

    def test_list_files_api_error(self, config, mock_credentials, mock_service, mock_auth_patches):
        """Test listing files when API call fails."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        mock_files_list = MagicMock()
        mock_files_list.list.return_value.execute.side_effect = Exception("API Error")
        mock_service.files.return_value = mock_files_list
        
        connector = GoogleDriveConnector(config)
        
        with pytest.raises(Exception, match="API Error"):
            connector.list_files()

    def setup_download_mocks(self, patches, service, metadata, downloader_chunks=None):
        """Helper to set up download mocks."""
        if downloader_chunks is None:
            downloader_chunks = [(0.5, False), (1.0, True)]
        
        mock_files_get = MagicMock()
        mock_files_get.get.return_value.execute.return_value = metadata
        mock_files_get.get_media.return_value = MagicMock()
        service.files.return_value = mock_files_get
        
        mock_downloader = MagicMock()
        mock_downloader.next_chunk.side_effect = downloader_chunks
        
        return mock_files_get, mock_downloader

    def test_download_file(self, config, mock_credentials, mock_service, mock_file_metadata, 
                           mock_auth_patches, temp_dir):
        """Test downloading a file from Google Drive."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        mock_files_get, mock_downloader = self.setup_download_mocks(
            mock_auth_patches, mock_service, mock_file_metadata
        )
        
        with patch('agentic_doc.connectors.MediaIoBaseDownload', return_value=mock_downloader):
            connector = GoogleDriveConnector(config)
            result_path = connector.download_file("file_id_123")
            
            assert isinstance(result_path, Path)
            assert result_path.name == "test_document.pdf"
            
            mock_files_get.get.assert_called_once_with(fileId="file_id_123")
            mock_files_get.get_media.assert_called_once_with(fileId="file_id_123")

    def test_download_file_with_local_path(self, config, mock_credentials, mock_service, 
                                           mock_file_metadata, mock_auth_patches, temp_dir):
        """Test downloading a file to a specific local path."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        mock_files_get, mock_downloader = self.setup_download_mocks(
            mock_auth_patches, mock_service, mock_file_metadata, downloader_chunks=[(1.0, True)]
        )
        
        with patch('agentic_doc.connectors.MediaIoBaseDownload', return_value=mock_downloader):
            connector = GoogleDriveConnector(config)
            local_path = str(temp_dir / "custom_name.pdf")
            result_path = connector.download_file("file_id_123", local_path)
            
            assert result_path == Path(local_path)

    def test_download_file_api_error(self, config, mock_credentials, mock_service, mock_auth_patches):
        """Test downloading a file when API call fails."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        mock_files_get = MagicMock()
        mock_files_get.get.return_value.execute.side_effect = Exception("Download failed")
        mock_service.files.return_value = mock_files_get
        
        connector = GoogleDriveConnector(config)
        
        with pytest.raises(Exception, match="Download failed"):
            connector.download_file("file_id_123")

    def test_get_file_info(self, config, mock_credentials, mock_service, mock_file_metadata, mock_auth_patches):
        """Test getting file metadata."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        mock_files_get = MagicMock()
        mock_files_get.get.return_value.execute.return_value = mock_file_metadata
        mock_service.files.return_value = mock_files_get
        
        connector = GoogleDriveConnector(config)
        info = connector.get_file_info("file_id_123")
        
        mock_files_get.get.assert_called_once_with(
            fileId="file_id_123",
            fields="id, name, mimeType, size, createdTime, modifiedTime"
        )
        
        assert info["id"] == "file_id_123"
        assert info["name"] == "test_document.pdf"
        assert info["mimeType"] == "application/pdf"
        assert info["size"] == 1024
        assert info["created"] == "2023-01-01T00:00:00Z"
        assert info["modified"] == "2023-01-02T00:00:00Z"

    def test_get_file_info_api_error(self, config, mock_credentials, mock_service, mock_auth_patches):
        """Test getting file info when API call fails."""
        self.setup_auth_mocks(mock_auth_patches, mock_credentials, mock_service, token_exists=False)
        
        mock_files_get = MagicMock()
        mock_files_get.get.return_value.execute.side_effect = Exception("API Error")
        mock_service.files.return_value = mock_files_get
        
        connector = GoogleDriveConnector(config)
        
        with pytest.raises(Exception, match="API Error"):
            connector.get_file_info("file_id_123")


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
        config = GoogleDriveConnectorConfig(client_secret_file="test")
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
