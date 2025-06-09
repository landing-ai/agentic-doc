import fnmatch
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import httpx
import structlog
from pydantic import BaseModel

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False

import boto3  # type: ignore
from botocore.client import ClientCreator  # type: ignore
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseDownload  # type: ignore

_LOGGER = structlog.getLogger(__name__)


class ConnectorConfig(BaseModel):
    """Base configuration for connectors."""

    connector_type: str


class LocalConnectorConfig(ConnectorConfig):
    """Configuration for local file connector."""

    connector_type: str = "local"
    recursive: bool = False


class GoogleDriveConnectorConfig(ConnectorConfig):
    """Configuration for Google Drive connector."""

    connector_type: str = "google_drive"
    client_secret_file: Optional[str] = None
    folder_id: Optional[str] = None


class S3ConnectorConfig(ConnectorConfig):
    """Configuration for S3 connector."""

    connector_type: str = "s3"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    region_name: str = "us-east-1"
    bucket_name: str


class URLConnectorConfig(ConnectorConfig):
    """Configuration for URL connector."""

    connector_type: str = "url"
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30


class BaseConnector(ABC):
    """Abstract base class for document connectors."""

    _VALID_EXTENSIONS = [
        ".bmp",
        ".dib",
        ".dcx",
        ".eps",
        ".ps",
        ".gif",
        ".icns",
        ".ico",
        ".im",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".pcd",
        ".pcx",
        ".png",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pnm",
        ".sgi",
        ".rgb",
        ".rgba",
        ".bw",
        ".spider",
        ".tga",
        ".targa",
        ".tif",
        ".tiff",
        ".webp",
        ".xbm",
        ".jp2",
        ".j2k",
        ".jpf",
        ".jpx",
        ".j2c",
        ".pdf",
        ".heif",
        ".heic",
    ]

    def __init__(self, config: ConnectorConfig):
        self.config = config

    @abstractmethod
    def list_files(
        self, path: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """
        List available files from the connector.

        Args:
            path: Optional base path to list files from.
            pattern: Optional glob-style pattern to filter results (ignored for URL connectors).

        Returns:
            A list of file identifiers or paths.
        """
        pass

    @abstractmethod
    def download_file(self, file_id: str, local_path: Optional[str] = None) -> Path:
        """
        Download a file to local storage.

        Args:
            file_id: Identifier for the file to download. Format varies by connector:
                - Local: File system path (e.g., "/path/to/file.pdf")
                - Google Drive: Google Drive file ID (e.g., "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
                - S3: S3 object key/path (e.g., "documents/report.pdf")
                - URL: Complete HTTP/HTTPS URL (e.g., "https://example.com/file.pdf")
            local_path: Optional local path to save to

        Returns:
            Path to the downloaded file

        Note:
            Use list_files() to get valid file_id values for each connector type.
        """
        pass

    @abstractmethod
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Get metadata about a file.

        Args:
            file_id: Identifier for the file

        Returns:
            Dictionary containing file metadata
        """
        pass


class LocalConnector(BaseConnector):
    """Connector for local files."""

    def __init__(self, config: LocalConnectorConfig):
        super().__init__(config)
        self.config: LocalConnectorConfig = config

    def list_files(
        self, path: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """List local files."""
        search_path = Path(path) if path else Path.cwd()

        if not search_path.exists():
            raise FileNotFoundError(f"Path does not exist: {search_path}")

        if search_path.is_file():
            return [str(search_path)]

        globber = search_path.rglob if self.config.recursive else search_path.glob
        if pattern:
            files = list(globber(pattern))
        else:
            files = [
                f
                for f in globber("*")
                if f.is_file() and f.suffix.lower() in LocalConnector._VALID_EXTENSIONS
            ]

        return [str(f) for f in files if f.is_file()]

    def download_file(self, file_id: str, local_path: Optional[str] = None) -> Path:
        """For local files, just return the path if it exists."""
        file_path = Path(file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get local file metadata."""
        file_path = Path(file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "suffix": file_path.suffix,
        }


class GoogleDriveConnector(BaseConnector):
    """Connector for Google Drive files."""

    def __init__(self, config: GoogleDriveConnectorConfig):
        super().__init__(config)
        self.config: GoogleDriveConnectorConfig = config
        self._service: Optional[Resource] = None

    def _get_service(self) -> Resource:
        """Initialize Google Drive service with user-friendly OAuth."""
        if self._service is None:
            scopes = ["https://www.googleapis.com/auth/drive.readonly"]
            creds = None

            # Check if we have stored credentials
            if os.path.exists("token.json"):
                creds = Credentials.from_authorized_user_file("token.json", scopes)

            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if self.config.client_secret_file:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.config.client_secret_file, scopes=scopes
                        )
                        creds = flow.run_local_server(port=0)
                    else:
                        raise ValueError("client_secret_file must be provided")

                # Save credentials for next time
                with open("token.json", "w") as token:
                    token.write(creds.to_json())

            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def list_files(
        self, path: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """List files in Google Drive"""
        service = self._get_service()

        # Build query
        query_parts = []
        if self.config.folder_id:
            query_parts.append(f"'{self.config.folder_id}' in parents")
        elif path:
            query_parts.append(f"'{path}' in parents")

        # Filter by file types
        file_types = ["mimeType='application/pdf'", "mimeType contains 'image/'"]
        query_parts.append(f"({' or '.join(file_types)})")

        query = " and ".join(query_parts)

        try:
            results = (
                service.files()
                .list(q=query, fields="files(id, name, mimeType, size)")
                .execute()
            )
            files = results.get("files", [])

            # Apply glob pattern filtering
            if pattern:
                files = [f for f in files if fnmatch.fnmatch(f["name"], pattern)]

            return [file["id"] for file in files]
        except Exception as e:
            _LOGGER.error(f"Error listing Google Drive files: {e}")
            raise

    def download_file(self, file_id: str, local_path: Optional[str] = None) -> Path:
        """Download file from Google Drive."""
        service = self._get_service()

        try:
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata["name"]

            # Create local path if not provided
            if local_path is None:
                temp_dir = tempfile.mkdtemp()
                local_path_obj: Path = Path(temp_dir) / file_name
            else:
                local_path_obj = Path(local_path)
                local_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            request = service.files().get_media(fileId=file_id)
            with open(local_path_obj, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

            _LOGGER.info(f"Downloaded Google Drive file {file_id} to {local_path_obj}")
            return local_path_obj

        except Exception as e:
            _LOGGER.error(f"Error downloading Google Drive file {file_id}: {e}")
            raise

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get Google Drive file metadata."""
        service = self._get_service()

        try:
            file_metadata = (
                service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, mimeType, size, createdTime, modifiedTime",
                )
                .execute()
            )

            return {
                "id": file_metadata["id"],
                "name": file_metadata["name"],
                "mimeType": file_metadata["mimeType"],
                "size": int(file_metadata.get("size", 0)),
                "created": file_metadata.get("createdTime"),
                "modified": file_metadata.get("modifiedTime"),
            }

        except Exception as e:
            _LOGGER.error(f"Error getting Google Drive file info for {file_id}: {e}")
            raise


class S3Connector(BaseConnector):
    """Connector for Amazon S3 files."""

    def __init__(self, config: S3ConnectorConfig):
        super().__init__(config)
        self.config: S3ConnectorConfig = config
        self._client: Optional[ClientCreator] = None

    def _get_client(self) -> ClientCreator:
        """Initialize S3 client if not already done."""
        if self._client is None:
            kwargs = {"region_name": self.config.region_name}

            if self.config.aws_access_key_id:
                kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            if self.config.aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            if self.config.aws_session_token:
                kwargs["aws_session_token"] = self.config.aws_session_token

            self._client = boto3.client("s3", **kwargs)

        return self._client

    def list_files(
        self, path: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """List files in S3 bucket"""
        client = self._get_client()

        try:
            kwargs = {"Bucket": self.config.bucket_name}
            if path:
                kwargs["Prefix"] = path

            response = client.list_objects_v2(**kwargs)

            files = []
            for obj in response.get("Contents", []):
                key = obj["Key"]

                # Filter by file extension (documents and images)
                if any(
                    key.lower().endswith(ext) for ext in S3Connector._VALID_EXTENSIONS
                ):
                    # Apply glob pattern filtering
                    if not pattern or fnmatch.fnmatch(key, pattern):
                        files.append(key)

            return files

        except Exception as e:
            _LOGGER.error(f"Error listing S3 files: {e}")
            raise

    def download_file(self, file_id: str, local_path: Optional[str] = None) -> Path:
        """Download file from S3."""
        client = self._get_client()

        try:
            # Create local path if not provided
            if local_path is None:
                temp_dir = tempfile.mkdtemp()
                file_name = Path(file_id).name
                local_path_obj = Path(temp_dir) / file_name
            else:
                local_path_obj = Path(local_path)
                local_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            client.download_file(self.config.bucket_name, file_id, str(local_path_obj))

            _LOGGER.info(f"Downloaded S3 file {file_id} to {local_path_obj}")
            return local_path_obj

        except Exception as e:
            _LOGGER.error(f"Error downloading S3 file {file_id}: {e}")
            raise

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get S3 file metadata."""
        client = self._get_client()

        try:
            response = client.head_object(Bucket=self.config.bucket_name, Key=file_id)

            return {
                "key": file_id,
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"],
                "content_type": response.get("ContentType"),
            }

        except Exception as e:
            _LOGGER.error(f"Error getting S3 file info for {file_id}: {e}")
            raise


class URLConnector(BaseConnector):
    """Connector for files accessible via HTTP/HTTPS URLs."""

    def __init__(self, config: URLConnectorConfig):
        super().__init__(config)
        self.config: URLConnectorConfig = config

    def list_files(
        self, path: Optional[str] = None, pattern: Optional[str] = None
    ) -> List[str]:
        """For URL connector, just return the provided path as a single file."""
        if path:
            return [path]
        return []

    def download_file(self, file_id: str, local_path: Optional[str] = None) -> Path:
        """Download file from URL."""
        try:
            # Create local path if not provided
            if local_path is None:
                temp_dir = tempfile.mkdtemp()
                file_name = Path(file_id).name or "downloaded_file"
                local_path_obj = Path(temp_dir) / file_name
            else:
                local_path_obj = Path(local_path)
                local_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            headers = self.config.headers or {}

            with httpx.stream(
                "GET", file_id, headers=headers, timeout=self.config.timeout
            ) as response:
                response.raise_for_status()

                with open(local_path_obj, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            _LOGGER.info(f"Downloaded URL {file_id} to {local_path_obj}")
            return local_path_obj

        except Exception as e:
            _LOGGER.error(f"Error downloading URL {file_id}: {e}")
            raise

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get file info from URL headers."""
        try:
            headers = self.config.headers or {}

            response = httpx.head(file_id, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()

            return {
                "url": file_id,
                "content_length": response.headers.get("content-length"),
                "content_type": response.headers.get("content-type"),
                "last_modified": response.headers.get("last-modified"),
            }

        except Exception as e:
            _LOGGER.error(f"Error getting URL file info for {file_id}: {e}")
            raise


def create_connector(config: ConnectorConfig) -> BaseConnector:
    """Factory function to create appropriate connector based on config type."""
    connector_map: Dict[str, Type[BaseConnector]] = {
        "local": LocalConnector,
        "google_drive": GoogleDriveConnector,
        "s3": S3Connector,
        "url": URLConnector,
    }

    connector_class = connector_map.get(config.connector_type)
    if not connector_class:
        raise ValueError(f"Unknown connector type: {config.connector_type}")

    return connector_class(config)
