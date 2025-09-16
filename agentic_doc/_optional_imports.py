"""Helper functions for optional dependency imports with clear error messages.

This module provides lazy import functions that only attempt to import optional
dependencies when actually needed, providing clear error messages if they're missing.
"""

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import boto3
    import cv2
    import pymupdf
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import Resource
    from googleapiclient.http import MediaIoBaseDownload


def import_boto3() -> Any:
    """Lazy import boto3 with helpful error message."""
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "The S3 connector requires boto3. "
            "Install with: pip install 'agentic-doc[s3]' or pip install boto3"
        )


def import_botocore() -> Any:
    """Lazy import botocore client."""
    try:
        from botocore.client import ClientCreator
        return ClientCreator
    except ImportError:
        raise ImportError(
            "The S3 connector requires boto3/botocore. "
            "Install with: pip install 'agentic-doc[s3]' or pip install boto3"
        )


def import_google_packages() -> Dict[str, Any]:
    """Import all Google-related packages."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build, Resource
        from googleapiclient.http import MediaIoBaseDownload

        return {
            'Request': Request,
            'Credentials': Credentials,
            'InstalledAppFlow': InstalledAppFlow,
            'build': build,
            'Resource': Resource,
            'MediaIoBaseDownload': MediaIoBaseDownload
        }
    except ImportError as e:
        raise ImportError(
            "The Google Drive connector requires Google API packages. "
            "Install with: pip install 'agentic-doc[google-drive]' or install "
            "google-api-python-client, google-auth-oauthlib, and google-auth manually. "
            f"Missing package: {str(e)}"
        )


def import_cv2() -> Any:
    """Lazy import OpenCV."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "Visualization features require opencv-python-headless. "
            "Install with: pip install 'agentic-doc[visualization]' or "
            "pip install opencv-python-headless"
        )


def import_pymupdf() -> Any:
    """Lazy import PyMuPDF."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        raise ImportError(
            "PDF visualization requires pymupdf. "
            "Install with: pip install 'agentic-doc[visualization]' or "
            "pip install pymupdf"
        )


# Note: pillow_heif import helper removed as it's not currently used
# Can be re-added if HEIF support is implemented in the future

# OpenCV font constant value (cv2.FONT_HERSHEY_SIMPLEX)
# This allows us to use the constant without importing cv2
FONT_HERSHEY_SIMPLEX = 0