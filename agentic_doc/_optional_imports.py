"""Helper functions for optional dependency imports with clear error messages.

This module provides lazy import functions that only attempt to import optional
dependencies when actually needed, providing clear error messages if they're missing.
"""

from types import ModuleType
from typing import Any, Dict


def import_boto3() -> ModuleType:
    """Lazy import boto3 with helpful error message.

    Returns:
        The boto3 module

    Raises:
        ImportError: If boto3 is not installed
    """
    try:
        import boto3

        return boto3  # type: ignore[no-any-return]
    except ImportError:
        raise ImportError(
            "The S3 connector requires boto3. "
            "Install with: pip install 'agentic-doc[s3]' or pip install boto3"
        )


def import_botocore() -> type:
    """Lazy import botocore ClientCreator.

    Returns:
        The ClientCreator class from botocore

    Raises:
        ImportError: If botocore is not installed
    """
    try:
        from botocore.client import ClientCreator

        return ClientCreator  # type: ignore[no-any-return]
    except ImportError:
        raise ImportError(
            "The S3 connector requires boto3/botocore. "
            "Install with: pip install 'agentic-doc[s3]' or pip install boto3"
        )


def import_numpy() -> ModuleType:
    """Lazy import numpy.

    Returns:
        The numpy module

    Raises:
        ImportError: If numpy is not installed
    """
    try:
        import numpy

        return numpy
    except ImportError:
        raise ImportError(
            "Visualization features require numpy. "
            "Install with: pip install 'agentic-doc[visualization]' or "
            "pip install numpy"
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
            "Request": Request,
            "Credentials": Credentials,
            "InstalledAppFlow": InstalledAppFlow,
            "build": build,
            "Resource": Resource,
            "MediaIoBaseDownload": MediaIoBaseDownload,
        }
    except ImportError as e:
        raise ImportError(
            "The Google Drive connector requires Google API packages. "
            "Install with: pip install 'agentic-doc[google-drive]' or install "
            "google-api-python-client, google-auth-oauthlib, and google-auth manually. "
            f"Missing package: {str(e)}"
        )


def import_cv2() -> ModuleType:
    """Lazy import OpenCV.

    Returns:
        The cv2 module

    Raises:
        ImportError: If opencv-python-headless is not installed
    """
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "Visualization features require opencv-python-headless. "
            "Install with: pip install 'agentic-doc[visualization]' or "
            "pip install opencv-python-headless"
        )


def import_pymupdf() -> ModuleType:
    """Lazy import PyMuPDF.

    Returns:
        The pymupdf module

    Raises:
        ImportError: If pymupdf is not installed
    """
    try:
        import pymupdf

        return pymupdf  # type: ignore[no-any-return]
    except ImportError:
        raise ImportError(
            "PDF visualization requires pymupdf. "
            "Install with: pip install 'agentic-doc[visualization]' or "
            "pip install pymupdf"
        )
