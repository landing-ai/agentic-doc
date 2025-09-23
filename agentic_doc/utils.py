import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Union, Optional, List, TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import requests
import structlog
from pydantic_core import Url
from pypdf import PdfReader, PdfWriter
from tenacity import RetryCallState

from agentic_doc.common import Chunk, ChunkGroundingBox, Document, ParsedDocument
from agentic_doc.config import get_settings

# Optional imports for visualization
try:
    import cv2
    import numpy as np
    import pymupdf
    from PIL import Image
    from ade_visualization import (
        viz_parsed_document as _viz_parsed_document,
        viz_chunks as _viz_chunks,
        save_groundings_as_images as _save_groundings_as_images,
        VisualizationConfig,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    if TYPE_CHECKING:
        import numpy as np
        from PIL import Image
        from ade_visualization import VisualizationConfig

_LOGGER = structlog.getLogger(__name__)


def check_endpoint_and_api_key(endpoint_url: str, api_key: str) -> None:
    """Check if the API key is valid and if the endpoint is up."""
    if not api_key:
        raise ValueError("API key is not set. Please provide a valid API key.")

    headers = {"Authorization": f"Basic {api_key}"}

    try:
        response = requests.head(endpoint_url, headers=headers, timeout=5)
    except requests.exceptions.ConnectionError:
        raise ValueError(f'The endpoint URL "{endpoint_url}" is down or invalid.')

    if response.status_code == 404:
        raise ValueError("API key is not valid for this endpoint.")
    elif response.status_code == 401:
        raise ValueError("API key is invalid")

    _LOGGER.info("API key is valid.")


def get_file_type(file_path: Path) -> Literal["pdf", "image"]:
    """Get the file type of the input file by checking its magic number.

    PDF files start with '%PDF-' (25 50 44 46 2D in hex)
    """
    try:
        with open(file_path, "rb") as f:
            # Read the first 5 bytes to check for PDF magic number
            header = f.read(5)
            if header == b"%PDF-":
                return "pdf"
            return "image"
    except Exception as e:
        _LOGGER.warning(f"Error checking file type: {e}")
        # Fallback to extension check if file reading fails
        return "pdf" if file_path.suffix.lower() == ".pdf" else "image"


def save_groundings_as_images(
    file_path: Path,
    chunks: list[Chunk],
    save_dir: Path,
    inplace: bool = True,
) -> dict[str, List[Path]]:
    """
    Save the chunks as images based on the bounding box in each chunk.

    Args:
        file_path (Path): The path to the input document file.
        chunks (list[Chunk]): The chunks to save or update.
        save_dir (Path): The directory to save the images of the chunks.
        inplace (bool): Whether to update the input chunks in place.

    Returns:
        dict[str, Path]: The dictionary of saved image paths. The key is the chunk id and the value is the path to the saved image.
    """
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization is not available. Install it with: "
            "pip install ade-visualization"
        )
    return _save_groundings_as_images(file_path, chunks, save_dir, inplace)


def get_chunk_from_reference(chunk_id: str, chunks: list[dict]) -> Optional[dict]:
    return next((chunk for chunk in chunks if chunk.get("chunk_id") == chunk_id), None)


def split_pdf(
    input_pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    split_size: int = 10,
) -> list[Document]:
    """
    Splits a PDF file into smaller PDFs, each with at most max_pages pages.

    Args:
        input_pdf_path (str | Path): Path to the input PDF file.
        output_dir (str | Path): Directory where mini PDF files will be saved.
        split_size (int): Maximum number of pages per mini PDF file (default is 10).
    """
    input_pdf_path = Path(input_pdf_path)
    assert input_pdf_path.exists(), f"Input PDF file not found: {input_pdf_path}"
    assert (
        0 < split_size <= 100
    ), "split_size must be greater than 0 and less than or equal to 100"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    pdf_reader = PdfReader(input_pdf_path)
    total_pages = len(pdf_reader.pages)
    _LOGGER.info(
        f"Splitting PDF: '{input_pdf_path}' into {total_pages // split_size} parts under '{output_dir}'"
    )
    file_count = 1

    output_pdfs = []
    # Process the PDF in chunks of max_pages pages
    for start in range(0, total_pages, split_size):
        pdf_writer = PdfWriter()
        # Add up to max_pages pages to the new PDF writer
        for page_num in range(start, min(start + split_size, total_pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        output_pdf = os.path.join(output_dir, f"{input_pdf_path.stem}_{file_count}.pdf")
        with open(output_pdf, "wb") as out_file:
            pdf_writer.write(out_file)
        _LOGGER.info(f"Created {output_pdf}")
        file_count += 1
        output_pdfs.append(
            Document(
                file_path=output_pdf,
                start_page_idx=start,
                end_page_idx=min(start + split_size - 1, total_pages - 1),
            )
        )

    return output_pdfs


def log_retry_failure(retry_state: RetryCallState) -> None:
    settings = get_settings()
    if retry_state.outcome and retry_state.outcome.failed:
        if settings.retry_logging_style == "log_msg":
            exception = retry_state.outcome.exception()
            func_name = (
                retry_state.fn.__name__ if retry_state.fn else "unknown_function"
            )
            # TODO: add a link to the error FAQ page
            _LOGGER.debug(
                f"'{func_name}' failed on attempt {retry_state.attempt_number}. Error: '{exception}'.",
            )
        elif settings.retry_logging_style == "inline_block":
            # Print yellow progress block that updates on the same line
            print(
                f"\r\033[33m{'█' * retry_state.attempt_number}\033[0m",
                end="",
                flush=True,
            )
        elif settings.retry_logging_style == "none":
            pass
        else:
            raise ValueError(
                f"Invalid retry logging style: {settings.retry_logging_style}"
            )


def viz_parsed_document(
    file_path: Union[str, Path],
    parsed_document: ParsedDocument,
    *,
    output_dir: Union[str, Path, None] = None,
    viz_config: Union['VisualizationConfig', None] = None,
) -> List['Image.Image']:
    """Visualize a parsed document with bounding boxes."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization is not available. Install it with: "
            "pip install ade-visualization"
        )
    return _viz_parsed_document(file_path, parsed_document, output_dir=output_dir, viz_config=viz_config)


def viz_chunks(
    img: 'np.ndarray',
    chunks: List[Chunk],
    viz_config: Union['VisualizationConfig', None] = None,
) -> 'np.ndarray':
    """Visualize chunks on an image."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization is not available. Install it with: "
            "pip install ade-visualization"
        )
    return _viz_chunks(img, chunks, viz_config)


def download_file(file_url: Url, output_filepath: str) -> None:
    """
    Downloads a file from the given media URL to the specified local path.

    Parameters:
    media_url (Url): The URL of the media file to download.
    path (str): The local file system path where the file should be saved.

    Raises:
    Exception: If the download fails (non-200 status code).
    """
    _LOGGER.info(f"Downloading file from '{file_url}' to '{output_filepath}'")
    with httpx.stream("GET", str(file_url), timeout=None) as response:
        if response.status_code != 200:
            raise Exception(
                f"Download failed for '{file_url}'. Status code: {response.status_code} {response.text}"
            )

        with open(output_filepath, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024):
                f.write(chunk)


def is_valid_httpurl(url: str) -> bool:
    """Check if the given URL is a valid HTTP URL."""
    try:
        parsed_url = urlparse(url)
        return parsed_url.scheme in ["http", "https"]
    except Exception:
        return False
