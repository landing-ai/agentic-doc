import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np
import pymupdf
import structlog
from pypdf import PdfReader, PdfWriter
from tenacity import RetryCallState

from agentic_doc.common import Chunk, ChunkGroundingBox, ChunkType, Document
from agentic_doc.config import settings

_LOGGER = structlog.getLogger(__name__)


def get_file_type(file_path: Path) -> Literal["pdf", "image"]:
    """Get the file type of the input file."""
    return "pdf" if file_path.suffix.lower() == ".pdf" else "image"


def save_groundings_as_images(
    file_path: Path,
    chunks: list[Chunk],
    save_dir: Path,
    inplace: bool = True,
) -> dict[str, list[Path]]:
    """
    Save the chunks as images based on the bounding box in each chunk.

    Args:
        file_path (Path): The path to the input document file.
        chunks (list[Chunk]): The chunks to save or update.
        chunk_save_dir (Path): The directory to save the images of the chunks.
        inplace (bool): Whether to update the input chunks in place.

    Returns:
        dict[str, Path]: The dictionary of saved image paths. The key is the chunk id and the value is the path to the saved image.
    """
    file_type = get_file_type(file_path)
    _LOGGER.info(
        f"Saving {len(chunks)} chunks as images to '{save_dir}'",
        file_path=file_path,
        file_type=file_type,
    )
    result: dict[str, list[Path]] = {}
    save_dir.mkdir(parents=True, exist_ok=True)
    if file_type == "image":
        img = cv2.imread(str(file_path))
        return _crop_groundings(img, chunks, save_dir, inplace)

    assert file_type == "pdf"
    chunks_by_page_idx = defaultdict(list)
    for chunk in chunks:
        if chunk.chunk_type == ChunkType.error:
            continue

        page_idx = chunk.grounding[0].page
        chunks_by_page_idx[page_idx].append(chunk)

    with pymupdf.open(file_path) as pdf_doc:
        for page_idx, chunks in sorted(chunks_by_page_idx.items()):
            page_img = page_to_image(pdf_doc, page_idx)
            page_result = _crop_groundings(page_img, chunks, save_dir, inplace)
            result.update(page_result)

    return result


def page_to_image(
    pdf_doc: pymupdf.Document, page_idx: int, dpi: int = 96
) -> np.ndarray:
    """Convert a PDF page to an image. We specifically use pymupdf because it is self-contained and correctly renders annotations."""
    page = pdf_doc[page_idx]
    # Scale image and use RGB colorspace
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, -1)
    # Ensure the image has 3 channels (sometimes it may include an alpha channel)
    if img.shape[-1] == 4:  # If RGBA, drop the alpha channel
        img = img[..., :3]
    return img


def _crop_groundings(
    img: np.ndarray,
    chunks: list[Chunk],
    crop_save_dir: Path,
    inplace: bool = True,
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for c in chunks:
        if c.chunk_type == ChunkType.error:
            continue
        for i, grounding in enumerate(c.grounding):
            if grounding.box is None:
                _LOGGER.error(
                    "Grounding has no bounding box in non-error chunk",
                    grounding=grounding,
                    chunk=c,
                )
                continue

            cropped = _crop_image(img, grounding.box)
            # Convert the cropped image to PNG bytes
            is_success, buffer = cv2.imencode(".png", cropped)
            if not is_success:
                _LOGGER.error(
                    "Failed to encode cropped image as PNG",
                    grounding=grounding,
                )
                continue

            page = f"page_{grounding.page}"
            crop_save_path = crop_save_dir / page / f"{c.chunk_id}_{i}.png"
            crop_save_path.parent.mkdir(parents=True, exist_ok=True)
            crop_save_path.write_bytes(buffer.tobytes())
            assert c.chunk_id is not None
            result[c.chunk_id].append(crop_save_path)
            if inplace:
                c.grounding[i].image_path = crop_save_path

    return result


def _crop_image(image: np.ndarray, bbox: ChunkGroundingBox) -> np.ndarray:
    # Extract coordinates from the bounding box
    xmin, ymin, xmax, ymax = bbox.l, bbox.t, bbox.r, bbox.b

    # Convert normalized coordinates to absolute coordinates
    height, width = image.shape[:2]
    assert 0 <= xmin <= 1 and 0 <= ymin <= 1 and 0 <= xmax <= 1 and 0 <= ymax <= 1
    xmin = math.floor(xmin * width)
    xmax = math.ceil(xmax * width)
    ymin = math.floor(ymin * height)
    ymax = math.ceil(ymax * height)

    # Ensure coordinates are valid
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    return image[ymin:ymax, xmin:xmax]


def split_pdf(
    input_pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    split_size: int = 2,
) -> list[Document]:
    """
    Splits a PDF file into smaller PDFs, each with at most max_pages pages.

    Args:
        input_pdf_path (str | Path): Path to the input PDF file.
        output_dir (str | Path): Directory where mini PDF files will be saved.
        split_size (int): Maximum number of pages per mini PDF file (default is 2, which is the server endpoint's limit).
    """
    input_pdf_path = Path(input_pdf_path)
    assert input_pdf_path.exists(), f"Input PDF file not found: {input_pdf_path}"
    assert input_pdf_path.suffix == ".pdf", "Input file must be a PDF"
    assert (
        0 < split_size <= 2
    ), "split_size must be greater than 0 and less than or equal to 2"

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
