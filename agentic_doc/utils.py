import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Union
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
import pymupdf
import structlog
from PIL import Image
from pydantic import AnyHttpUrl
from pypdf import PdfReader, PdfWriter
from tenacity import RetryCallState

from agentic_doc.common import (
    Chunk,
    ChunkGroundingBox,
    ChunkType,
    Document,
    ParsedDocument,
)
from agentic_doc.config import VisualizationConfig, settings

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
        save_dir (Path): The directory to save the images of the chunks.
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
    pdf_doc: pymupdf.Document, page_idx: int, dpi: int = settings.pdf_to_image_dpi
) -> np.ndarray:
    """Convert a PDF page to an image. We specifically use pymupdf because it is self-contained and correctly renders annotations."""
    page = pdf_doc[page_idx]
    # Scale image and use RGB colorspace
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    img: np.ndarray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, -1
    )
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
    xmin_f, ymin_f, xmax_f, ymax_f = bbox.l, bbox.t, bbox.r, bbox.b

    # Convert normalized coordinates to absolute coordinates
    height, width = image.shape[:2]
    assert (
        0 <= xmin_f <= 1 and 0 <= ymin_f <= 1 and 0 <= xmax_f <= 1 and 0 <= ymax_f <= 1
    )
    xmin = math.floor(xmin_f * width)
    xmax = math.ceil(xmax_f * width)
    ymin = math.floor(ymin_f * height)
    ymax = math.ceil(ymax_f * height)

    # Ensure coordinates are valid
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    result: np.ndarray = image[ymin:ymax, xmin:xmax]
    return result


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


def viz_parsed_document(
    file_path: Union[str, Path],
    parsed_document: ParsedDocument,
    *,
    output_dir: Union[str, Path, None] = None,
    viz_config: Union[VisualizationConfig, None] = None,
) -> list[Image.Image]:
    if viz_config is None:
        viz_config = VisualizationConfig()

    viz_result_np: list[np.ndarray] = []
    file_path = Path(file_path)
    file_type = get_file_type(file_path)
    _LOGGER.info(f"Visualizing parsed document of: '{file_path}'")
    if file_type == "image":
        img = _read_img_rgb(str(file_path))
        viz_np = viz_chunks(img, parsed_document.chunks, viz_config)
        viz_result_np.append(viz_np)
    else:
        with pymupdf.open(file_path) as pdf_doc:
            for page_idx in range(
                parsed_document.start_page_idx, parsed_document.end_page_idx + 1
            ):
                if page_idx > 1:
                    break
                img = page_to_image(pdf_doc, page_idx)
                chunks = [
                    chunk
                    for chunk in parsed_document.chunks
                    if chunk.grounding[0].page == page_idx
                ]
                viz_np = viz_chunks(img, chunks, viz_config)
                viz_result_np.append(viz_np)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, viz_np in enumerate(viz_result_np):
            viz_np = cv2.cvtColor(viz_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{file_path.stem}_viz_page_{i}.png"), viz_np)

    return [Image.fromarray(viz_np) for viz_np in viz_result_np]


def viz_chunks(
    img: np.ndarray,
    chunks: list[Chunk],
    viz_config: Union[VisualizationConfig, None] = None,
) -> np.ndarray:
    if viz_config is None:
        viz_config = VisualizationConfig()

    viz = img.copy()
    viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    for i, chunk in enumerate(chunks):
        show_grounding_idx = len(chunk.grounding) > 1
        for j, grounding in enumerate(chunk.grounding):
            assert grounding.box is not None
            xmin, ymin, xmax, ymax = (
                max(0, math.floor(grounding.box.l * width)),
                max(0, math.floor(grounding.box.t * height)),
                min(width, math.ceil(grounding.box.r * width)),
                min(height, math.ceil(grounding.box.b * height)),
            )
            box = (xmin, ymin, xmax, ymax)
            idx = f"{i}-{j}" if show_grounding_idx else f"{i}"
            _place_mark(
                viz,
                box,
                text=f"{idx} {chunk.chunk_type}",
                color_bgr=viz_config.color_map[chunk.chunk_type],
                viz_config=viz_config,
            )

    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
    return viz


def _place_mark(
    img: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    text: str,
    *,
    color_bgr: tuple[int, int, int],
    viz_config: VisualizationConfig,
) -> None:
    text_color = color_bgr
    (text_width, text_height), baseline = cv2.getTextSize(
        text, viz_config.font, viz_config.font_scale, viz_config.thickness
    )
    text_x = int((box_xyxy[0] + box_xyxy[2] - text_width) // 2)
    text_y = int((box_xyxy[1] + box_xyxy[3] + text_height) // 2)

    # Draw the text background with opacity
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (text_x - viz_config.padding, text_y - text_height - viz_config.padding),
        (
            text_x + text_width + viz_config.padding,
            text_y + baseline + viz_config.padding,
        ),
        viz_config.text_bg_color,
        -1,
    )
    cv2.addWeighted(
        overlay, viz_config.text_bg_opacity, img, 1 - viz_config.text_bg_opacity, 0, img
    )

    # Draw the text on top
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        viz_config.font,
        viz_config.font_scale,
        text_color,
        viz_config.thickness,
        cv2.LINE_AA,
    )
    # Draw the bounding box
    cv2.rectangle(img, box_xyxy[:2], box_xyxy[2:], color_bgr, viz_config.thickness)


def _read_img_rgb(img_path: str) -> np.ndarray:
    """
    Read a image given its path.
    Arguments:
        img_path : image file path
    Returns:
        img (H, W, 3): a numpy array image in RGB format
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def download_file(file_url: AnyHttpUrl, output_filepath: str) -> None:
    """
    Downloads a file from the given media URL to the specified local path.

    Parameters:
    media_url (AnyHttpUrl): The URL of the media file to download.
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
