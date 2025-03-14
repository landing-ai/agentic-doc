import os
from pathlib import Path
from typing import Union

import structlog
from pypdf import PdfReader, PdfWriter
from tenacity import RetryCallState

from agentic_doc.common import Document
from agentic_doc.config import settings

_LOGGER = structlog.getLogger(__name__)


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
