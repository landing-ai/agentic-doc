import copy
from functools import partial
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Union, cast

import httpx
import structlog
import tenacity
from tqdm import tqdm

from agentic_doc.common import (
    Chunk,
    Document,
    ParsedDocument,
    RetryableError,
    Timer,
)
from agentic_doc.config import settings
from agentic_doc.utils import log_retry_failure, split_pdf

_LOGGER = structlog.getLogger(__name__)
_ENDPOINT_URL = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"


def parse_documents(file_paths: list[Union[str, Path]]) -> list[ParsedDocument]:
    """
    Parse a list of documents using the Landing AI Agentic Document Analysis API.

    Args:
        file_paths (list[str | Path]): The list of file paths to the documents to parse.

    Returns:
        list[dict[str, Any]]: The list of parsed documents. The list is sorted by the order of the input file paths.
    """
    for file_path in file_paths:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    _LOGGER.info(f"Parsing {len(file_paths)} documents")
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(
                    parse_and_save_document,  # type: ignore [arg-type]
                    file_paths,
                ),
                total=len(file_paths),
                desc="Parsing documents",
            )
        )


def parse_and_save_documents(
    file_paths: list[Union[str, Path]], *, result_save_dir: Union[str, Path]
) -> list[Path]:
    """
    Parse a list of documents and save the results to a local directory.

    Args:
        file_paths (list[str | Path]): The list of file paths to the documents to parse.
        result_save_dir (str | Path): The local directory to save the results.

    Returns:
        list[Path]: A list of json file paths to the saved results. The file paths are sorted by the order of the input file paths.
            The file name is the original file name with a timestamp appended. E.g. "document.pdf" -> "document_20250313_123456.json".
    """
    for file_path in file_paths:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    _LOGGER.info(f"Parsing {len(file_paths)} documents")
    _parse_func = partial(parse_and_save_document, result_save_dir=result_save_dir)
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, file_paths),  # type: ignore [arg-type]
                total=len(file_paths),
                desc="Parsing documents",
            )
        )


def parse_and_save_document(
    file_path: Union[str, Path],
    *,
    result_save_dir: Union[str, Path, None] = None,
) -> Union[Path, ParsedDocument]:
    """
    Parse a document and save the results to a local directory.

    Args:
        file_path (str | Path): The path to the document file.
        result_save_dir (str | Path): The local directory to save the results. If None, the parsed document data is returned.

    Returns:
        Path | ParsedDocument: The file path to the saved result or the parsed document data.
    """
    file_path = Path(file_path)
    file_type = "pdf" if file_path.suffix.lower() == ".pdf" else "image"

    if file_type == "image":
        result = _parse_image(file_path)
    elif file_type == "pdf":
        result = _parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if not result_save_dir:
        return result

    result_save_dir = Path(result_save_dir)
    result_save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = result_save_dir / f"{Path(file_path).stem}_{ts}.json"
    save_path.write_text(result.model_dump_json())
    _LOGGER.info(f"Saved the parsed result to '{save_path}'")

    return save_path


def _parse_pdf(file_path: Union[str, Path]) -> ParsedDocument:
    with tempfile.TemporaryDirectory() as temp_dir:
        parts = split_pdf(file_path, temp_dir)
        file_path = Path(file_path)
        part_results = _parse_doc_in_parallel(parts, doc_name=file_path.name)
        return _merge_part_results(part_results)


def _parse_image(file_path: Union[str, Path]) -> ParsedDocument:
    try:
        result_raw = _send_parsing_request(str(file_path))
        result_raw = {
            **result_raw["data"],
            "doc_type": "image",
            "start_page_idx": 0,
            "end_page_idx": 0,
        }
        return ParsedDocument.model_validate(result_raw)
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing image '{file_path}' due to: {error_msg}")
        chunks = [Chunk.error_chunk(error_msg, 0)]
        return ParsedDocument(
            markdown="",
            chunks=chunks,
            start_page_idx=0,
            end_page_idx=0,
            doc_type="image",
        )


def _merge_part_results(results: list[ParsedDocument]) -> ParsedDocument:
    if not results:
        _LOGGER.warning(
            f"No results to merge: {results}, returning empty ParsedDocument"
        )
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=0,
            end_page_idx=0,
            doc_type="pdf",
        )

    init_result = copy.deepcopy(results[0])
    for i in range(1, len(results)):
        _merge_next_part(init_result, results[i])

    return init_result


def _merge_next_part(curr: ParsedDocument, next: ParsedDocument) -> None:
    curr.markdown += "\n\n" + next.markdown
    next_chunks = next.chunks
    for chunk in next_chunks:
        for grounding in chunk.grounding:
            grounding.page += next.start_page_idx

    curr.chunks.extend(next_chunks)
    curr.end_page_idx = next.end_page_idx


def _parse_doc_in_parallel(
    doc_parts: list[Document], *, doc_name: str
) -> list[ParsedDocument]:
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        return list(
            tqdm(
                executor.map(_parse_doc_parts, doc_parts),
                total=len(doc_parts),
                desc=f"Parsing document parts from '{doc_name}'",
            )
        )


def _parse_doc_parts(doc: Document) -> ParsedDocument:
    try:
        _LOGGER.info(f"Start parsing document part: '{doc}'")
        result = _send_parsing_request(str(doc.file_path))
        _LOGGER.info(f"Successfully parsed document part: '{doc}'")
        return ParsedDocument.model_validate(
            {
                **result["data"],
                "start_page_idx": doc.start_page_idx,
                "end_page_idx": doc.end_page_idx,
                "doc_type": "pdf",
            }
        )
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing document '{doc}' due to: {error_msg}")
        chunks = [
            Chunk.error_chunk(error_msg, doc.start_page_idx + i)
            for i in range(doc.start_page_idx, doc.end_page_idx + 1)
        ]
        return ParsedDocument(
            markdown="",
            chunks=chunks,
            start_page_idx=doc.start_page_idx,
            end_page_idx=doc.end_page_idx,
            doc_type="pdf",
        )


@tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        exp_base=1.5, initial=1, max=settings.max_retry_wait_time, jitter=10
    ),
    stop=tenacity.stop_after_attempt(settings.max_retries),
    retry=tenacity.retry_if_exception_type(RetryableError),
    after=log_retry_failure,
)
def _send_parsing_request(file_path: str) -> dict[str, Any]:
    """
    Send a parsing request to the Landing AI Agentic Document Analysis API.

    Args:
        file_path (str): The path to the document file.

    Returns:
        dict[str, Any]: The parsed document data.
    """
    with Timer() as timer:
        file_type = "pdf" if Path(file_path).suffix.lower() == ".pdf" else "image"
        # TODO: check if the file extension is a supported image type
        with open(file_path, "rb") as file:
            files = {file_type: file}
            headers = {"Authorization": f"Basic {settings.vision_agent_api_key}"}
            response = httpx.post(
                _ENDPOINT_URL,
                files=files,
                headers=headers,
                timeout=None,
            )
            if response.status_code in [408, 429, 502, 503, 504]:
                raise RetryableError(response)

            response.raise_for_status()

    _LOGGER.info(
        f"Time taken to successfully parse a document chunk: {timer.elapsed:.2f} seconds"
    )
    result = cast(dict[str, Any], response.json())
    return result
