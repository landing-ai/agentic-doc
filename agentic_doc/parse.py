import copy
import importlib.metadata
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union, cast

import httpx
import structlog
import tenacity
from pydantic_core import Url
from tqdm import tqdm

from agentic_doc.common import (
    Document,
    PageError,
    ParsedDocument,
    RetryableError,
    Timer,
)
from agentic_doc.config import settings
from agentic_doc.connectors import (
    BaseConnector,
    ConnectorConfig,
    GoogleDriveConnectorConfig,
    LocalConnectorConfig,
    S3ConnectorConfig,
    URLConnectorConfig,
    create_connector,
)
from agentic_doc.utils import (
    download_file,
    get_file_type,
    is_valid_httpurl,
    log_retry_failure,
    save_groundings_as_images,
    split_pdf,
)

_LOGGER = structlog.getLogger(__name__)
_ENDPOINT_URL = f"{settings.endpoint_host}/v1/tools/agentic-document-analysis"
_LIB_VERSION = importlib.metadata.version("agentic-doc")


def parse(
    documents: Union[
        str, Path, Url, List[Union[str, Path, Url]], BaseConnector, ConnectorConfig
    ],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
) -> Union[ParsedDocument, List[ParsedDocument], List[Path]]:
    """
    Universal parse function that can handle single documents, lists of documents,
    or documents from various connectors.

    Args:
        documents: Can be:
            - Single document path/URL (str, Path, Url)
            - List of document paths/URLs
            - Connector instance (BaseConnector)
            - Connector configuration (ConnectorConfig)
        include_marginalia: Whether to include marginalia in the analysis
        include_metadata_in_markdown: Whether to include metadata in markdown output
        result_save_dir: Directory to save results (if provided, returns file paths)
        grounding_save_dir: Directory to save grounding images
        connector_path: Path for connector to search (when using connectors)
        connector_pattern: Pattern to filter files (when using connectors)

    Returns:
        - Single ParsedDocument (if single document and no result_save_dir)
        - List[ParsedDocument] (if multiple documents and no result_save_dir)
        - List[Path] (if result_save_dir is provided)
    """
    # Handle connector-based parsing
    if isinstance(documents, (BaseConnector, ConnectorConfig)):
        connector = (
            documents
            if isinstance(documents, BaseConnector)
            else create_connector(documents)
        )

        # Get list of files from connector
        file_list = connector.list_files(connector_path, connector_pattern)

        if not file_list:
            _LOGGER.warning("No files found with the specified connector")
            return [] if result_save_dir else []

        # Download files and get local paths
        local_paths = []
        for file_id in file_list:
            try:
                local_path = connector.download_file(file_id)
                local_paths.append(local_path)
            except Exception as e:
                _LOGGER.error(f"Failed to download file {file_id}: {e}")
                continue

        if not local_paths:
            _LOGGER.warning("No files were successfully downloaded")
            return [] if result_save_dir else []

        # Parse the downloaded files
        return _parse_document_list(
            local_paths,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            result_save_dir=result_save_dir,
            grounding_save_dir=grounding_save_dir,
        )

    # Handle single document
    elif isinstance(documents, (str, Path, Url)):
        if result_save_dir:
            result = parse_and_save_document(
                documents,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
                result_save_dir=result_save_dir,
                grounding_save_dir=grounding_save_dir,
            )
            return [result] if isinstance(result, Path) else result
        else:
            result = parse_and_save_document(
                documents,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
                grounding_save_dir=grounding_save_dir,
            )
            return result

    # Handle list of documents
    elif isinstance(documents, list):
        return _parse_document_list(
            documents,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            result_save_dir=result_save_dir,
            grounding_save_dir=grounding_save_dir,
        )

    else:
        raise ValueError(f"Unsupported documents type: {type(documents)}")


def _parse_document_list(
    documents: List[Union[str, Path, Url]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
) -> Union[List[ParsedDocument], List[Path]]:
    """Helper function to parse a list of documents."""
    if result_save_dir:
        return parse_and_save_documents(
            documents,
            result_save_dir=result_save_dir,
            grounding_save_dir=grounding_save_dir,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
    else:
        return parse_documents(
            documents,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            grounding_save_dir=grounding_save_dir,
        )


def parse_documents(
    documents: list[Union[str, Path, Url]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    grounding_save_dir: Union[str, Path, None] = None,
) -> list[ParsedDocument]:
    """
    Parse a list of documents using the Landing AI Agentic Document Analysis API.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse. Each document can be a local file path, a URL string, or a Pydantic `Url` object.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
    Returns:
        list[ParsedDocument]: The list of parsed documents. The list is sorted by the order of the input documents.
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func = partial(
        parse_and_save_document,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        grounding_save_dir=grounding_save_dir,
    )
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),  # type: ignore [arg-type]
                total=len(documents),
                desc="Parsing documents",
            )
        )


def parse_and_save_documents(
    documents: list[Union[str, Path, Url]],
    *,
    result_save_dir: Union[str, Path],
    grounding_save_dir: Union[str, Path, None] = None,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> list[Path]:
    """
    Parse a list of documents and save the results to a local directory.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse. Each document can be a local file path, a URL string, or a Pydantic `Url` object.
        result_save_dir (str | Path): The local directory to save the results.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
    Returns:
        list[Path]: A list of json file paths to the saved results. The file paths are sorted by the order of the input file paths.
            The file name is the original file name with a timestamp appended. E.g. "document.pdf" -> "document_20250313_123456.json".
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func = partial(
        parse_and_save_document,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
    )
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),  # type: ignore [arg-type]
                total=len(documents),
                desc="Parsing documents",
            )
        )


def parse_and_save_document(
    document: Union[str, Path, Url],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Union[str, Path, None] = None,
    grounding_save_dir: Union[str, Path, None] = None,
) -> Union[Path, ParsedDocument]:
    """
    Parse a document and save the results to a local directory.

    Args:
        document (str | Path | Url): The document to parse. It can be a local file path, a URL string, or a Pydantic `Url` object.
        result_save_dir (str | Path): The local directory to save the results. If None, the parsed document data is returned.

    Returns:
        Path | ParsedDocument: The file path to the saved result or the parsed document data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if isinstance(document, str) and is_valid_httpurl(document):
            document = Url(document)

        if isinstance(document, Url):
            output_file_path = Path(temp_dir) / Path(str(document)).name
            download_file(document, str(output_file_path))
            document = output_file_path
        else:
            document = Path(document)
            if isinstance(document, Path) and not document.exists():
                raise FileNotFoundError(f"File not found: {document}")

        file_type = get_file_type(document)

        if file_type == "image":
            result = _parse_image(
                document,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
            )
        elif file_type == "pdf":
            result = _parse_pdf(
                document,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = f"{Path(document).stem}_{ts}"
        if grounding_save_dir:
            grounding_save_dir = Path(grounding_save_dir) / result_name
            save_groundings_as_images(
                document, result.chunks, grounding_save_dir, inplace=True
            )
        if not result_save_dir:
            return result

        result_save_dir = Path(result_save_dir)
        result_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = result_save_dir / f"{result_name}.json"
        save_path.write_text(result.model_dump_json())
        _LOGGER.info(f"Saved the parsed result to '{save_path}'")

        return save_path


def _parse_pdf(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    with tempfile.TemporaryDirectory() as temp_dir:
        parts = split_pdf(file_path, temp_dir, settings.split_size)
        file_path = Path(file_path)
        part_results = _parse_doc_in_parallel(
            parts,
            doc_name=file_path.name,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
        return _merge_part_results(part_results)


def _parse_image(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    try:
        result_raw = _send_parsing_request(
            str(file_path),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
        result_raw = {
            **result_raw["data"],
            "errors": result_raw.get("errors", []),
            "doc_type": "image",
            "start_page_idx": 0,
            "end_page_idx": 0,
        }
        return ParsedDocument.model_validate(result_raw)
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing image '{file_path}' due to: {error_msg}")
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=0,
            end_page_idx=0,
            doc_type="image",
            errors=[PageError(page_num=0, error=error_msg, error_code=-1)],
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
    curr.errors.extend(next.errors)


def _parse_doc_in_parallel(
    doc_parts: list[Document],
    *,
    doc_name: str,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> list[ParsedDocument]:
    _parse_func = partial(
        _parse_doc_parts,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
    )
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, doc_parts),
                total=len(doc_parts),
                desc=f"Parsing document parts from '{doc_name}'",
            )
        )


def _parse_doc_parts(
    doc: Document,
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    try:
        _LOGGER.info(f"Start parsing document part: '{doc}'")
        result = _send_parsing_request(
            str(doc.file_path),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
        _LOGGER.info(f"Successfully parsed document part: '{doc}'")
        return ParsedDocument.model_validate(
            {
                **result["data"],
                "errors": result.get("errors", []),
                "start_page_idx": doc.start_page_idx,
                "end_page_idx": doc.end_page_idx,
                "doc_type": "pdf",
            }
        )
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing document '{doc}' due to: {error_msg}")
        errors = [
            PageError(page_num=i, error=error_msg, error_code=-1)
            for i in range(doc.start_page_idx, doc.end_page_idx + 1)
        ]
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=doc.start_page_idx,
            end_page_idx=doc.end_page_idx,
            doc_type="pdf",
            errors=errors,
        )


@tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        exp_base=1.5, initial=1, max=settings.max_retry_wait_time, jitter=10
    ),
    stop=tenacity.stop_after_attempt(settings.max_retries),
    retry=tenacity.retry_if_exception_type(RetryableError),
    after=log_retry_failure,
)
def _send_parsing_request(
    file_path: str,
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> dict[str, Any]:
    """
    Send a parsing request to the Landing AI Agentic Document Analysis API.

    Args:
        file_path (str): The path to the document file.
        include_marginalia (bool, optional): Whether to include marginalia in the analysis. Defaults to True.
        include_metadata_in_markdown (bool, optional): Whether to include metadata in the markdown output. Defaults to True.

    Returns:
        dict[str, Any]: The parsed document data.
    """
    with Timer() as timer:
        file_type = "pdf" if Path(file_path).suffix.lower() == ".pdf" else "image"
        # TODO: check if the file extension is a supported image type
        with open(file_path, "rb") as file:
            files = {file_type: file}
            data = {
                "include_marginalia": include_marginalia,
                "include_metadata_in_markdown": include_metadata_in_markdown,
            }
            headers = {
                "Authorization": f"Basic {settings.vision_agent_api_key}",
                "runtime_tag": f"agentic-doc-v{_LIB_VERSION}",
            }
            response = httpx.post(
                _ENDPOINT_URL,
                files=files,
                data=data,
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
