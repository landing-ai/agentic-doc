import copy
import importlib.metadata
import json
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx
import structlog
import tenacity
from pydantic_core import Url
from tqdm import tqdm
import jsonschema
from pypdf import PdfReader

from agentic_doc.common import (
    Document,
    PageError,
    ParsedDocument,
    RetryableError,
    T,
    Timer,
    create_metadata_model,
)
from agentic_doc.config import Settings, get_settings, ParseConfig
from agentic_doc.connectors import BaseConnector, ConnectorConfig, create_connector
from agentic_doc.utils import (
    check_endpoint_and_api_key,
    download_file,
    get_file_type,
    is_valid_httpurl,
    log_retry_failure,
    save_groundings_as_images,
    split_pdf,
)

_LOGGER = structlog.getLogger(__name__)
_LIB_VERSION = importlib.metadata.version("agentic-doc")


def _get_endpoint_url(settings: Settings) -> str:
    return f"{settings.endpoint_host}/v1/tools/agentic-document-analysis"


def parse(
    documents: Union[
        bytes,
        str,
        Path,
        Url,
        List[Union[str, Path, Url]],
        BaseConnector,
        ConnectorConfig,
    ],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> List[ParsedDocument[T]]:
    """
    Universal parse function that can handle single documents, lists of documents,
    or documents from various connectors.

    Args:
        documents: Can be:
            - Single document path/URL (str, Path, Url)
            - List of document paths/URLs
            - Connector instance (BaseConnector)
            - Connector configuration (ConnectorConfig)
            - Raw bytes of a document (either PDF or Image bytes)
        include_marginalia: Whether to include marginalia in the analysis
        include_metadata_in_markdown: Whether to include metadata in markdown output
        result_save_dir: Directory to save results
        grounding_save_dir: Directory to save grounding images
        connector_path: Path for connector to search (when using connectors)
        connector_pattern: Pattern to filter files (when using connectors)
        extraction_model: Pydantic model schema for field extraction (optional)
        extraction_schema: JSON schema for field extraction (optional)

    Returns:
        List[ParsedDocument]
    """
    settings = get_settings()
    if config and config.include_marginalia:
        include_marginalia = config.include_marginalia
    if config and config.include_metadata_in_markdown:
        include_metadata_in_markdown = config.include_metadata_in_markdown
    if config and config.extraction_model:
        extraction_model = config.extraction_model
    if config and config.extraction_schema:
        extraction_schema = config.extraction_schema

    check_endpoint_and_api_key(
        _get_endpoint_url(settings),
        api_key=(
            config.api_key
            if config and config.api_key
            else settings.vision_agent_api_key
        ),
    )

    # Convert input to list of document paths
    doc_paths = _get_document_paths(documents, connector_path, connector_pattern)

    if not doc_paths:
        _LOGGER.warning("No documents to parse")
        return []

    if extraction_schema and extraction_model:
        raise ValueError(
            "extraction_model and extraction_schema cannot be used together, you must provide only one of them"
        )

    # Parse all documents
    parse_results = _parse_document_list(
        doc_paths,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )

    # Convert results to ParsedDocument objects
    return _convert_to_parsed_documents(parse_results, result_save_dir)


def _get_document_paths(
    documents: Union[
        bytes,
        str,
        Path,
        Url,
        List[Union[str, Path, Url]],
        BaseConnector,
        ConnectorConfig,
    ],
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
) -> Sequence[Union[str, Path, Url]]:
    """Convert various input types to a list of document paths."""
    if isinstance(documents, (BaseConnector, ConnectorConfig)):
        return _get_paths_from_connector(documents, connector_path, connector_pattern)
    elif isinstance(documents, (str, Path, Url)):
        return [documents]
    elif isinstance(documents, list):
        return documents
    elif isinstance(documents, bytes):
        return _get_documents_from_bytes(documents)
    else:
        raise ValueError(f"Unsupported documents type: {type(documents)}")


def _get_paths_from_connector(
    connector_or_config: Union[BaseConnector, ConnectorConfig],
    connector_path: Optional[str],
    connector_pattern: Optional[str],
) -> List[Path]:
    """Download files from connector and return local paths."""
    connector = (
        connector_or_config
        if isinstance(connector_or_config, BaseConnector)
        else create_connector(connector_or_config)
    )

    file_list = connector.list_files(connector_path, connector_pattern)
    if not file_list:
        return []

    local_paths = []
    for file_id in file_list:
        try:
            local_path = connector.download_file(file_id)
            local_paths.append(local_path)
        except Exception as e:
            _LOGGER.error(f"Failed to download file {file_id}: {e}")

    return local_paths


def _get_documents_from_bytes(doc_bytes: bytes) -> List[Path]:
    """Save raw bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(doc_bytes)
        temp_file_path = Path(temp_file.name)
    return [temp_file_path]


def _convert_to_parsed_documents(
    parse_results: Union[List[ParsedDocument[T]], List[Path]],
    result_save_dir: Optional[Union[str, Path]],
) -> List[ParsedDocument[T]]:
    """Convert parse results to ParsedDocument objects."""
    parsed_docs = []

    for result in parse_results:
        if isinstance(result, ParsedDocument):
            parsed_docs.append(result)
        elif isinstance(result, Path):
            with open(result, encoding="utf-8") as f:
                data = json.load(f)
            parsed_doc: ParsedDocument[T] = ParsedDocument.model_validate(data)
            if result_save_dir:
                parsed_doc.result_path = result
            parsed_docs.append(parsed_doc)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    return parsed_docs


def _parse_document_list(
    documents: Sequence[Union[str, Path, Url]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> Union[List[ParsedDocument[T]], List[Path]]:
    """Helper function to parse a list of documents."""
    documents_list = list(documents)
    if result_save_dir:
        return parse_and_save_documents(
            documents_list,
            result_save_dir=result_save_dir,
            grounding_save_dir=grounding_save_dir,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
    else:
        return parse_documents(
            documents_list,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            grounding_save_dir=grounding_save_dir,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )


def parse_documents(
    documents: list[Union[str, Path, Url]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    grounding_save_dir: Union[str, Path, None] = None,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> list[ParsedDocument[T]]:
    """
    Parse a list of documents using the Landing AI Agentic Document Analysis API.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse. Each document can be a local file path, a URL string, or a Pydantic `Url` object.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
        extraction_model (type[BaseModel] | None): Schema for field extraction.
    Returns:
        list[ParsedDocument]: The list of parsed documents. The list is sorted by the order of the input documents.
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func: Callable[[Union[str, Path, Url]], ParsedDocument[T]] = partial(
        _parse_document_without_save,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    with ThreadPoolExecutor(max_workers=get_settings().batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),
                total=len(documents),
                desc="Parsing documents",
            )
        )


def _parse_document_without_save(
    document: Union[str, Path, Url],
    include_marginalia: bool,
    include_metadata_in_markdown: bool,
    grounding_save_dir: Union[str, Path, None],
    extraction_model: Optional[type[T]],
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> ParsedDocument[T]:
    """Wrapper to ensure parse_and_save_document returns ParsedDocument when no save dir."""
    result = parse_and_save_document(
        document,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=None,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    # When result_save_dir is None, parse_and_save_document returns ParsedDocument[T]
    assert isinstance(result, ParsedDocument)
    return result


def parse_and_save_documents(
    documents: list[Union[str, Path, Url]],
    *,
    result_save_dir: Union[str, Path],
    grounding_save_dir: Union[str, Path, None] = None,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> list[Path]:
    """
    Parse a list of documents and save the results to a local directory.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse. Each document can be a local file path, a URL string, or a Pydantic `Url` object.
        result_save_dir (str | Path): The local directory to save the results.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
        extraction_model (type[BaseModel] | None): Schema for field extraction.
    Returns:
        list[Path]: A list of json file paths to the saved results. The file paths are sorted by the order of the input file paths.
            The file name is the original file name with a timestamp appended. E.g. "document.pdf" -> "document_20250313_123456.json".
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")

    _parse_func: Callable[[Union[str, Path, Url]], Path] = partial(
        _parse_document_with_save,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    with ThreadPoolExecutor(max_workers=get_settings().batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),
                total=len(documents),
                desc="Parsing documents",
            )
        )


def _parse_document_with_save(
    document: Union[str, Path, Url],
    include_marginalia: bool,
    include_metadata_in_markdown: bool,
    result_save_dir: Union[str, Path],
    grounding_save_dir: Union[str, Path, None],
    extraction_model: Optional[type[T]],
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> Path:
    """Wrapper to ensure parse_and_save_document returns Path when save dir provided."""
    result = parse_and_save_document(
        document,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    # When result_save_dir is provided, parse_and_save_document returns Path
    assert isinstance(result, Path)
    return result


def parse_and_save_document(
    document: Union[str, Path, Url],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Union[str, Path, None] = None,
    grounding_save_dir: Union[str, Path, None] = None,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> Union[Path, ParsedDocument[T]]:
    """
    Parse a document and save the results to a local directory.

    Args:
        document (str | Path | Url): The document to parse. It can be a local file path, a URL string, or a Pydantic `Url` object.
        result_save_dir (str | Path): The local directory to save the results. If None, the parsed document data is returned.
        extraction_model (type[BaseModel] | None): Schema for field extraction.
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
                extraction_model=extraction_model,
                extraction_schema=extraction_schema,
                config=config,
            )
        elif file_type == "pdf":
            result = _parse_pdf(
                document,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
                extraction_model=extraction_model,
                extraction_schema=extraction_schema,
                config=config,
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
        save_path.write_text(result.model_dump_json(), encoding="utf-8")
        _LOGGER.info(f"Saved the parsed result to '{save_path}'")

        return save_path


def _parse_pdf(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> ParsedDocument[T]:
    settings = get_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        if extraction_model or extraction_schema is not None:
            total_pages = 0
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
            split_size = (
                config.extraction_split_size
                if config and config.extraction_split_size
                else settings.extraction_split_size
            )
            if total_pages > split_size:
                raise ValueError(
                    f"Document has {total_pages} pages, which exceeds the maximum of {settings.extraction_split_size} pages "
                    "allowed when using field extraction. "
                    f"Please use a document with {split_size} pages or fewer."
                )
        else:
            split_size = (
                config.split_size
                if config and config.split_size
                else settings.split_size
            )

        parts = split_pdf(file_path, temp_dir, split_size)
        file_path = Path(file_path)
        part_results = _parse_doc_in_parallel(
            parts,
            doc_name=file_path.name,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
        return _merge_part_results(part_results)


def _parse_image(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> ParsedDocument[T]:
    try:
        result_raw = _send_parsing_request(
            str(file_path),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
        result_raw = {
            **result_raw["data"],
            "errors": result_raw.get("errors", []),
            "extraction_error": result_raw.get("extraction_error", None),
            "doc_type": "image",
            "start_page_idx": 0,
            "end_page_idx": 0,
        }

        # Handle extraction validation and assignment
        if (
            extraction_model
            and "extracted_schema" in result_raw
            and result_raw["extracted_schema"]
        ):
            result_raw["extraction"] = extraction_model.model_validate(
                result_raw["extracted_schema"]
            )
        elif (
            extraction_schema
            and "extracted_schema" in result_raw
            and result_raw["extracted_schema"]
        ):
            jsonschema.validate(
                instance=result_raw["extracted_schema"],
                schema=extraction_schema,
            )
            result_raw["extraction"] = result_raw["extracted_schema"]

        if (
            extraction_model
            and "extraction_metadata" in result_raw
            and result_raw["extraction_metadata"]
        ):
            metadata_model = create_metadata_model(extraction_model)
            result_raw["extraction_metadata"] = metadata_model.model_validate(
                result_raw["extraction_metadata"]
            )

        if extraction_schema:
            return ParsedDocument[Any].model_validate(result_raw)
        else:
            return ParsedDocument.model_validate(result_raw)
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing image '{file_path}' due to: {error_msg}")
        return ParsedDocument(
            markdown="",
            chunks=[],
            extraction_metadata=None,
            extraction=None,
            start_page_idx=0,
            end_page_idx=0,
            doc_type="image",
            result_path=None,
            errors=[PageError(page_num=0, error=error_msg, error_code=-1)],
        )


def _merge_part_results(results: list[ParsedDocument[T]]) -> ParsedDocument[T]:
    if not results:
        _LOGGER.warning(
            f"No results to merge: {results}, returning empty ParsedDocument"
        )
        return ParsedDocument(
            markdown="",
            chunks=[],
            extraction_metadata=None,
            extraction=None,
            start_page_idx=0,
            end_page_idx=0,
            doc_type="pdf",
            result_path=None,
        )

    init_result = copy.deepcopy(results[0])
    for i in range(1, len(results)):
        _merge_next_part(init_result, results[i])

    return init_result


def _merge_next_part(curr: ParsedDocument[T], next: ParsedDocument[T]) -> None:
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
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> list[ParsedDocument[T]]:
    _parse_func: Callable[[Document], ParsedDocument[T]] = partial(
        _parse_doc_parts,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    with ThreadPoolExecutor(max_workers=get_settings().max_workers) as executor:
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
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> ParsedDocument[T]:
    try:
        _LOGGER.info(f"Start parsing document part: '{doc}'")
        result = _send_parsing_request(
            str(doc.file_path),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
        _LOGGER.info(f"Successfully parsed document part: '{doc}'")
        result_data = {
            **result["data"],
            "errors": result.get("errors", []),
            "extraction_error": result.get("extraction_error", None),
            "start_page_idx": doc.start_page_idx,
            "end_page_idx": doc.end_page_idx,
            "doc_type": "pdf",
        }

        if (
            extraction_model
            and "extracted_schema" in result_data
            and result_data["extracted_schema"]
        ):
            result_data["extraction"] = extraction_model.model_validate(
                result_data["extracted_schema"]
            )
        elif (
            extraction_schema
            and "extracted_schema" in result_data
            and result_data["extracted_schema"]
        ):
            jsonschema.validate(
                instance=result_data["extracted_schema"],
                schema=extraction_schema,
            )
            result_data["extraction"] = result_data["extracted_schema"]

        if (
            extraction_model
            and "extraction_metadata" in result_data
            and result_data["extraction_metadata"]
        ):
            metadata_model = create_metadata_model(extraction_model)
            result_data["extraction_metadata"] = metadata_model.model_validate(
                result_data["extraction_metadata"]
            )

        if extraction_schema:
            return ParsedDocument[Any].model_validate(result_data)
        else:
            return ParsedDocument.model_validate(result_data)
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
            extraction_metadata=None,
            extraction=None,
            start_page_idx=doc.start_page_idx,
            end_page_idx=doc.end_page_idx,
            doc_type="pdf",
            result_path=Path(doc.file_path),
            errors=errors,
        )


# TODO: read retry settings at runtime (not at import time)
@tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        exp_base=1.5, initial=1, max=get_settings().max_retry_wait_time, jitter=10
    ),
    stop=tenacity.stop_after_attempt(get_settings().max_retries),
    retry=tenacity.retry_if_exception_type(RetryableError),
    after=log_retry_failure,
)
def _send_parsing_request(
    file_path: str,
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: Optional[ParseConfig] = None,
) -> dict[str, Any]:
    """
    Send a parsing request to the Landing AI Agentic Document Analysis API.

    Args:
        file_path (str): The path to the document file.
        include_marginalia (bool, optional): Whether to include marginalia in the analysis. Defaults to True.
        include_metadata_in_markdown (bool, optional): Whether to include metadata in the markdown output. Defaults to True.
        extraction_model (type[BaseModel] | None): Schema for field extraction. If provided, ensures the response matches this schema.

    Returns:
        dict[str, Any]: The parsed document data.
    """
    settings = get_settings()
    with Timer() as timer:
        file_type = "pdf" if Path(file_path).suffix.lower() == ".pdf" else "image"
        # TODO: check if the file extension is a supported image type
        with open(file_path, "rb") as file:
            files = {file_type: file}
            data: dict[str, Any] = {
                "include_marginalia": include_marginalia,
                "include_metadata_in_markdown": include_metadata_in_markdown,
            }

            def resolve_refs(obj: Any, defs: Dict[str, Any]) -> Any:
                if isinstance(obj, dict):
                    if "$ref" in obj:
                        ref_name = obj["$ref"].split("/")[-1]
                        return resolve_refs(copy.deepcopy(defs[ref_name]), defs)
                    return {k: resolve_refs(v, defs) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [resolve_refs(item, defs) for item in obj]
                return obj

            if extraction_model is not None:
                schema = extraction_model.model_json_schema()
                defs = schema.pop("$defs", {})
                schema = resolve_refs(schema, defs)
                data["fields_schema"] = json.dumps(schema)
            elif extraction_schema is not None:
                data["fields_schema"] = json.dumps(extraction_schema)

            api_key = (
                config.api_key
                if config and config.api_key
                else settings.vision_agent_api_key
            )
            headers = {
                "Authorization": f"Basic {api_key}",
                "runtime_tag": f"agentic-doc-v{_LIB_VERSION}",
            }

            response = httpx.post(
                _get_endpoint_url(settings),
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
    result: dict[str, Any] = response.json()

    return result
