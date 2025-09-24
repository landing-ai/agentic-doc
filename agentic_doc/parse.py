import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

import structlog
from pydantic_core import Url
from tqdm import tqdm

from agentic_doc.common import (
    ParsedDocument,
    T,
    dump_parsed_doc_json,
    FigureCaptioningType,
)
from agentic_doc.config import Settings, get_settings, ParseConfig
from agentic_doc.parse_utils import (
    _get_document_paths,
    _convert_to_parsed_documents,
    _parse_pdf,
    _parse_image,
    _merge_part_results,
    _merge_next_part,
    _parse_doc_in_parallel,
    _parse_doc_parts,
    _send_parsing_request,
    _extract_fields_schema,
)
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
    config: ParseConfig = ParseConfig(),
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
        config: ParseConfig object containing additional configuration options

    Returns:
        List[ParsedDocument]
    """
    settings = get_settings()

    # Apply config overrides if provided
    if config.include_marginalia is not None:
        include_marginalia = config.include_marginalia
    if config.include_metadata_in_markdown is not None:
        include_metadata_in_markdown = config.include_metadata_in_markdown
    if config.extraction_model:
        extraction_model = config.extraction_model
    if config.extraction_schema:
        extraction_schema = config.extraction_schema

        if (
            config.figure_captioning_type == FigureCaptioningType.custom
            and not config.figure_captioning_prompt
        ):
            raise ValueError(
                "figure_captioning_prompt must be provided when figure_captioning_type is 'custom'."
            )

    check_endpoint_and_api_key(
        f"{settings.endpoint_host}/v1/tools/agentic-document-analysis",
        api_key=(
            config.api_key
            if config.api_key
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
    documents_list = list(doc_paths)
    parse_results = parse_documents(
        documents_list,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )

    # Convert results to ParsedDocument objects
    return _convert_to_parsed_documents(
        parse_results, result_save_dir, extraction_model, extraction_schema
    )


def parse_documents(
    documents: list[Union[str, Path, Url]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Union[str, Path, None] = None,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> Union[list[ParsedDocument[T]], list[Path]]:
    """
    Parse a list of documents using the Landing AI Agentic Document Analysis API.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse. Each document can be a local file path, a URL string, or a Pydantic `Url` object.
        result_save_dir (str | Path | None): The local directory to save the results. If None, returns ParsedDocument objects instead.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
        extraction_model (type[BaseModel] | None): Schema for field extraction.
    Returns:
        list[ParsedDocument] | list[Path]: The list of parsed documents or file paths to saved results.
    """
    if config.include_marginalia is not None:
        include_marginalia = config.include_marginalia
    if config.include_metadata_in_markdown is not None:
        include_metadata_in_markdown = config.include_metadata_in_markdown
    if config.extraction_model:
        extraction_model = config.extraction_model
    if config.extraction_schema:
        extraction_schema = config.extraction_schema

        if (
            config.figure_captioning_type == FigureCaptioningType.custom
            and not config.figure_captioning_prompt
        ):
            raise ValueError(
                "figure_captioning_prompt must be provided when figure_captioning_type is 'custom'."
            )

    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func = partial(
        parse_and_save_document,
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


# Backward compatibility alias
def parse_and_save_documents(
    documents: list[Union[str, Path, Url]],
    *,
    result_save_dir: Union[str, Path],
    grounding_save_dir: Union[str, Path, None] = None,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> list[Path]:
    """Backward compatibility alias for parse_documents with result_save_dir."""
    result = parse_documents(
        documents,
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
        extraction_model=extraction_model,
        extraction_schema=extraction_schema,
        config=config,
    )
    # Type assertion for backward compatibility
    assert all(isinstance(r, Path) for r in result)
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
    config: ParseConfig = ParseConfig(),
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
        json_str = dump_parsed_doc_json(result)
        save_path.write_text(json_str, encoding="utf-8")
        _LOGGER.info(f"Saved the parsed result to '{save_path}'")

        return save_path