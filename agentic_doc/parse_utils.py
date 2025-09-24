import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import jsonschema
import structlog
import importlib.metadata
from landingai_ade import LandingAIADE
from agentic_doc.config import ParseConfig, get_settings
from pydantic_core import Url
from agentic_doc.common import ParsedDocument, SplitType, T, FigureCaptioningType, Timer

if TYPE_CHECKING:
    from agentic_doc.common import create_metadata_model

_LOGGER = structlog.get_logger(__name__)
_LIB_VERSION = importlib.metadata.version("agentic-doc")

# Global client cache
_global_client: Optional[LandingAIADE] = None


def _get_client(config: ParseConfig = ParseConfig()) -> LandingAIADE:
    """Get the global LandingAIADE client, creating it if needed."""
    global _global_client
    if _global_client is None:
        settings = get_settings()
        api_key = config.api_key if config.api_key else settings.vision_agent_api_key

        # Auto-detect environment based on endpoint_host
        if config.environment:
            environment = config.environment
        else:
            endpoint_host = settings.endpoint_host
            if "eu-west-1" in endpoint_host or "eu." in endpoint_host:
                environment = "eu"
            else:
                environment = "production"

        _global_client = LandingAIADE(apikey=api_key, environment=environment)
    return _global_client


def _process_extraction_data(
    result_raw: dict[str, Any],
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
) -> tuple[Any, Any]:
    """Process extraction validation and return extraction and extraction_metadata."""
    from agentic_doc.common import create_metadata_model

    extraction = None
    extraction_metadata = None

    extracted_schema = result_raw.get("extracted_schema")
    if extraction_model and extracted_schema:
        extraction = extraction_model.model_validate(extracted_schema)
    elif extraction_schema and extracted_schema:
        jsonschema.validate(instance=extracted_schema, schema=extraction_schema)
        extraction = extracted_schema

    raw_extraction_metadata = result_raw.get("extraction_metadata")
    if extraction_model and raw_extraction_metadata:
        metadata_model = create_metadata_model(extraction_model)
        extraction_metadata = metadata_model.model_validate(raw_extraction_metadata)

    return extraction, extraction_metadata


def _send_parsing_request(
    file_path: str,
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
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
    client = _get_client(config)

    # Prepare request data
    data: dict[str, Any] = {
        "include_marginalia": include_marginalia,
        "include_metadata_in_markdown": include_metadata_in_markdown,
        "figure_captioning_type": FigureCaptioningType.verbose.value,
        "split": config.split.value,
    }

    # Include figure captioning and split parameters from config
    if config.figure_captioning_type:
        data["figure_captioning_type"] = config.figure_captioning_type.value
    if config.figure_captioning_prompt:
        data["figure_captioning_prompt"] = config.figure_captioning_prompt
    if config.enable_rotation_detection is not None:
        data["enable_rotation_detection"] = config.enable_rotation_detection

    fields_schema = _extract_fields_schema(extraction_model, extraction_schema)
    if fields_schema is not None:
        data["fields_schema"] = fields_schema

    # Time only the actual API call
    with Timer() as timer:
        with open(file_path, "rb") as file:
            try:
                response = client.parse(
                    document=file,
                    split="page" if data.get("split") == "page" else None,
                    extra_headers={"runtime_tag": f"agentic-doc-v{_LIB_VERSION}"},
                    extra_body=data,
                    timeout=config.timeout,
                )
            except Exception as e:
                # TODO: look at landingai-ade parse and see what exceptions it raises and handle accordingly
                # Check for retryable errors based on status code if available
                _LOGGER.info(
                    f"LandingAI ade-python API call failed with error: {e}. "
                )
                raise e

    _LOGGER.info(
        f"Time taken to successfully parse a document chunk: {timer.elapsed:.2f} seconds"
    )

    return response.model_dump()


def _parse_image(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> ParsedDocument[T]:
    try:
        response = _send_parsing_request(
            str(file_path),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
        result = {
            **response["data"],
            "errors": response.get("errors", []),
            "extraction_error": response.get("extraction_error", None),
            "doc_type": "image",
            "start_page_idx": 0,
            "end_page_idx": 0,
            "metadata": response.get("metadata"),
        }

        # Handle split for images - images are always single page
        if config.split == SplitType.page and isinstance(result["markdown"], str):
            result["markdown"] = [result["markdown"]]

        # Handle extraction validation and assignment
        extraction, extraction_metadata = _process_extraction_data(
            result, extraction_model, extraction_schema
        )
        if extraction:
            result["extraction"] = extraction
        if extraction_metadata:
            result["extraction_metadata"] = extraction_metadata

        if extraction_schema:
            return ParsedDocument[Any].model_validate(result)
        else:
            return ParsedDocument.model_validate(result)
    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(f"Error parsing image '{file_path}' due to: {error_msg}")
        empty_markdown = [] if config.split == SplitType.page else ""
        return ParsedDocument(
            markdown=empty_markdown,
            chunks=[],
            extraction_metadata=None,
            extraction=None,
            errors=[error_msg],
            extraction_error=error_msg,
            doc_type="image",
            start_page_idx=0,
            end_page_idx=0,
            metadata={},
        )


def _parse_doc_parts(
    doc_parts_tasks: list[dict[str, Any]],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> ParsedDocument[T]:
    try:
        response = _send_parsing_request(
            json.dumps(doc_parts_tasks),
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            extraction_model=extraction_model,
            extraction_schema=extraction_schema,
            config=config,
        )
        result = {
            **response["data"],
            "errors": response.get("errors", []),
            "extraction_error": response.get("extraction_error", None),
            "metadata": response.get("metadata"),
        }

        # Handle split for documents
        if config.split == SplitType.page and isinstance(result["markdown"], str):
            result["markdown"] = [result["markdown"]]

        # Handle extraction validation and assignment
        extraction, extraction_metadata = _process_extraction_data(
            result, extraction_model, extraction_schema
        )
        if extraction:
            result["extraction"] = extraction
        if extraction_metadata:
            result["extraction_metadata"] = extraction_metadata

        # Return appropriate ParsedDocument type
        if extraction_schema:
            return ParsedDocument[Any].model_validate(result)
        else:
            return ParsedDocument.model_validate(result)

    except Exception as e:
        error_msg = str(e)
        _LOGGER.error(
            f"Error parsing document parts due to: {error_msg}"
        )
        empty_markdown = [] if config.split == SplitType.page else ""
        return ParsedDocument(
            markdown=empty_markdown,
            chunks=[],
            extraction_metadata=None,
            extraction=None,
            errors=[error_msg],
            extraction_error=error_msg,
            doc_type="unknown",
            start_page_idx=0,
            end_page_idx=0,
            metadata={},
        )


def _get_document_paths(
    documents: Union[
        bytes,
        str,
        Path,
        Url,
        list[Union[str, Path, Url]],
        Any,  # BaseConnector
        Any,  # ConnectorConfig
    ],
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
) -> list[Union[str, Path, Url]]:
    """Convert various input types to a list of document paths."""
    # Import here to avoid circular imports
    from agentic_doc.connectors import BaseConnector, ConnectorConfig

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
    connector_or_config: Any,  # Union[BaseConnector, ConnectorConfig]
    connector_path: Optional[str],
    connector_pattern: Optional[str],
) -> list[Path]:
    """Download files from connector and return local paths."""
    from agentic_doc.connectors import BaseConnector, create_connector

    connector = (
        connector_or_config
        if isinstance(connector_or_config, BaseConnector)
        else create_connector(connector_or_config)
    )

    file_list = connector.list_files(connector_path, connector_pattern)

    local_paths = []
    for file_id in file_list:
        try:
            local_path = connector.download_file(file_id)
            local_paths.append(local_path)
        except Exception as e:
            _LOGGER.error(f"Failed to download file {file_id}: {e}")

    return local_paths


def _get_documents_from_bytes(doc_bytes: bytes) -> list[Path]:
    """Save raw bytes to a temporary file and return its path."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(doc_bytes)
        temp_file_path = Path(temp_file.name)
    return [temp_file_path]


def _convert_to_parsed_documents(
    parse_results: Union[list[ParsedDocument[T]], list[Path]],
    result_save_dir: Optional[Union[str, Path]],
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
) -> list[ParsedDocument[T]]:
    """Convert parse results to ParsedDocument objects."""
    from agentic_doc.common import create_metadata_model

    parsed_docs = []

    for result in parse_results:
        if isinstance(result, ParsedDocument):
            parsed_docs.append(result)
        elif isinstance(result, Path):
            with open(result, encoding="utf-8") as f:
                data = json.load(f)

            if extraction_model and "extraction" in data:
                data["extraction"] = extraction_model.model_validate(data["extraction"])
            if extraction_schema and "extracted_schema" in data:
                jsonschema.validate(
                    instance=data["extracted_schema"],
                    schema=extraction_schema,
                )
                data["extraction"] = data["extracted_schema"]
            if extraction_model and "extraction_metadata" in data:
                metadata_model = create_metadata_model(extraction_model)
                data["extraction_metadata"] = metadata_model.model_validate(
                    data["extraction_metadata"]
                )
            if extraction_schema:
                parsed_doc: ParsedDocument[Any] = ParsedDocument[Any].model_validate(
                    data
                )
            else:
                parsed_doc = ParsedDocument.model_validate(data)
            if result_save_dir:
                parsed_doc.result_path = result
            parsed_docs.append(parsed_doc)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    return parsed_docs


def _parse_pdf(
    file_path: Union[str, Path],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> ParsedDocument[T]:
    """Parse a PDF file."""
    import tempfile
    from pypdf import PdfReader
    from agentic_doc.utils import split_pdf

    settings = get_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        if extraction_model or extraction_schema is not None:
            total_pages = 0
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
            split_size = (
                config.extraction_split_size
                if config.extraction_split_size
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
                if config.split_size
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
        split_type = config.split
        return _merge_part_results(part_results, split_type)


def _merge_part_results(
    results: list[ParsedDocument[T]], split: SplitType = SplitType.full
) -> ParsedDocument[T]:
    """Merge multiple ParsedDocument results into one."""
    import copy

    if not results:
        _LOGGER.warning(
            f"No results to merge: {results}, returning empty ParsedDocument"
        )
        empty_markdown = [] if split == SplitType.page else ""
        return ParsedDocument(
            markdown=empty_markdown,
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
        _merge_next_part(init_result, results[i], split)

    return init_result


def _merge_next_part(
    curr: ParsedDocument[T],
    next: ParsedDocument[T],
    split: SplitType = SplitType.full,
) -> None:
    """Merge next ParsedDocument part into current one."""
    if split == SplitType.page:
        # When split is page, both curr.markdown and next.markdown should be lists
        if isinstance(curr.markdown, list) and isinstance(next.markdown, list):
            curr.markdown.extend(next.markdown)
        elif isinstance(curr.markdown, str) and isinstance(next.markdown, str):
            # Convert to list if they're strings (shouldn't happen but handle gracefully)
            curr.markdown = [curr.markdown] + [next.markdown]
        elif isinstance(curr.markdown, list) and isinstance(next.markdown, str):
            curr.markdown.append(next.markdown)
        elif isinstance(curr.markdown, str) and isinstance(next.markdown, list):
            curr.markdown = [curr.markdown] + next.markdown
    else:
        # When split is full, join with newlines
        if isinstance(curr.markdown, str) and isinstance(next.markdown, str):
            curr.markdown += "\n\n" + next.markdown
        elif isinstance(curr.markdown, list) and isinstance(next.markdown, list):
            # Join lists and convert to string
            curr.markdown = "\n\n".join(curr.markdown + next.markdown)
        elif isinstance(curr.markdown, str) and isinstance(next.markdown, list):
            curr.markdown = curr.markdown + "\n\n" + "\n\n".join(next.markdown)
        elif isinstance(curr.markdown, list) and isinstance(next.markdown, str):
            curr.markdown = "\n\n".join(curr.markdown) + "\n\n" + next.markdown

    next_chunks = next.chunks
    for chunk in next_chunks:
        for grounding in chunk.grounding:
            grounding.page += next.start_page_idx

    curr.chunks.extend(next_chunks)
    curr.end_page_idx = next.end_page_idx
    curr.errors.extend(next.errors)


def _parse_doc_in_parallel(
    doc_parts: list[Any],  # list[Document]
    *,
    doc_name: str,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    extraction_model: Optional[type[T]] = None,
    extraction_schema: Optional[dict[str, Any]] = None,
    config: ParseConfig = ParseConfig(),
) -> list[ParsedDocument[T]]:
    """Parse document parts in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from tqdm import tqdm

    _parse_func = partial(
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


def _extract_fields_schema(extraction_model: Optional[type[T]], extraction_schema: Optional[dict[str, Any]]) -> Optional[str]:
    def resolve_refs(obj: Any, defs: Dict[str, Any]) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve_refs(copy.deepcopy(defs[ref_name]), defs)
            return {k: resolve_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item, defs) for item in obj]
        return obj

    if extraction_model:
        schema = extraction_model.model_json_schema()
        defs = schema.pop("$defs", {})
        schema = resolve_refs(schema, defs)
        return json.dumps(schema)
    elif extraction_schema:
        return json.dumps(extraction_schema)
    return None 