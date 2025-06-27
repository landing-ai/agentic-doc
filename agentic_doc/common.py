import inspect
import time
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import httpx
from pydantic import BaseModel, Field, create_model


class ChunkType(str, Enum):
    table = "table"
    figure = "figure"
    text = "text"
    marginalia = "marginalia"


class ChunkGroundingBox(BaseModel):
    """
    A bounding box of a chunk.

    The coordinates are in the format of [left, top, right, bottom].
    """

    l: float  # noqa: E741
    t: float
    r: float
    b: float


class ChunkGrounding(BaseModel):
    page: int
    box: ChunkGroundingBox
    # NOTE: image_path doesn't come from the server API, so it's null by default
    image_path: Union[Path, None] = None


class Chunk(BaseModel):
    text: str
    grounding: list[ChunkGrounding]
    chunk_type: ChunkType
    chunk_id: str


class PageError(BaseModel):
    page_num: int
    error: str
    error_code: int


T = TypeVar("T", bound=BaseModel)
VT = TypeVar("VT")


class MetadataType(BaseModel, Generic[VT]):
    value: Optional[VT] = None
    chunk_references: List[str]
    experimental_confidence: Optional[float] = None


def create_metadata_model(model: type[BaseModel]) -> type[BaseModel]:
    """
    Recursively creates a new Pydantic model from an existing one,
    replacing all leaf-level field types with MetadataType.
    """
    fields: Dict[str, Any] = {}
    for name, field in model.model_fields.items():
        field_type = field.annotation

        origin = get_origin(field_type)

        # Handle Optional/Union types
        if origin is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                if inspect.isclass(non_none_type) and issubclass(
                    non_none_type, BaseModel
                ):
                    metadata_type = create_metadata_model(non_none_type)
                    fields[name] = (Optional[metadata_type], Field(default=None))
                else:
                    fields[name] = (
                        Optional[MetadataType[non_none_type]],  # type: ignore[valid-type]
                        Field(default=None),
                    )
                continue

        # Handle lists
        if origin is list:
            inner_type = get_args(field_type)[0]
            if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
                metadata_inner_type = create_metadata_model(inner_type)
                fields[name] = (
                    List[metadata_inner_type],  # type: ignore[valid-type]
                    Field(default_factory=list),  # type: ignore[arg-type]
                )
            else:
                fields[name] = (
                    List[MetadataType[inner_type]],  # type: ignore[valid-type]
                    Field(default_factory=list),  # type: ignore[arg-type]
                )
            continue

        # Handle nested models
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            fields[name] = (create_metadata_model(field_type), Field())
        else:
            # Replace primitive leaf with MetadataType[original type]
            fields[name] = (
                MetadataType[field_type],  # type: ignore[valid-type]
                Field(),
            )

    return create_model(f"{model.__name__}Metadata", **fields)


class ParsedDocument(BaseModel, Generic[T]):
    markdown: str
    chunks: list[Chunk]
    extraction: Optional[Union[T, Dict[str, Any]]] = None
    extraction_metadata: Optional[Union[Dict[str, Any], BaseModel]] = None
    start_page_idx: int
    end_page_idx: int
    doc_type: Literal["pdf", "image"]
    result_path: Optional[Path] = None
    errors: list[PageError] = Field(default_factory=list)
    extraction_error: Optional[str] = None


class RetryableError(Exception):
    def __init__(self, response: httpx.Response):
        self.response = response
        self.reason = f"{response.status_code} - {response.text}"

    def __str__(self) -> str:
        return self.reason


class Document(BaseModel):
    file_path: Path = Field(description="The local file path to the document file")
    start_page_idx: int = Field(
        description="The index of the first page in the file", ge=0
    )
    end_page_idx: int = Field(
        description="The index of the last page in the file", ge=0
    )

    def __str__(self) -> str:
        return f"File name: {self.file_path.name}\tPage: [{self.start_page_idx}:{self.end_page_idx}]"


class Timer:
    """A context manager for timing code execution in a thread-safe manner."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
