import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union

import httpx
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    title = "title"  # type: ignore [assignment]
    page_header = "page_header"
    page_footer = "page_footer"
    page_number = "page_number"
    key_value = "key_value"
    form = "form"
    table = "table"
    figure = "figure"
    text = "text"
    error = "error"


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
    # NOTE: could be None if error happens in parsing the chunk
    box: Union[ChunkGroundingBox, None]
    # NOTE: image_path doesn't come from the server API, so it's null by default
    image_path: Union[Path, None] = None


class Chunk(BaseModel):
    text: str
    grounding: list[ChunkGrounding]
    chunk_type: ChunkType
    chunk_id: Union[str, None]

    @staticmethod
    def error_chunk(error_msg: str, page_idx: int) -> "Chunk":
        return Chunk(
            text=error_msg,
            grounding=[ChunkGrounding(page=page_idx, box=None)],
            chunk_type=ChunkType.error,
            chunk_id=None,
        )


class ParsedDocument(BaseModel):
    markdown: str
    chunks: list[Chunk]
    start_page_idx: int
    end_page_idx: int
    doc_type: Literal["pdf", "image"]


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
