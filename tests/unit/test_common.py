import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    Document,
    DocumentMetadata,
    ParsedDocument,
    RetryableError,
    Timer,
    create_metadata_model,
    MetadataType,
)

from typing import List, Optional
from pydantic import BaseModel


def test_chunk_type_enum():
    # Test all the enumeration values
    assert ChunkType.table == "table"
    assert ChunkType.figure == "figure"
    assert ChunkType.text == "text"
    assert ChunkType.marginalia == "marginalia"


def test_chunk_grounding_box():
    # Test creating a ChunkGroundingBox
    box = ChunkGroundingBox(l=0.1, t=0.2, r=0.8, b=0.9)

    # Check attributes
    assert box.l == 0.1
    assert box.t == 0.2
    assert box.r == 0.8
    assert box.b == 0.9

    # Test serialization/deserialization
    box_dict = box.model_dump()
    box2 = ChunkGroundingBox.model_validate(box_dict)
    assert box2.l == box.l
    assert box2.t == box.t
    assert box2.r == box.r
    assert box2.b == box.b


def test_chunk_grounding():
    # Test creating a ChunkGrounding with box
    box = ChunkGroundingBox(l=0.1, t=0.2, r=0.8, b=0.9)
    grounding = ChunkGrounding(page=0, box=box)

    # Check attributes
    assert grounding.page == 0
    assert grounding.box == box
    assert grounding.image_path is None

    # Test with image_path
    image_path = Path("/path/to/image.png")
    grounding_with_image = ChunkGrounding(page=0, box=box, image_path=image_path)
    assert grounding_with_image.image_path == image_path

    # Note: box field is required in ChunkGrounding, so we can't test with None box

    # Test serialization/deserialization
    grounding_dict = grounding.model_dump()
    grounding2 = ChunkGrounding.model_validate(grounding_dict)
    assert grounding2.page == grounding.page
    assert grounding2.box.l == grounding.box.l
    assert grounding2.image_path == grounding.image_path


def test_chunk():
    # Test creating a Chunk
    box = ChunkGroundingBox(l=0.1, t=0.2, r=0.8, b=0.9)
    grounding = ChunkGrounding(page=0, box=box)
    chunk = Chunk(
        text="Test Text",
        grounding=[grounding],
        chunk_type=ChunkType.text,
        chunk_id="123",
    )

    # Check attributes
    assert chunk.text == "Test Text"
    assert len(chunk.grounding) == 1
    assert chunk.grounding[0] == grounding
    assert chunk.chunk_type == ChunkType.text
    assert chunk.chunk_id == "123"

    # Test creating a Chunk with multiple groundings
    grounding2 = ChunkGrounding(page=1, box=box)
    chunk_multi = Chunk(
        text="Multi Page",
        grounding=[grounding, grounding2],
        chunk_type=ChunkType.text,
        chunk_id="456",
    )
    assert len(chunk_multi.grounding) == 2

    # Note: chunk_id is required in the Chunk model, so we can't test with None

    # Test serialization/deserialization
    chunk_dict = chunk.model_dump()
    chunk2 = Chunk.model_validate(chunk_dict)
    assert chunk2.text == chunk.text
    assert chunk2.chunk_type == chunk.chunk_type
    assert chunk2.chunk_id == chunk.chunk_id
    assert len(chunk2.grounding) == len(chunk.grounding)


def test_page_error():
    # Test creating a PageError
    from agentic_doc.common import PageError

    error_msg = "Test error message"
    page_num = 42
    error_code = -1

    page_error = PageError(page_num=page_num, error=error_msg, error_code=error_code)

    # Check the error
    assert page_error.page_num == page_num
    assert page_error.error == error_msg
    assert page_error.error_code == error_code


def test_parsed_document():
    # Create test chunks
    box = ChunkGroundingBox(l=0.1, t=0.2, r=0.8, b=0.9)
    grounding1 = ChunkGrounding(page=0, box=box)
    grounding2 = ChunkGrounding(page=1, box=box)

    chunk1 = Chunk(
        text="Title", grounding=[grounding1], chunk_type=ChunkType.text, chunk_id="1"
    )

    chunk2 = Chunk(
        text="Content", grounding=[grounding2], chunk_type=ChunkType.text, chunk_id="2"
    )

    # Create ParsedDocument
    doc = ParsedDocument(
        markdown="# Title\n\nContent",
        chunks=[chunk1, chunk2],
        start_page_idx=0,
        end_page_idx=1,
        doc_type="pdf",
    )

    # Check attributes
    assert doc.markdown == "# Title\n\nContent"
    assert len(doc.chunks) == 2
    assert doc.chunks[0] == chunk1
    assert doc.chunks[1] == chunk2
    assert doc.start_page_idx == 0
    assert doc.end_page_idx == 1
    assert doc.doc_type == "pdf"

    # Test metadata field defaults to None
    assert doc.metadata is None

    # Test ParsedDocument with metadata
    metadata = DocumentMetadata(
        parse_id="12345-abcde",
        filename="test.pdf",
        page_count=2,
        processed_at="2023-08-21T19:00:00Z",
        processing_time_ms=1500,
        pages_processed=2,
        user_id="test@example.com"
    )

    doc_with_metadata = ParsedDocument(
        markdown="# Title\n\nContent",
        chunks=[chunk1, chunk2],
        start_page_idx=0,
        end_page_idx=1,
        doc_type="pdf",
        metadata=metadata
    )

    assert doc_with_metadata.metadata is not None
    assert doc_with_metadata.metadata.parse_id == "12345-abcde"
    assert doc_with_metadata.metadata.filename == "test.pdf"
    assert doc_with_metadata.metadata.page_count == 2
    assert doc_with_metadata.metadata.processed_at == "2023-08-21T19:00:00Z"
    assert doc_with_metadata.metadata.processing_time_ms == 1500
    assert doc_with_metadata.metadata.pages_processed == 2
    assert doc_with_metadata.metadata.user_id == "test@example.com"

    # Test with image doc_type
    image_doc = ParsedDocument(
        markdown="Image content",
        chunks=[chunk1],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="image",
    )
    assert image_doc.doc_type == "image"

    # Test serialization/deserialization
    doc_dict = doc.model_dump()
    doc2 = ParsedDocument.model_validate(doc_dict)
    assert doc2.markdown == doc.markdown
    assert len(doc2.chunks) == len(doc.chunks)
    assert doc2.start_page_idx == doc.start_page_idx
    assert doc2.end_page_idx == doc.end_page_idx
    assert doc2.doc_type == doc.doc_type


def test_retryable_error():
    # Create a mock response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    # Create a RetryableError
    error = RetryableError(mock_response)

    # Check the error
    assert error.response == mock_response
    assert error.reason == "429 - Rate limit exceeded"
    assert str(error) == "429 - Rate limit exceeded"


def test_document():
    # Create a Document
    file_path = Path("/path/to/file.pdf")
    doc = Document(file_path=file_path, start_page_idx=0, end_page_idx=5)

    # Check attributes
    assert doc.file_path == file_path
    assert doc.start_page_idx == 0
    assert doc.end_page_idx == 5

    # Test string representation
    assert str(doc) == "File name: file.pdf\tPage: [0:5]"

    # Test validation
    with pytest.raises(ValueError):
        # start_page_idx must be >= 0
        Document(file_path=file_path, start_page_idx=-1, end_page_idx=5)

    with pytest.raises(ValueError):
        # end_page_idx must be >= 0
        Document(file_path=file_path, start_page_idx=0, end_page_idx=-1)


def test_timer():
    # Test the Timer context manager
    timer = Timer()

    # Time should be 0 initially
    assert timer.elapsed == 0.0

    # Test with a short sleep
    with patch("time.perf_counter") as mock_time:
        # Setup mock to simulate time passing
        mock_time.side_effect = [0.0, 1.5]  # start=0.0, end=1.5

        with timer:
            pass  # timer is running

        # After context, elapsed should be updated
        assert timer.elapsed == 1.5

    # Test normal usage
    with timer:
        time.sleep(0.01)  # Small sleep

    # Verify elapsed time is positive
    assert timer.elapsed > 0


def test_create_metadata_model():
    # Simple nested model
    class Researcher(BaseModel):
        age: int
        name: str

    class TopLevelModel(BaseModel):
        id: int
        researcher: Researcher

    MetadataModel = create_metadata_model(TopLevelModel)
    metadata_instance = MetadataModel(
        id={"confidence": 0.5, "chunk_references": ["dummy"], "value": 5},
        researcher={
            "age": {
                "confidence": 0.5,
                "chunk_references": ["dummy", "dummy"],
                "value": 5,
            },
            "name": {
                "confidence": 0.5,
                "chunk_references": ["dummy"],
                "value": "john doe",
            },
        },
    )

    assert isinstance(metadata_instance.id, MetadataType[int])
    assert isinstance(metadata_instance.researcher.age, MetadataType[int])
    assert isinstance(metadata_instance.researcher.name, MetadataType[str])

    # Test with Optional fields
    class ModelWithOptional(BaseModel):
        required_field: str
        optional_field: Optional[str] = None

    MetadataWithOptional = create_metadata_model(ModelWithOptional)

    optional_instance = MetadataWithOptional(
        required_field={
            "confidence": 0.5,
            "chunk_references": ["dummy"],
            "value": "dummy",
        },
        optional_field=None,
    )

    assert isinstance(optional_instance.required_field, MetadataType[str])
    assert optional_instance.optional_field is None

    # Test with list fields
    class ModelWithList(BaseModel):
        items: List[Researcher]

    MetadataWithList = create_metadata_model(ModelWithList)

    list_instance = MetadataWithList(
        items=[
            {
                "age": {"chunk_references": ["dummy"]},
                "name": {"chunk_references": ["dummy"]},
            },
            {
                "age": {"chunk_references": ["dummy"]},
                "name": {"chunk_references": ["dummy"]},
            },
        ]
    )

    assert isinstance(list_instance.items, list)
    assert len(list_instance.items) == 2
    assert isinstance(list_instance.items[0].age, MetadataType[int])

    # Test with list of primitive types
    class ModelWithPrimitiveList(BaseModel):
        tags: List[str]

    MetadataWithPrimitiveList = create_metadata_model(ModelWithPrimitiveList)

    primitive_list_instance = MetadataWithPrimitiveList(
        tags=[{"chunk_references": ["dummy"]}, {"chunk_references": ["dummy"]}]
    )

    assert isinstance(primitive_list_instance.tags, list)
    assert len(primitive_list_instance.tags) == 2
    assert isinstance(primitive_list_instance.tags[0], MetadataType[str])
    assert "chunk_references" in primitive_list_instance.tags[0].__class__.model_fields


def test_document_metadata():
    # Test DocumentMetadata with all fields
    metadata = DocumentMetadata(
        parse_id="12345-abcde-67890",
        filename="test_document.pdf",
        page_count=5,
        processed_at="2023-08-21T19:00:00Z",
        processing_time_ms=2500,
        pages_processed=5,
        user_id="user@example.com"
    )

    assert metadata.parse_id == "12345-abcde-67890"
    assert metadata.filename == "test_document.pdf"
    assert metadata.page_count == 5
    assert metadata.processed_at == "2023-08-21T19:00:00Z"
    assert metadata.processing_time_ms == 2500
    assert metadata.pages_processed == 5
    assert metadata.user_id == "user@example.com"

    # Test DocumentMetadata with default values
    empty_metadata = DocumentMetadata()
    assert empty_metadata.parse_id is None
    assert empty_metadata.filename is None
    assert empty_metadata.page_count is None
    assert empty_metadata.processed_at is None
    assert empty_metadata.processing_time_ms is None
    assert empty_metadata.pages_processed is None
    assert empty_metadata.user_id is None

    # Test serialization/deserialization
    metadata_dict = metadata.model_dump()
    metadata2 = DocumentMetadata.model_validate(metadata_dict)
    assert metadata2.parse_id == metadata.parse_id
    assert metadata2.filename == metadata.filename
    assert metadata2.page_count == metadata.page_count


def test_extraction_metadata_type_validation():
    class NestedModel(BaseModel):
        field1: str
        field2: int

    class ComplexModel(BaseModel):
        simple_field: str
        optional_field: Optional[int] = None
        nested_field: NestedModel
        list_field: List[str]
        nested_list_field: List[NestedModel]

    # Create the metadata model
    MetadataModel = create_metadata_model(ComplexModel)

    metadata_instance = MetadataModel(
        simple_field={
            "confidence": 0.5,
            "chunk_references": ["text"],
            "value": "string",
        },
        optional_field={"confidence": 0.4, "chunk_references": ["low"]},
        nested_field={
            "field1": {"chunk_references": ["table"]},
            "field2": {"chunk_references": ["high"]},
        },
        list_field=[
            {"chunk_references": ["text1"]},
            {"chunk_references": ["text2"]},
        ],  # List of primitive metadata
        nested_list_field=[
            {
                "field1": {"chunk_references": ["page1"]},
                "field2": {"chunk_references": ["medium"]},
            }
        ],
    )

    # Verify types
    assert isinstance(metadata_instance.simple_field, MetadataType[str])
    assert isinstance(metadata_instance.optional_field, MetadataType[int])
    assert hasattr(metadata_instance.nested_field, "field1")
    assert hasattr(metadata_instance.nested_field, "field2")
    assert isinstance(metadata_instance.nested_field.field1, MetadataType[str])
    assert isinstance(metadata_instance.nested_field.field2, MetadataType[int])
    assert isinstance(metadata_instance.list_field, list)
    if len(metadata_instance.list_field) > 0:
        assert isinstance(metadata_instance.list_field[0], MetadataType[str])
    assert isinstance(metadata_instance.nested_list_field, list)

    metadata_with_none = MetadataModel(
        simple_field={"chunk_references": ["text"]},
        optional_field=None,
        nested_field={
            "field1": {"chunk_references": ["table"]},
            "field2": {"chunk_references": ["high"]},
        },
        list_field=[],
        nested_list_field=[],
    )

    assert metadata_with_none.optional_field is None
