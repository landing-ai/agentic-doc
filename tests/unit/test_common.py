import time
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import httpx

from agentic_doc.common import (
    ChunkType,
    ChunkGroundingBox, 
    ChunkGrounding,
    Chunk,
    ParsedDocument,
    RetryableError,
    Document,
    Timer
)


def test_chunk_type_enum():
    # Test all the enumeration values
    assert ChunkType.title == "title"
    assert ChunkType.page_header == "page_header"
    assert ChunkType.page_footer == "page_footer"
    assert ChunkType.page_number == "page_number"
    assert ChunkType.key_value == "key_value"
    assert ChunkType.form == "form"
    assert ChunkType.table == "table"
    assert ChunkType.figure == "figure"
    assert ChunkType.text == "text"
    assert ChunkType.error == "error"


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
    
    # Test with None box
    grounding_no_box = ChunkGrounding(page=0, box=None)
    assert grounding_no_box.box is None
    
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
        chunk_type=ChunkType.title,
        chunk_id="123"
    )
    
    # Check attributes
    assert chunk.text == "Test Text"
    assert len(chunk.grounding) == 1
    assert chunk.grounding[0] == grounding
    assert chunk.chunk_type == ChunkType.title
    assert chunk.chunk_id == "123"
    
    # Test creating a Chunk with multiple groundings
    grounding2 = ChunkGrounding(page=1, box=box)
    chunk_multi = Chunk(
        text="Multi Page",
        grounding=[grounding, grounding2],
        chunk_type=ChunkType.text,
        chunk_id="456"
    )
    assert len(chunk_multi.grounding) == 2
    
    # Test creating a Chunk with no chunk_id
    chunk_no_id = Chunk(
        text="No ID",
        grounding=[grounding],
        chunk_type=ChunkType.text,
        chunk_id=None
    )
    assert chunk_no_id.chunk_id is None
    
    # Test serialization/deserialization
    chunk_dict = chunk.model_dump()
    chunk2 = Chunk.model_validate(chunk_dict)
    assert chunk2.text == chunk.text
    assert chunk2.chunk_type == chunk.chunk_type
    assert chunk2.chunk_id == chunk.chunk_id
    assert len(chunk2.grounding) == len(chunk.grounding)


def test_error_chunk():
    # Test creating an error chunk
    error_msg = "Test error message"
    page_idx = 42
    
    error_chunk = Chunk.error_chunk(error_msg, page_idx)
    
    # Check the error chunk
    assert error_chunk.text == error_msg
    assert error_chunk.chunk_type == ChunkType.error
    assert error_chunk.chunk_id is None
    assert len(error_chunk.grounding) == 1
    assert error_chunk.grounding[0].page == page_idx
    assert error_chunk.grounding[0].box is None


def test_parsed_document():
    # Create test chunks
    box = ChunkGroundingBox(l=0.1, t=0.2, r=0.8, b=0.9)
    grounding1 = ChunkGrounding(page=0, box=box)
    grounding2 = ChunkGrounding(page=1, box=box)
    
    chunk1 = Chunk(
        text="Title",
        grounding=[grounding1],
        chunk_type=ChunkType.title,
        chunk_id="1"
    )
    
    chunk2 = Chunk(
        text="Content",
        grounding=[grounding2],
        chunk_type=ChunkType.text,
        chunk_id="2"
    )
    
    # Create ParsedDocument
    doc = ParsedDocument(
        markdown="# Title\n\nContent",
        chunks=[chunk1, chunk2],
        start_page_idx=0,
        end_page_idx=1,
        doc_type="pdf"
    )
    
    # Check attributes
    assert doc.markdown == "# Title\n\nContent"
    assert len(doc.chunks) == 2
    assert doc.chunks[0] == chunk1
    assert doc.chunks[1] == chunk2
    assert doc.start_page_idx == 0
    assert doc.end_page_idx == 1
    assert doc.doc_type == "pdf"
    
    # Test with image doc_type
    image_doc = ParsedDocument(
        markdown="Image content",
        chunks=[chunk1],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="image"
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