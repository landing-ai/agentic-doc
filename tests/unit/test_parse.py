import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import httpx
from pydantic_core import Url

from agentic_doc.parse import (
    parse_documents,
    parse_and_save_documents,
    parse_and_save_document,
    _parse_pdf,
    _parse_image,
    _merge_part_results,
    _merge_next_part,
    _parse_doc_in_parallel,
    _parse_doc_parts,
    _send_parsing_request
)
from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    Document,
    ParsedDocument,
    RetryableError
)


def test_parse_and_save_documents_with_invalid_file(sample_pdf_path, results_dir):
    # Arrange
    input_files = [
        sample_pdf_path.parent / "invalid.pdf",  # Non-existent file
        sample_pdf_path,
    ]

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        parse_and_save_documents(input_files, result_save_dir=results_dir)


def test_parse_and_save_documents_empty_list(results_dir):
    # Act
    result_paths = parse_and_save_documents([], result_save_dir=results_dir)

    # Assert
    assert result_paths == []


def test_parse_documents_with_file_paths(mock_parsed_document):
    # Setup mock for _parse_pdf and _parse_image
    with patch("agentic_doc.parse.parse_and_save_document") as mock_parse:
        mock_parse.return_value = mock_parsed_document
        
        # Create test file paths
        file_paths = [
            "/path/to/document1.pdf",
            "/path/to/document2.jpg",
        ]
        
        # Call the function under test
        results = parse_documents(file_paths)
        
        # Check that parse_and_save_document was called for each file
        assert mock_parse.call_count == 2
        
        # Check the results
        assert len(results) == 2
        assert results[0] == mock_parsed_document
        assert results[1] == mock_parsed_document


def test_parse_documents_with_grounding_save_dir(mock_parsed_document, temp_dir):
    # Setup mock for parse_and_save_document
    with patch("agentic_doc.parse.parse_and_save_document") as mock_parse:
        mock_parse.return_value = mock_parsed_document
        
        # Call the function under test with grounding_save_dir
        results = parse_documents(
            ["/path/to/document.pdf"],
            grounding_save_dir=temp_dir
        )
        
        # Check that the grounding_save_dir was passed to parse_and_save_document
        mock_parse.assert_called_once_with(
            "/path/to/document.pdf",
            grounding_save_dir=temp_dir
        )


def test_parse_and_save_documents_with_url(mock_parsed_document, temp_dir):
    # Setup mock for parse_and_save_document
    with patch("agentic_doc.parse.parse_and_save_document") as mock_parse:
        # Configure mock to return a file path
        mock_file_path = Path(temp_dir) / "result.json"
        mock_parse.return_value = mock_file_path
        
        # Call the function under test with a URL
        result_paths = parse_and_save_documents(
            ["https://example.com/document.pdf"],
            result_save_dir=temp_dir,
            grounding_save_dir=temp_dir
        )
        
        # Check that parse_and_save_document was called with the URL and the right parameters
        mock_parse.assert_called_once_with(
            "https://example.com/document.pdf",
            result_save_dir=temp_dir,
            grounding_save_dir=temp_dir
        )
        
        # Check the results
        assert len(result_paths) == 1
        assert result_paths[0] == mock_file_path


def test_parse_and_save_document_with_local_file(temp_dir, mock_parsed_document):
    # Create a test file
    test_file = temp_dir / "test.pdf"
    with open(test_file, "wb") as f:
        f.write(b"%PDF-1.7\n")
    
    # Mock _parse_pdf function
    with patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document):
        # Call function without result_save_dir (should return parsed document)
        result = parse_and_save_document(test_file)
        assert isinstance(result, ParsedDocument)
        assert result == mock_parsed_document
        
        # Call function with result_save_dir (should return file path)
        result_dir = temp_dir / "results"
        result = parse_and_save_document(test_file, result_save_dir=result_dir)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".json"
        
        # Check that the result JSON contains the expected data
        with open(result) as f:
            result_data = json.load(f)
            assert "markdown" in result_data
            assert "chunks" in result_data
            assert "start_page_idx" in result_data
            assert "end_page_idx" in result_data
            assert "doc_type" in result_data


def test_parse_and_save_document_with_url(temp_dir, mock_parsed_document):
    # Mock download_file and _parse_pdf functions
    with patch("agentic_doc.parse.download_file") as mock_download, \
         patch("agentic_doc.parse.get_file_type", return_value="pdf"), \
         patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document):
        
        # Call function with URL
        result = parse_and_save_document("https://example.com/document.pdf")
        
        # Check that download_file was called
        mock_download.assert_called_once()
        
        # Check that the result is the parsed document
        assert isinstance(result, ParsedDocument)
        assert result == mock_parsed_document


def test_parse_and_save_document_with_invalid_file_type(temp_dir):
    # Create a test file that isn't a PDF or image
    test_file = temp_dir / "test.txt"
    with open(test_file, "w") as f:
        f.write("This is not a PDF or image")
    
    # Mock get_file_type to return an unsupported file type
    with patch("agentic_doc.parse.get_file_type", return_value="txt"):
        # Call function and check that it raises ValueError
        with pytest.raises(ValueError) as exc_info:
            parse_and_save_document(test_file)
        
        assert "Unsupported file type" in str(exc_info.value)


def test_parse_pdf(temp_dir, mock_parsed_document):
    # Create a test PDF file
    pdf_path = temp_dir / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")
    
    # Mock split_pdf and _parse_doc_in_parallel functions
    with patch("agentic_doc.parse.split_pdf") as mock_split, \
         patch("agentic_doc.parse._parse_doc_in_parallel") as mock_parse_parts:
        
        # Setup mocks
        mock_split.return_value = [
            Document(file_path=temp_dir / "test_1.pdf", start_page_idx=0, end_page_idx=1),
            Document(file_path=temp_dir / "test_2.pdf", start_page_idx=2, end_page_idx=3)
        ]
        mock_parse_parts.return_value = [mock_parsed_document, mock_parsed_document]
        
        # Call the function under test
        result = _parse_pdf(pdf_path)
        
        # Check that split_pdf was called with the right arguments
        mock_split.assert_called_once()
        
        # Check that _parse_doc_in_parallel was called
        mock_parse_parts.assert_called_once()
        
        # Check that the result is a ParsedDocument
        assert isinstance(result, ParsedDocument)


def test_parse_image(temp_dir, mock_parsed_document):
    # Create a test image file
    img_path = temp_dir / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(b"JFIF")
    
    # Mock _send_parsing_request function
    with patch("agentic_doc.parse._send_parsing_request") as mock_send_request:
        # Setup mock to return a valid response
        mock_send_request.return_value = {
            "data": {
                "markdown": mock_parsed_document.markdown,
                "chunks": [chunk.model_dump() for chunk in mock_parsed_document.chunks]
            }
        }
        
        # Call the function under test
        result = _parse_image(img_path)
        
        # Check that _send_parsing_request was called with the right arguments
        mock_send_request.assert_called_once_with(str(img_path))
        
        # Check that the result is a ParsedDocument with the expected values
        assert isinstance(result, ParsedDocument)
        assert result.markdown == mock_parsed_document.markdown
        assert result.doc_type == "image"
        assert result.start_page_idx == 0
        assert result.end_page_idx == 0


def test_parse_image_with_error(temp_dir):
    # Create a test image file
    img_path = temp_dir / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(b"JFIF")
    
    # Mock _send_parsing_request function to raise an exception
    error_msg = "Test error"
    with patch("agentic_doc.parse._send_parsing_request", side_effect=Exception(error_msg)):
        # Call the function under test
        result = _parse_image(img_path)
        
        # Check that the result contains an error chunk
        assert isinstance(result, ParsedDocument)
        assert result.doc_type == "image"
        assert result.start_page_idx == 0
        assert result.end_page_idx == 0
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_type == ChunkType.error
        assert result.chunks[0].text == error_msg


def test_merge_part_results_empty_list():
    # Call the function with an empty list
    result = _merge_part_results([])
    
    # Check that it returns an empty ParsedDocument
    assert isinstance(result, ParsedDocument)
    assert result.markdown == ""
    assert result.chunks == []
    assert result.start_page_idx == 0
    assert result.end_page_idx == 0
    assert result.doc_type == "pdf"


def test_merge_part_results_single_item(mock_parsed_document):
    # Call the function with a single item
    result = _merge_part_results([mock_parsed_document])
    
    # Check that it returns the item as is
    assert result == mock_parsed_document


def test_merge_part_results_multiple_items(mock_multi_page_parsed_document):
    # Create two parsed documents to merge
    doc1 = ParsedDocument(
        markdown="# Document 1",
        chunks=[
            Chunk(
                text="Document 1",
                chunk_type=ChunkType.title,
                chunk_id="1",
                grounding=[ChunkGrounding(page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2))]
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf"
    )
    
    doc2 = ParsedDocument(
        markdown="# Document 2",
        chunks=[
            Chunk(
                text="Document 2",
                chunk_type=ChunkType.title,
                chunk_id="2",
                grounding=[ChunkGrounding(page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2))]
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf"
    )
    
    # Call the function
    result = _merge_part_results([doc1, doc2])
    
    # Check the merged result
    assert result.markdown == "# Document 1\n\n# Document 2"
    assert len(result.chunks) == 2
    assert result.start_page_idx == 0
    assert result.end_page_idx == 1
    
    # Check that the page numbers were updated in the second document's chunks
    assert result.chunks[1].grounding[0].page == 1


def test_merge_next_part():
    # Create two ParsedDocuments to merge
    current_doc = ParsedDocument(
        markdown="# Current Doc",
        chunks=[
            Chunk(
                text="Current Doc",
                chunk_type=ChunkType.title,
                chunk_id="1",
                grounding=[ChunkGrounding(page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2))]
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf"
    )
    
    next_doc = ParsedDocument(
        markdown="# Next Doc",
        chunks=[
            Chunk(
                text="Next Doc",
                chunk_type=ChunkType.title,
                chunk_id="2",
                grounding=[ChunkGrounding(page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2))]
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf"
    )
    
    # Call the function
    _merge_next_part(current_doc, next_doc)
    
    # Check that the current_doc was updated
    assert current_doc.markdown == "# Current Doc\n\n# Next Doc"
    assert len(current_doc.chunks) == 2
    assert current_doc.end_page_idx == 1
    
    # Check that the page number was updated for the next doc's chunk
    assert current_doc.chunks[1].grounding[0].page == 1


def test_parse_doc_in_parallel(mock_parsed_document):
    # Create Document objects for testing
    doc_parts = [
        Document(file_path="/path/to/doc1.pdf", start_page_idx=0, end_page_idx=1),
        Document(file_path="/path/to/doc2.pdf", start_page_idx=2, end_page_idx=3)
    ]
    
    # Mock _parse_doc_parts
    with patch("agentic_doc.parse._parse_doc_parts", return_value=mock_parsed_document):
        # Call the function
        results = _parse_doc_in_parallel(doc_parts, doc_name="test.pdf")
        
        # Check the results
        assert len(results) == 2
        assert results[0] == mock_parsed_document
        assert results[1] == mock_parsed_document


def test_parse_doc_parts_success(mock_parsed_document):
    # Create a Document object for testing
    doc = Document(file_path="/path/to/doc.pdf", start_page_idx=0, end_page_idx=1)
    
    # Mock _send_parsing_request
    with patch("agentic_doc.parse._send_parsing_request") as mock_send_request:
        # Setup mock to return a valid response
        mock_send_request.return_value = {
            "data": {
                "markdown": mock_parsed_document.markdown,
                "chunks": [chunk.model_dump() for chunk in mock_parsed_document.chunks]
            }
        }
        
        # Call the function
        result = _parse_doc_parts(doc)
        
        # Check that _send_parsing_request was called with the right arguments
        mock_send_request.assert_called_once_with(str(doc.file_path))
        
        # Check the result
        assert isinstance(result, ParsedDocument)
        assert result.markdown == mock_parsed_document.markdown
        assert result.start_page_idx == 0
        assert result.end_page_idx == 1
        assert result.doc_type == "pdf"


def test_parse_doc_parts_error():
    # Create a Document object for testing
    doc = Document(file_path="/path/to/doc.pdf", start_page_idx=0, end_page_idx=1)
    
    # Mock _send_parsing_request to raise an exception
    error_msg = "Test error"
    with patch("agentic_doc.parse._send_parsing_request", side_effect=Exception(error_msg)):
        # Call the function
        result = _parse_doc_parts(doc)
        
        # Check that the result contains error chunks for each page
        assert isinstance(result, ParsedDocument)
        assert result.doc_type == "pdf"
        assert result.start_page_idx == 0
        assert result.end_page_idx == 1
        assert len(result.chunks) == 2  # One error chunk per page
        
        # Check the first error chunk
        assert result.chunks[0].chunk_type == ChunkType.error
        assert result.chunks[0].text == error_msg
        assert result.chunks[0].grounding[0].page == 0
        
        # Check the second error chunk
        assert result.chunks[1].chunk_type == ChunkType.error
        assert result.chunks[1].text == error_msg
        assert result.chunks[1].grounding[0].page == 1


def test_send_parsing_request_success():
    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"markdown": "Test", "chunks": []}}
    
    # Mock httpx.post to return the mock response
    with patch("agentic_doc.parse.httpx.post", return_value=mock_response), \
         patch("agentic_doc.parse.open", MagicMock()), \
         patch("agentic_doc.parse.Path") as mock_path:
        
        # Setup mock to make the suffix check work
        mock_path_instance = MagicMock()
        mock_path_instance.suffix.lower.return_value = ".pdf"
        mock_path.return_value = mock_path_instance
        
        # Call the function
        result = _send_parsing_request("test.pdf")
        
        # Check that the result matches the mock response
        assert result == {"data": {"markdown": "Test", "chunks": []}}


def test_send_parsing_request_retryable_error():
    # Create a mock response with a retryable error status code
    mock_response = MagicMock()
    mock_response.status_code = 429  # Rate limit error
    mock_response.text = "Rate limit exceeded"
    
    # Mock httpx.post to return the mock response
    with patch("agentic_doc.parse.httpx.post", return_value=mock_response), \
         patch("agentic_doc.parse.open", MagicMock()), \
         patch("agentic_doc.parse.Path") as mock_path:
        
        # Setup mock to make the suffix check work
        mock_path_instance = MagicMock()
        mock_path_instance.suffix.lower.return_value = ".pdf"
        mock_path.return_value = mock_path_instance
        
        # Call the function and check that it raises RetryableError
        with pytest.raises(RetryableError) as exc_info:
            _send_parsing_request("test.pdf")
        
        # Check that the exception contains the expected data
        assert exc_info.value.response == mock_response
        assert str(exc_info.value) == "429 - Rate limit exceeded"