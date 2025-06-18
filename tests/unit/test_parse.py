import json
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    Document,
    ParsedDocument,
)
from agentic_doc.connectors import (
    LocalConnector,
    LocalConnectorConfig,
)
from agentic_doc.parse import (
    _merge_next_part,
    _merge_part_results,
    _parse_doc_in_parallel,
    _parse_doc_parts,
    _parse_image,
    _parse_pdf,
    _send_parsing_request,
    parse,
    parse_and_save_document,
    parse_and_save_documents,
    parse_documents,
)


@pytest.fixture(autouse=True)
def patch_check_api_key():
    with patch("agentic_doc.parse.check_endpoint_and_api_key"):
        yield


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
            ["/path/to/document.pdf"], grounding_save_dir=temp_dir
        )

        # Check that the grounding_save_dir was passed to parse_and_save_document
        mock_parse.assert_called_once_with(
            "/path/to/document.pdf",
            grounding_save_dir=temp_dir,
            include_marginalia=True,
            include_metadata_in_markdown=True,
            result_save_dir=None,
            extraction_model=None,
            extraction_schema=None,
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
            include_marginalia=True,
            include_metadata_in_markdown=True,
            result_save_dir=temp_dir,
            grounding_save_dir=temp_dir,
            extraction_model=None,
            extraction_schema=None,
        )

        # Check that parse_and_save_document was called with the URL and the right parameters
        mock_parse.assert_called_once_with(
            "https://example.com/document.pdf",
            include_marginalia=True,
            include_metadata_in_markdown=True,
            result_save_dir=temp_dir,
            grounding_save_dir=temp_dir,
            extraction_model=None,
            extraction_schema=None,
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
    with (
        patch("agentic_doc.parse.download_file") as mock_download,
        patch("agentic_doc.parse.get_file_type", return_value="pdf"),
        patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document),
    ):
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
    with (
        patch("agentic_doc.parse.split_pdf") as mock_split,
        patch("agentic_doc.parse._parse_doc_in_parallel") as mock_parse_parts,
    ):
        # Setup mocks
        mock_split.return_value = [
            Document(
                file_path=temp_dir / "test_1.pdf", start_page_idx=0, end_page_idx=1
            ),
            Document(
                file_path=temp_dir / "test_2.pdf", start_page_idx=2, end_page_idx=3
            ),
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
                "chunks": [chunk.model_dump() for chunk in mock_parsed_document.chunks],
            }
        }

        # Call the function under test
        result = _parse_image(img_path)

        # Check that _send_parsing_request was called with the right arguments
        mock_send_request.assert_called_once_with(
            str(img_path),
            include_marginalia=True,
            include_metadata_in_markdown=True,
            extraction_model=None,
            extraction_schema=None,
        )

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
    with patch(
        "agentic_doc.parse._send_parsing_request", side_effect=Exception(error_msg)
    ):
        # Call the function under test
        result = _parse_image(img_path)

        # Check that the result contains no chunks but has an error in the errors field
        assert isinstance(result, ParsedDocument)
        assert result.doc_type == "image"
        assert result.start_page_idx == 0
        assert result.end_page_idx == 0
        assert len(result.chunks) == 0
        assert len(result.errors) == 1
        assert result.errors[0].page_num == 0
        assert result.errors[0].error == error_msg
        assert result.errors[0].error_code == -1


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
                chunk_type=ChunkType.text,
                chunk_id="1",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    doc2 = ParsedDocument(
        markdown="# Document 2",
        chunks=[
            Chunk(
                text="Document 2",
                chunk_type=ChunkType.text,
                chunk_id="2",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf",
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
                chunk_type=ChunkType.text,
                chunk_id="1",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    next_doc = ParsedDocument(
        markdown="# Next Doc",
        chunks=[
            Chunk(
                text="Next Doc",
                chunk_type=ChunkType.text,
                chunk_id="2",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf",
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
        Document(file_path="/path/to/doc2.pdf", start_page_idx=2, end_page_idx=3),
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
                "chunks": [chunk.model_dump() for chunk in mock_parsed_document.chunks],
            }
        }

        # Call the function
        result = _parse_doc_parts(doc)

        # Check that _send_parsing_request was called with the right arguments
        mock_send_request.assert_called_once_with(
            str(doc.file_path),
            include_marginalia=True,
            include_metadata_in_markdown=True,
            extraction_model=None,
            extraction_schema=None,
        )

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
    with patch(
        "agentic_doc.parse._send_parsing_request", side_effect=Exception(error_msg)
    ):
        # Call the function
        result = _parse_doc_parts(doc)

        # Check that the result contains no chunks but has errors for each page
        assert isinstance(result, ParsedDocument)
        assert result.doc_type == "pdf"
        assert result.start_page_idx == 0
        assert result.end_page_idx == 1
        assert len(result.chunks) == 0  # No chunks on error
        assert len(result.errors) == 2  # One error per page

        # Check the first error
        assert result.errors[0].page_num == 0
        assert result.errors[0].error == error_msg
        assert result.errors[0].error_code == -1

        # Check the second error
        assert result.errors[1].page_num == 1
        assert result.errors[1].error == error_msg
        assert result.errors[1].error_code == -1


def test_send_parsing_request_success():
    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"markdown": "Test", "chunks": []}}

    # Mock httpx.post to return the mock response
    with (
        patch("agentic_doc.parse.httpx.post", return_value=mock_response),
        patch("agentic_doc.parse.open", MagicMock()),
        patch("agentic_doc.parse.Path") as mock_path,
    ):
        # Setup mock to make the suffix check work
        mock_path_instance = MagicMock()
        mock_path_instance.suffix.lower.return_value = ".pdf"
        mock_path.return_value = mock_path_instance

        # Call the function
        result = _send_parsing_request("test.pdf")

        # Check that the result matches the mock response
        assert result == {"data": {"markdown": "Test", "chunks": []}}


def test_parse_and_save_document_with_grounding_save_dir(
    temp_dir, mock_parsed_document
):
    # Test that grounding images are saved when grounding_save_dir is provided
    test_file = temp_dir / "test.pdf"
    with open(test_file, "wb") as f:
        f.write(b"%PDF-1.7\n")

    grounding_dir = temp_dir / "groundings"

    # Mock the required functions
    with (
        patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document),
        patch("agentic_doc.parse.save_groundings_as_images") as mock_save_groundings,
    ):
        result = parse_and_save_document(test_file, grounding_save_dir=grounding_dir)
        # Check that save_groundings_as_images was called
        args, kwargs = mock_save_groundings.call_args
        assert args[0] == test_file
        assert args[1] == mock_parsed_document.chunks
        assert str(args[2]).startswith(str(grounding_dir))
        assert kwargs.get("inplace") is True


def test_parse_pdf_with_empty_result(temp_dir):
    # Test parsing a PDF that returns no chunks
    pdf_path = temp_dir / "empty.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    with (
        patch("agentic_doc.parse.split_pdf") as mock_split,
        patch("agentic_doc.parse._parse_doc_in_parallel") as mock_parse_parts,
    ):
        # Mock an empty result
        empty_doc = ParsedDocument(
            markdown="", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="pdf"
        )

        mock_split.return_value = [
            Document(
                file_path=temp_dir / "empty_1.pdf", start_page_idx=0, end_page_idx=0
            )
        ]
        mock_parse_parts.return_value = [empty_doc]

        result = _parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert len(result.chunks) == 0
        assert result.markdown == ""


def test_merge_part_results_with_errors(mock_parsed_document):
    # Test merging results that contain errors
    from agentic_doc.common import PageError

    doc_with_errors = ParsedDocument(
        markdown="# Document with errors",
        chunks=[],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
        errors=[PageError(page_num=0, error="Test error", error_code=-1)],
    )

    result = _merge_part_results([mock_parsed_document, doc_with_errors])

    # Should merge both documents and preserve errors
    assert isinstance(result, ParsedDocument)
    assert len(result.errors) == 1
    assert result.errors[0].error == "Test error"


def test_parse_documents_with_mixed_file_types(temp_dir):
    # Test parsing a mix of file types
    pdf_path = temp_dir / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    img_path = temp_dir / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(b"JFIF")

    # Mock the parsing functions
    mock_pdf_doc = ParsedDocument(
        markdown="# PDF Document",
        chunks=[],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    mock_img_doc = ParsedDocument(
        markdown="# Image Document",
        chunks=[],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="image",
    )

    with (
        patch("agentic_doc.parse._parse_pdf", return_value=mock_pdf_doc),
        patch("agentic_doc.parse._parse_image", return_value=mock_img_doc),
    ):
        results = parse_documents([str(pdf_path), str(img_path)])

        assert len(results) == 2
        assert results[0].doc_type == "pdf"
        assert results[1].doc_type == "image"


def test_send_parsing_request_with_different_file_types(temp_dir):
    # Test that _send_parsing_request handles different file extensions correctly

    # Test with PDF
    pdf_path = temp_dir / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"markdown": "PDF Test", "chunks": []}}

    with (
        patch("agentic_doc.parse.httpx.post", return_value=mock_response),
        patch("agentic_doc.parse.open", MagicMock()),
    ):
        result = _send_parsing_request(str(pdf_path))
        assert result["data"]["markdown"] == "PDF Test"

    # Test with image
    img_path = temp_dir / "test.png"
    with open(img_path, "wb") as f:
        f.write(b"PNG")

    mock_response.json.return_value = {"data": {"markdown": "Image Test", "chunks": []}}

    with (
        patch("agentic_doc.parse.httpx.post", return_value=mock_response),
        patch("agentic_doc.parse.open", MagicMock()),
    ):
        result = _send_parsing_request(str(img_path))
        assert result["data"]["markdown"] == "Image Test"


def test_document_string_representation():
    # Test the string representation of Document objects
    doc = Document(
        file_path=Path("/path/to/test_document.pdf"), start_page_idx=5, end_page_idx=10
    )

    expected_str = "File name: test_document.pdf\tPage: [5:10]"
    assert str(doc) == expected_str


def test_parse_pdf_handles_single_page_document(temp_dir):
    # Test parsing a single-page PDF
    pdf_path = temp_dir / "single_page.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    single_page_doc = ParsedDocument(
        markdown="# Single Page",
        chunks=[],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    with (
        patch("agentic_doc.parse.split_pdf") as mock_split,
        patch("agentic_doc.parse._parse_doc_in_parallel") as mock_parse_parts,
    ):
        mock_split.return_value = [
            Document(
                file_path=temp_dir / "single_1.pdf", start_page_idx=0, end_page_idx=0
            )
        ]
        mock_parse_parts.return_value = [single_page_doc]

        result = _parse_pdf(pdf_path)

        assert result.start_page_idx == 0
        assert result.end_page_idx == 0
        assert result.doc_type == "pdf"


class TestParseFunctionConsolidated:
    """Test the consolidated parse function."""

    def test_parse_single_document(self, temp_dir, mock_parsed_document):
        """Test parsing a single document."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        with patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document):
            result = parse(test_file)

            assert all(isinstance(res, ParsedDocument) for res in result)
            assert result == [mock_parsed_document]

    def test_parse_single_document_with_save_dir(self, temp_dir, mock_parsed_document):
        """Test parsing a single document with save directory."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        result_dir = temp_dir / "results"

        with patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document):
            result = parse(test_file, result_save_dir=result_dir)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], ParsedDocument)

    def test_parse_multiple_documents(self, temp_dir, mock_parsed_document):
        """Test parsing multiple documents."""
        test_files = [temp_dir / "test1.pdf", temp_dir / "test2.pdf"]
        for f in test_files:
            with open(f, "wb") as file:
                file.write(b"%PDF-1.7\n")

        with patch(
            "agentic_doc.parse.parse_documents",
            return_value=[mock_parsed_document, mock_parsed_document],
        ) as mock_parse:
            result = parse([str(f) for f in test_files])

            assert isinstance(result, list)
            assert len(result) == 2
            mock_parse.assert_called_once()

    def test_parse_with_grounding_save_dir(self, temp_dir, mock_parsed_document):
        """Test parsing with grounding save directory."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        grounding_dir = temp_dir / "groundings"

        with (
            patch("agentic_doc.parse._parse_pdf", return_value=mock_parsed_document),
            patch(
                "agentic_doc.parse.save_groundings_as_images"
            ) as mock_save_groundings,
        ):
            result = parse(test_file, grounding_save_dir=grounding_dir)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], ParsedDocument)
            # Verify that save_groundings_as_images was called
            mock_save_groundings.assert_called_once()

    def test_parse_with_local_connector_config(self, temp_dir, mock_parsed_document):
        """Test parsing with local connector configuration."""
        # Create test files
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        config = LocalConnectorConfig()

        with (
            patch("agentic_doc.parse.create_connector") as mock_create,
            patch(
                "agentic_doc.parse._parse_document_list",
                return_value=[mock_parsed_document],
            ) as mock_parse_list,
        ):
            # Mock connector
            mock_connector = MagicMock()
            mock_connector.list_files.return_value = [str(test_file)]
            mock_connector.download_file.return_value = test_file
            mock_create.return_value = mock_connector

            result = parse(config, connector_path=str(temp_dir))

            assert isinstance(result, list)
            mock_create.assert_called_once_with(config)
            mock_connector.list_files.assert_called_once_with(str(temp_dir), None)

    def test_parse_with_local_connector_instance(self, temp_dir, mock_parsed_document):
        """Test parsing with local connector instance."""
        # Create test files
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        config = LocalConnectorConfig()
        connector = LocalConnector(config)

        with (
            patch.object(connector, "list_files", return_value=[str(test_file)]),
            patch.object(connector, "download_file", return_value=test_file),
            patch(
                "agentic_doc.parse._parse_document_list",
                return_value=[mock_parsed_document],
            ) as mock_parse_list,
        ):
            result = parse(connector, connector_path=str(temp_dir))

            assert isinstance(result, list)
            connector.list_files.assert_called_once_with(str(temp_dir), None)

    def test_parse_with_connector_no_files_found(self, temp_dir):
        """Test parsing with connector when no files are found."""
        config = LocalConnectorConfig()

        with patch("agentic_doc.parse.create_connector") as mock_create:
            # Mock connector that returns no files
            mock_connector = MagicMock()
            mock_connector.list_files.return_value = []
            mock_create.return_value = mock_connector

            result = parse(config, connector_path=str(temp_dir))

            assert result == []

    def test_parse_with_connector_download_failures(
        self, temp_dir, mock_parsed_document
    ):
        """Test parsing with connector when some downloads fail."""
        config = LocalConnectorConfig()

        with (
            patch("agentic_doc.parse.create_connector") as mock_create,
            patch(
                "agentic_doc.parse._parse_document_list",
                return_value=[mock_parsed_document],
            ) as mock_parse_list,
        ):
            # Mock connector
            mock_connector = MagicMock()
            mock_connector.list_files.return_value = ["file1.pdf", "file2.pdf"]
            # First download succeeds, second fails
            mock_connector.download_file.side_effect = [
                Path("file1.pdf"),
                Exception("Download failed"),
            ]
            mock_create.return_value = mock_connector

            result = parse(config)

            # Should continue with successful downloads
            assert isinstance(result, list)
            assert mock_connector.download_file.call_count == 2
            mock_parse_list.assert_called_once()

    def test_parse_with_connector_all_downloads_fail(self, temp_dir):
        """Test parsing with connector when all downloads fail."""
        config = LocalConnectorConfig()

        with patch("agentic_doc.parse.create_connector") as mock_create:
            # Mock connector
            mock_connector = MagicMock()
            mock_connector.list_files.return_value = ["file1.pdf", "file2.pdf"]
            mock_connector.download_file.side_effect = Exception("Download failed")
            mock_create.return_value = mock_connector

            result = parse(config)

            assert result == []

    def test_parse_unsupported_type(self):
        """Test parsing with unsupported document type."""
        with pytest.raises(ValueError, match="Unsupported documents type"):
            parse(123)  # Invalid type

    def test_parse_with_marginalia_and_metadata_flags(
        self, temp_dir, mock_parsed_document
    ):
        """Test parsing with marginalia and metadata flags."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(
                test_file, include_marginalia=False, include_metadata_in_markdown=False
            )

            mock_parse.assert_called_once_with(
                test_file,
                include_marginalia=False,
                include_metadata_in_markdown=False,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=None,
                extraction_schema=None,
            )

    def test_parse_with_bytes(self, mock_parsed_document):
        """Test parsing with bytes."""
        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(
                b"%PDF-1.7\n",
                include_marginalia=False,
                include_metadata_in_markdown=False,
            )

            mock_parse.assert_called_once_with(
                ANY,
                include_marginalia=False,
                include_metadata_in_markdown=False,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=None,
                extraction_schema=None,
            )

    def test_parse_list_with_save_dir(self, temp_dir, mock_parsed_document):
        """Test parsing list of documents with save directory."""
        test_files = [temp_dir / "test1.pdf", temp_dir / "test2.pdf"]
        test_save_files = [temp_dir / "result1.json", temp_dir / "result2.json"]
        for f in test_files:
            with open(f, "wb") as file:
                file.write(b"%PDF-1.7\n")

        for f in test_save_files:
            with open(f, "w") as file:
                file.write(
                    '{"markdown": "", "chunks": [], "start_page_idx": 0, "end_page_idx": 0, "doc_type": "pdf"}'
                )

        result_dir = temp_dir / "results"

        with patch(
            "agentic_doc.parse.parse_and_save_documents",
            return_value=[Path(test_save_files[0]), Path(test_save_files[1])],
        ) as mock_parse:
            result = parse([str(f) for f in test_files], result_save_dir=result_dir)

            assert isinstance(result, list)
            assert len(result) == 2
            mock_parse.assert_called_once()

    def test_parse_url_string(self, mock_parsed_document):
        """Test parsing a URL string."""
        url = "https://example.com/document.pdf"

        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(url)

            assert all(isinstance(res, ParsedDocument) for res in result)
            mock_parse.assert_called_once_with(
                url,
                include_marginalia=True,
                include_metadata_in_markdown=True,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=None,
                extraction_schema=None,
            )

    def test_parse_with_extraction_model(self, temp_dir, mock_parsed_document):
        """Test parsing with an extraction model."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        class EmployeeFields(BaseModel):
            employee_name: str = Field(description="the full name of the employee")
            gross_pay: float = Field(description="the gross pay of the employee")

        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(test_file, extraction_model=EmployeeFields)
            assert all(isinstance(res, ParsedDocument) for res in result)
            mock_parse.assert_called_once_with(
                test_file,
                include_marginalia=True,
                include_metadata_in_markdown=True,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=EmployeeFields,
                extraction_schema=None,
            )

    def test_extraction_metadata_with_simple_model(self, sample_image_path):
        class PersonInfo(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Test Document\nName: John Doe\nAge: 30",
                    "chunks": [
                        {
                            "text": "Name: John Doe",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {"name": "John Doe", "age": 30},
                    "extraction_metadata": {
                        "name": {"chunk_references": ["high"]},
                        "age": {"chunk_references": ["medium"]},
                    },
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_model=PersonInfo)

            # Verify extraction is correctly typed
            assert isinstance(result[0].extraction, PersonInfo)
            assert result[0].extraction.name == "John Doe"
            assert result[0].extraction.age == 30

            # Verify extraction_metadata is correctly typed
            metadata = result[0].extraction_metadata
            assert metadata is not None

            # Check that metadata fields are dict[str, list[str]]
            assert isinstance(metadata.name, dict)
            assert isinstance(metadata.age, dict)

            # Check specific metadata values
            assert metadata.name["chunk_references"] == ["high"]
            assert metadata.age["chunk_references"] == ["medium"]

    def test_extraction_metadata_with_nested_models(self, sample_image_path):
        """Test extraction_metadata functionality with nested models."""

        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City")

        class Person(BaseModel):
            name: str = Field(description="Person's name")
            address: Address = Field(description="Person's address")

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Person Info\nName: Jane Smith\nAddress: 123 Main St, Springfield",
                    "chunks": [
                        {
                            "text": "Name: Jane Smith",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {
                        "name": "Jane Smith",
                        "address": {"street": "123 Main St", "city": "Springfield"},
                    },
                    "extraction_metadata": {
                        "name": {"chunk_references": ["high"]},
                        "address": {
                            "street": {"chunk_references": ["medium"]},
                            "city": {"chunk_references": ["high"]},
                        },
                    },
                },
                "errors": [],
            }
            result = parse(sample_image_path, extraction_model=Person)

            assert isinstance(result[0].extraction, Person)
            assert result[0].extraction.name == "Jane Smith"
            assert isinstance(result[0].extraction.address, Address)
            assert result[0].extraction.address.street == "123 Main St"
            assert result[0].extraction.address.city == "Springfield"

            metadata = result[0].extraction_metadata
            assert metadata is not None

            assert isinstance(metadata.name, dict)
            assert metadata.name["chunk_references"] == ["high"]

            assert hasattr(metadata, "address")
            assert hasattr(metadata.address, "street")
            assert hasattr(metadata.address, "city")

            assert isinstance(metadata.address.street, dict)
            assert isinstance(metadata.address.city, dict)
            assert metadata.address.street["chunk_references"] == ["medium"]
            assert metadata.address.city["chunk_references"] == ["high"]

    def test_extraction_metadata_with_optional_fields(self, sample_image_path):
        """Test extraction_metadata functionality with optional fields."""

        class PersonWithOptional(BaseModel):
            name: str = Field(description="Person's name")
            phone: Optional[str] = Field(default=None, description="Phone number")
            email: Optional[str] = Field(default=None, description="Email address")

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Contact Info\nName: Bob Johnson\nEmail: bob@example.com",
                    "chunks": [
                        {
                            "text": "Name: Bob Johnson",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {
                        "name": "Bob Johnson",
                        "phone": None,
                        "email": "bob@example.com",
                    },
                    "extraction_metadata": {
                        "name": {"chunk_references": ["high"]},
                        "phone": None,
                        "email": {"chunk_references": ["medium"]},
                    },
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_model=PersonWithOptional)

            assert isinstance(result[0].extraction, PersonWithOptional)
            assert result[0].extraction.name == "Bob Johnson"
            assert result[0].extraction.phone is None
            assert result[0].extraction.email == "bob@example.com"

            metadata = result[0].extraction_metadata
            assert metadata is not None

            assert isinstance(metadata.name, dict)
            assert metadata.name["chunk_references"] == ["high"]

            assert metadata.phone is None  # Optional field with no data should be None
            assert isinstance(metadata.email, dict)
            assert metadata.email["chunk_references"] == ["medium"]

    def test_extraction_metadata_with_list_fields(self, sample_image_path):
        """Test extraction_metadata functionality with list fields."""

        class Skill(BaseModel):
            name: str = Field(description="Skill name")
            level: str = Field(description="Skill level")

        class PersonWithSkills(BaseModel):
            name: str = Field(description="Person's name")
            skills: List[Skill] = Field(description="List of skills")

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Resume\nName: Alice Brown\nSkills: Python (Expert), Java (Intermediate)",
                    "chunks": [
                        {
                            "text": "Name: Alice Brown",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {
                        "name": "Alice Brown",
                        "skills": [
                            {"name": "Python", "level": "Expert"},
                            {"name": "Java", "level": "Intermediate"},
                        ],
                    },
                    "extraction_metadata": {
                        "name": {"chunk_references": ["high"]},
                        "skills": [
                            {
                                "name": {"chunk_references": ["high"]},
                                "level": {"chunk_references": ["high"]},
                            },
                            {
                                "name": {"chunk_references": ["high"]},
                                "level": {"chunk_references": ["medium"]},
                            },
                        ],
                    },
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_model=PersonWithSkills)

            assert isinstance(result[0].extraction, PersonWithSkills)
            assert result[0].extraction.name == "Alice Brown"
            assert len(result[0].extraction.skills) == 2
            assert result[0].extraction.skills[0].name == "Python"
            assert result[0].extraction.skills[0].level == "Expert"

            metadata = result[0].extraction_metadata
            assert metadata is not None

            assert isinstance(metadata.name, dict)
            assert metadata.name["chunk_references"] == ["high"]

            assert isinstance(metadata.skills, list)
            assert len(metadata.skills) == 2

            first_skill_meta = metadata.skills[0]
            assert isinstance(first_skill_meta.name, dict)
            assert isinstance(first_skill_meta.level, dict)
            assert first_skill_meta.name["chunk_references"] == ["high"]
            assert first_skill_meta.level["chunk_references"] == ["high"]

            second_skill_meta = metadata.skills[1]
            assert isinstance(second_skill_meta.name, dict)
            assert isinstance(second_skill_meta.level, dict)
            assert second_skill_meta.name["chunk_references"] == ["high"]
            assert second_skill_meta.level["chunk_references"] == ["medium"]

    def test_extraction_metadata_error(self, sample_image_path):
        """Test extraction_metadata error."""

        class Skill(BaseModel):
            name: str = Field(description="Skill name")
            level: str = Field(description="Skill level")

        class PersonWithSkills(BaseModel):
            name: str = Field(description="Person's name")
            skills: List[Skill] = Field(description="List of skills")

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Resume\nName: Alice Brown\nSkills: Python (Expert), Java (Intermediate)",
                    "chunks": [
                        {
                            "text": "Name: Alice Brown",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {
                        "name": "Alice Brown",
                        "skills": [
                            {"name": "Python", "level": "Expert"},
                            {"name": "Java", "level": "Intermediate"},
                        ],
                    },
                    "extraction_metadata": {
                        "name": "Alice Brown",
                        "skills": [
                            {
                                "name": {"chunk_references": ["high"]},
                                "level": {"chunk_references": ["high"]},
                            },
                            {
                                "name": {"chunk_references": ["high"]},
                                "level": {"chunk_references": ["medium"]},
                            },
                        ],
                    },
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_model=PersonWithSkills)
            assert result[0].extraction is None
            assert result[0].extraction_metadata is None
            assert "validation error" in result[0].errors[0].error.lower()

    def test_parse_with_extraction_schema(self, temp_dir, mock_parsed_document):
        """Test parsing with an extraction schema."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        extraction_schema = {
            "type": "object",
            "properties": {
                "employee_name": {"type": "string", "description": "the full name of the employee"},
                "gross_pay": {"type": "number", "description": "the gross pay of the employee"}
            },
            "required": ["employee_name", "gross_pay"]
        }

        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(test_file, extraction_schema=extraction_schema)
            assert all(isinstance(res, ParsedDocument) for res in result)
            mock_parse.assert_called_once_with(
                test_file,
                include_marginalia=True,
                include_metadata_in_markdown=True,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=None,
                extraction_schema=extraction_schema,
            )

    def test_parse_with_extraction_schema_validation(self, sample_image_path):
        """Test that extraction_schema validates the response correctly."""
        extraction_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"}
            },
            "required": ["name", "age"]
        }

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Test Document\nName: John Doe\nAge: 30",
                    "chunks": [
                        {
                            "text": "Name: John Doe",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {"name": "John Doe", "age": 30},
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_schema=extraction_schema)

            # Verify extraction is a dict (not a Pydantic model)
            assert isinstance(result[0].extraction, dict)
            assert result[0].extraction["name"] == "John Doe"
            assert result[0].extraction["age"] == 30

    def test_parse_with_extraction_schema_validation_error(self, sample_image_path):
        """Test that extraction_schema validation errors are handled correctly."""
        extraction_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"}
            },
            "required": ["name", "age"]
        }

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            # Return data that doesn't match the schema (age is string instead of integer)
            mock_request.return_value = {
                "data": {
                    "markdown": "# Test Document\nName: John Doe\nAge: thirty",
                    "chunks": [
                        {
                            "text": "Name: John Doe",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {"name": "John Doe", "age": "thirty"},  # Invalid: age should be integer
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_schema=extraction_schema)

            assert result[0].extraction is None
            assert len(result[0].errors) > 0

    def test_parse_with_extraction_schema_complex(self, sample_image_path):
        """Test extraction_schema with complex nested schema."""
        extraction_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string", "description": "Street address"},
                        "city": {"type": "string", "description": "City"}
                    },
                    "required": ["street", "city"]
                },
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Skill name"},
                            "level": {"type": "string", "description": "Skill level"}
                        },
                        "required": ["name", "level"]
                    }
                }
            },
            "required": ["name", "address", "skills"]
        }

        with patch("agentic_doc.parse._send_parsing_request") as mock_request:
            mock_request.return_value = {
                "data": {
                    "markdown": "# Resume\nName: Alice Brown\nAddress: 123 Main St, Springfield\nSkills: Python (Expert), Java (Intermediate)",
                    "chunks": [
                        {
                            "text": "Name: Alice Brown",
                            "grounding": [
                                {
                                    "page": 0,
                                    "box": {"l": 0.1, "t": 0.1, "r": 0.9, "b": 0.2},
                                }
                            ],
                            "chunk_type": "text",
                            "chunk_id": "1",
                        }
                    ],
                    "extracted_schema": {
                        "name": "Alice Brown",
                        "address": {
                            "street": "123 Main St",
                            "city": "Springfield"
                        },
                        "skills": [
                            {"name": "Python", "level": "Expert"},
                            {"name": "Java", "level": "Intermediate"}
                        ]
                    },
                },
                "errors": [],
            }

            result = parse(sample_image_path, extraction_schema=extraction_schema)

            # Verify extraction is a dict with complex nested structure
            assert isinstance(result[0].extraction, dict)
            assert result[0].extraction["name"] == "Alice Brown"
            assert isinstance(result[0].extraction["address"], dict)
            assert result[0].extraction["address"]["street"] == "123 Main St"
            assert result[0].extraction["address"]["city"] == "Springfield"
            assert isinstance(result[0].extraction["skills"], list)
            assert len(result[0].extraction["skills"]) == 2
            assert result[0].extraction["skills"][0]["name"] == "Python"
            assert result[0].extraction["skills"][0]["level"] == "Expert"

    def test_parse_with_both_extraction_model_and_schema_error(self, temp_dir):
        """Test that providing both extraction_model and extraction_schema raises an error."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        class EmployeeFields(BaseModel):
            employee_name: str = Field(description="the full name of the employee")
            gross_pay: float = Field(description="the gross pay of the employee")

        extraction_schema = {
            "type": "object",
            "properties": {
                "employee_name": {"type": "string"},
                "gross_pay": {"type": "number"}
            }
        }

        with pytest.raises(ValueError, match="extraction_model and extraction_schema cannot be used together"):
            parse(test_file, extraction_model=EmployeeFields, extraction_schema=extraction_schema)

    def test_parse_with_neither_extraction_model_nor_schema(self, temp_dir, mock_parsed_document):
        """Test parsing without any extraction model or schema."""
        test_file = temp_dir / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"%PDF-1.7\n")

        with patch(
            "agentic_doc.parse.parse_and_save_document",
            return_value=mock_parsed_document,
        ) as mock_parse:
            result = parse(test_file)
            assert all(isinstance(res, ParsedDocument) for res in result)
            mock_parse.assert_called_once_with(
                test_file,
                include_marginalia=True,
                include_metadata_in_markdown=True,
                grounding_save_dir=None,
                result_save_dir=None,
                extraction_model=None,
                extraction_schema=None,
            )

