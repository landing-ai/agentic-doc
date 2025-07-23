import json
import os

import pytest
from pydantic import BaseModel, Field

from agentic_doc.common import ChunkType, ParsedDocument, MetadataType
from agentic_doc.config import settings, get_settings, ParseConfig
from agentic_doc.parse import (
    parse,
    parse_and_save_document,
    parse_and_save_documents,
    parse_documents,
)


def test_parse_and_save_documents_multiple_inputs(sample_image_path, results_dir):
    # Arrange
    input_file = sample_image_path

    # Act
    result_paths = parse_and_save_documents(
        [
            input_file,
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        ],
        result_save_dir=results_dir,
        grounding_save_dir=results_dir,
    )

    # Assert
    assert len(result_paths) == 2
    for result_path in result_paths:
        result_path = result_paths[0]
        assert result_path.exists()

        # Verify the saved JSON can be loaded and has expected structure
        with open(result_path) as f:
            result_data = json.load(f)

        parsed_doc = ParsedDocument.model_validate(result_data)
        assert parsed_doc.markdown
        assert len(parsed_doc.chunks) > 0
        assert parsed_doc.start_page_idx == 0
        assert parsed_doc.end_page_idx == 0
        assert len(parsed_doc.errors) == 0


def test_parse_and_save_documents_single_pdf(sample_pdf_path, results_dir):
    # Arrange
    input_file = sample_pdf_path

    # Act
    result_paths = parse_and_save_documents(
        [input_file],
        result_save_dir=results_dir,
        grounding_save_dir=results_dir,
    )

    # Assert
    assert len(result_paths) == 1
    result_path = result_paths[0]
    assert result_path.exists()

    # Verify the saved JSON can be loaded and has expected structure
    with open(result_path) as f:
        result_data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(result_data)
    assert parsed_doc.markdown
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx == 3
    assert parsed_doc.doc_type == "pdf"
    assert len(parsed_doc.chunks) >= 10
    # Verify that chunks are ordered by page number
    for i in range(1, len(parsed_doc.chunks)):
        prev_page = parsed_doc.chunks[i - 1].grounding[0].page
        curr_page = parsed_doc.chunks[i].grounding[0].page
        assert (
            curr_page >= prev_page
        ), f"Chunks not ordered by page: chunk {i - 1} (page {prev_page}) followed by chunk {i} (page {curr_page})"

    # Verify that there were no errors
    assert len(parsed_doc.errors) == 0

    # Verify that there were no errors
    assert len(parsed_doc.errors) == 0

    # Verify that grounding images were saved
    for chunk in parsed_doc.chunks:
        for grounding in chunk.grounding:
            assert grounding.image_path.exists()


def test_parse_single_image(sample_image_path):
    # Act
    result = parse_documents([sample_image_path])

    # Assert
    assert len(result) == 1
    parsed_doc = result[0]

    # Check basic structure
    assert parsed_doc.doc_type == "image"
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx == 0
    assert parsed_doc.markdown
    assert len(parsed_doc.chunks) > 0

    # Check chunk structure
    for chunk in parsed_doc.chunks:
        assert chunk.text
        assert len(chunk.grounding) > 0
        for grounding in chunk.grounding:
            assert grounding.page == 0
            if grounding.box:
                assert 0 <= grounding.box.l <= 1
                assert 0 <= grounding.box.t <= 1
                assert 0 <= grounding.box.r <= 1
                assert 0 <= grounding.box.b <= 1


@pytest.mark.skipif(
    not get_settings().vision_agent_api_key,
    reason="API key not set, skipping integration test that requires actual API call",
)
def test_parse_and_save_document_with_url(results_dir):
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Act
    result_path = parse_and_save_document(
        url, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    # Assert
    assert result_path.exists()
    assert result_path.suffix == ".json"

    # Verify JSON content
    with open(result_path) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)
    assert parsed_doc.doc_type == "pdf"
    assert parsed_doc.markdown

    # Check for non-error chunks
    non_error_chunks = [c for c in parsed_doc.chunks]
    assert len(non_error_chunks) > 0

    # Check groundings
    for chunk in non_error_chunks:
        for grounding in chunk.grounding:
            if grounding.image_path:
                assert os.path.isfile(grounding.image_path)


def test_parse_multipage_pdf(multi_page_pdf, results_dir):
    # Act
    result = parse_and_save_document(
        multi_page_pdf, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    # Assert
    assert result.exists()

    # Verify JSON content
    with open(result) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)
    assert parsed_doc.doc_type == "pdf"

    # Multi-page PDF should have end_page_idx > 0
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx > 0

    # Check that there are chunks from multiple pages
    page_indices = set(
        grounding.page for chunk in parsed_doc.chunks for grounding in chunk.grounding
    )

    # There should be at least 2 pages with content
    assert len(page_indices) > 1, "Expected chunks from multiple pages"

    # Page indices should be consecutive
    assert page_indices == set(range(min(page_indices), max(page_indices) + 1))


def test_parse_complex_pdf_with_table_and_image(complex_pdf, results_dir):
    # Act
    result = parse_and_save_document(
        complex_pdf, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    # Assert
    assert result.exists()

    # Verify JSON content
    with open(result) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)

    # Check for specific chunk types that should be present in a complex PDF
    chunk_types = [chunk.chunk_type for chunk in parsed_doc.chunks]

    # The complex PDF fixture has text, table, and potentially a figure
    assert ChunkType.text in chunk_types, "Text chunk not found"
    assert ChunkType.table in chunk_types, "Table chunk not found"

    # Count chunks by type
    type_counts = {}
    for chunk in parsed_doc.chunks:
        if chunk.chunk_type not in type_counts:
            type_counts[chunk.chunk_type] = 0
        type_counts[chunk.chunk_type] += 1

    # Print chunk type counts for debugging if test fails
    print(f"Chunk type counts: {type_counts}")

    # Check that there are multiple text chunks (since the PDF has multiple text sections)
    assert type_counts.get(ChunkType.text, 0) >= 1, "Expected at least one text chunk"

    # Check that there is at least one table chunk
    assert type_counts.get(ChunkType.table, 0) >= 1, "Expected at least one table chunk"


@pytest.mark.skipif(
    not get_settings().vision_agent_api_key,
    reason="API key not set, skipping integration test that requires actual API call",
)
def test_parse_multiple_documents_batch(
    multi_page_pdf, complex_pdf, sample_image_path, results_dir
):
    # Arrange - mix of different document types
    input_files = [
        multi_page_pdf,
        complex_pdf,
        sample_image_path,
    ]

    # Act
    result_paths = parse_and_save_documents(
        input_files, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    # Assert
    assert len(result_paths) == 3

    # Check that all files were saved
    for path in result_paths:
        assert path.exists()
        assert path.suffix == ".json"

    # Verify each result has the correct structure
    file_types = []
    for i, path in enumerate(result_paths):
        with open(path) as f:
            data = json.load(f)

        parsed_doc = ParsedDocument.model_validate(data)
        file_types.append(parsed_doc.doc_type)

        # Check basic doc properties
        assert parsed_doc.markdown
        assert len(parsed_doc.chunks) > 0

        # Check for non-error chunks
        non_error_chunks = [c for c in parsed_doc.chunks]
        assert len(non_error_chunks) > 0, f"Document {i} has only error chunks"

    # Make sure we got the expected mix of document types
    assert "pdf" in file_types
    assert "image" in file_types


def test_parse_documents_error_handling_mixed_valid_invalid(
    sample_image_path, results_dir
):
    # Test parsing a mix of valid and invalid document paths
    input_files = [
        sample_image_path,  # Valid image
        "/path/to/nonexistent.pdf",  # Invalid path
    ]

    # Should raise FileNotFoundError for the invalid file
    with pytest.raises(FileNotFoundError):
        parse_and_save_documents(input_files, result_save_dir=results_dir)

def test_field_extraction_with_results_save_dir(sample_pdf_path, results_dir):
    class ExtractedFields(BaseModel):
        employee_name: str = Field(description="")

    schema = {
        "properties": {
            "employee_name": {
                "description": "",
                "title": "Employee Name", 
                "type": "string"
            }
        },
        "required": ["employee_name"],
        "title": "ExtractedFields",
        "type": "object"
    }
    
    result1 = parse(sample_pdf_path, extraction_schema=schema, result_save_dir=results_dir)
    result2 = parse(sample_pdf_path, extraction_model=ExtractedFields, result_save_dir=results_dir)
    
    assert hasattr(result1[0], 'extraction')
    assert hasattr(result1[0], 'extraction_metadata')
    assert hasattr(result2[0], 'extraction')
    assert hasattr(result2[0], 'extraction_metadata')
    assert 'employee_name' in result1[0].extraction
    assert 'employee_name' in result1[0].extraction_metadata
    assert hasattr(result2[0].extraction, 'employee_name')
    assert hasattr(result2[0].extraction_metadata, 'employee_name')

def test_parse_pdf_chunks_have_sequential_pages(sample_pdf_path, results_dir):
    # Test that PDF chunks are correctly ordered by page
    result_paths = parse_and_save_documents(
        [sample_pdf_path], result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    assert len(result_paths) == 1
    result_path = result_paths[0]

    with open(result_path) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)

    # Collect all page numbers from chunks
    all_page_numbers = []
    for chunk in parsed_doc.chunks:
        for grounding in chunk.grounding:
            all_page_numbers.append(grounding.page)

    # Pages should be in order and start from 0
    unique_pages = sorted(set(all_page_numbers))
    assert unique_pages[0] == 0
    assert unique_pages == list(range(len(unique_pages)))  # Sequential pages


def test_parse_documents_markdown_not_empty(sample_image_path, results_dir):
    # Test that parsed documents have non-empty markdown
    result_paths = parse_and_save_documents(
        [sample_image_path], result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    assert len(result_paths) == 1
    result_path = result_paths[0]

    with open(result_path) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)

    # Markdown should not be empty for a valid document
    assert parsed_doc.markdown.strip() != ""
    assert len(parsed_doc.markdown) > 0


def test_parse_documents_chunk_ids_unique(multi_page_pdf, results_dir):
    # Test that all chunk IDs within a document are unique
    result_paths = parse_and_save_documents(
        [multi_page_pdf], result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    assert len(result_paths) == 1
    result_path = result_paths[0]

    with open(result_path) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)

    # Collect all chunk IDs
    chunk_ids = [chunk.chunk_id for chunk in parsed_doc.chunks]

    # All chunk IDs should be unique
    assert len(chunk_ids) == len(set(chunk_ids)), "Found duplicate chunk IDs"

    # All chunk IDs should be non-empty strings
    for chunk_id in chunk_ids:
        assert isinstance(chunk_id, str)
        assert len(chunk_id) > 0


def test_parse_and_save_documents_with_invalid_file(sample_pdf_path, results_dir):
    # Arrange
    input_files = [
        sample_pdf_path.parent / "invalid.pdf",  # Non-existent file
        sample_pdf_path,
    ]

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        parse_and_save_documents(input_files, result_save_dir=results_dir)


def test_parse_documents_grounding_boxes_valid(sample_image_path, results_dir):
    # Test that all grounding boxes have valid coordinates
    result_paths = parse_and_save_documents(
        [sample_image_path], result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    assert len(result_paths) == 1
    result_path = result_paths[0]

    with open(result_path) as f:
        data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(data)

    for chunk in parsed_doc.chunks:
        for grounding in chunk.grounding:
            box = grounding.box

            # All coordinates should be between 0 and 1
            assert 0 <= box.l <= 1, f"Invalid left coordinate: {box.l}"
            assert 0 <= box.t <= 1, f"Invalid top coordinate: {box.t}"
            assert 0 <= box.r <= 1, f"Invalid right coordinate: {box.r}"
            assert 0 <= box.b <= 1, f"Invalid bottom coordinate: {box.b}"

            # Right should be greater than left, bottom should be greater than top
            assert box.r > box.l, f"Right ({box.r}) should be > left ({box.l})"
            assert box.b > box.t, f"Bottom ({box.b}) should be > top ({box.t})"


def test_parse_with_document_bytes(sample_pdf_path, results_dir):
    with open(sample_pdf_path, "rb") as f:
        doc_bytes = f.read()

    # Act
    result_docs = parse(
        doc_bytes, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    # Assert
    assert len(result_docs) == 1
    result_path = result_docs[0].result_path
    assert result_path.exists()

    # Verify the saved JSON can be loaded and has expected structure
    with open(result_path) as f:
        result_data = json.load(f)

    parsed_doc = ParsedDocument.model_validate(result_data)
    assert parsed_doc.markdown
    assert len(parsed_doc.chunks) > 0
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx == 3


def test_parse_with_image_bytes(sample_image_path, results_dir):
    with open(sample_image_path, "rb") as f:
        doc_bytes = f.read()

    result = parse(
        doc_bytes, result_save_dir=results_dir, grounding_save_dir=results_dir
    )

    assert len(result) == 1
    parsed_doc = result[0]

    # Check basic structure
    assert parsed_doc.doc_type == "image"
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx == 0
    assert parsed_doc.markdown
    assert len(parsed_doc.chunks) > 0

    # Check chunk structure
    for chunk in parsed_doc.chunks:
        assert chunk.text
        assert len(chunk.grounding) > 0
        for grounding in chunk.grounding:
            assert grounding.page == 0
            if grounding.box:
                assert 0 <= grounding.box.l <= 1
                assert 0 <= grounding.box.t <= 1
                assert 0 <= grounding.box.r <= 1
                assert 0 <= grounding.box.b <= 1


def test_parse_with_extraction_model(sample_image_path):
    class SampleFormFields(BaseModel):
        eye_color: str = Field(description="Eye color")

    result_path = parse(sample_image_path, extraction_model=SampleFormFields)

    extraction_results = result_path[0].extraction
    assert extraction_results.eye_color == "green"


def test_extraction_metadata_simple(sample_image_path):
    class SampleFormFields(BaseModel):
        eye_color: str = Field(description="Eye color")

    result = parse(sample_image_path, extraction_model=SampleFormFields)

    assert len(result) == 1
    parsed_doc = result[0]
    assert parsed_doc.extraction is not None
    assert isinstance(parsed_doc.extraction, SampleFormFields)

    assert hasattr(parsed_doc.extraction_metadata, "eye_color")
    assert isinstance(parsed_doc.extraction_metadata.eye_color, MetadataType)
    assert hasattr(parsed_doc.extraction_metadata.eye_color, "chunk_references")
    assert isinstance(parsed_doc.extraction_metadata.eye_color.chunk_references, list)


def test_extraction_metadata_nested(sample_pdf_path):
    class Invoices(BaseModel):
        invoices_by_date: int = Field(description="Invoices by date")
        trans_date: str = Field(description="Transaction date")

    class Type(BaseModel):
        invoices_by_type: int = Field(description="Invoices by type")
        trans_type: str = Field(description="Transaction type")

    class Amount(BaseModel):
        invoices_by_trans_amount: int = Field(
            description="Invoices by transaction amount"
        )
        trans_amount: str = Field(description="Transaction amount")

    class SampleBookmarkFile(BaseModel):
        invoices: Invoices
        type: Type
        amount: Amount

    class SampleDataFile(BaseModel):
        invoices: Invoices
        type: Type
        amount: Amount

    class Files(BaseModel):
        sample_bookmark_file: SampleBookmarkFile
        sample_data_file: SampleDataFile

    def check_structure_matches(obj, model_class, is_metadata=False):
        """
        Recursively verify that obj has the same structure as model_class.
        If is_metadata=True, leaf values should be dict[str, list[str]],
        otherwise they should match the model's field types.
        """
        field_annotations = model_class.model_fields

        for field_name, field_info in field_annotations.items():
            assert hasattr(obj, field_name), f"Missing field: {field_name}"

            field_value = getattr(obj, field_name)
            field_type = field_info.annotation

            if hasattr(field_type, "__bases__") and BaseModel in field_type.__bases__:
                if is_metadata:
                    # Recursively check the nested structure
                    check_structure_matches(field_value, field_type, is_metadata=True)
                else:
                    # For extraction, should be actual model instances
                    assert isinstance(
                        field_value, field_type
                    ), f"Field {field_name} should be {field_type}"
                    check_structure_matches(field_value, field_type, is_metadata=False)
            else:
                # This is a leaf field
                if is_metadata:
                    assert isinstance(
                        field_value, MetadataType
                    ), f"Leaf field {field_name} should be MetadataType in metadata"
                    if hasattr(field_value, "value") and field_value.value != None:
                        assert isinstance(
                            field_value.value, field_type
                        ), f"Field {field_name}.value should be {field_type}"
                else:
                    # For extraction, check against the actual field type
                    assert isinstance(
                        field_value, field_type
                    ), f"Field {field_name} should be {field_type}"

    result = parse(sample_pdf_path, extraction_model=Files)

    assert len(result) == 1
    parsed_doc = result[0]

    # Check that extraction has the exact same type as Files
    assert parsed_doc.extraction is not None
    assert isinstance(parsed_doc.extraction, Files)
    check_structure_matches(parsed_doc.extraction, Files, is_metadata=False)

    # Check that extraction_metadata has the same structure but with dict[str, list[str]] leaves
    assert parsed_doc.extraction_metadata is not None
    check_structure_matches(parsed_doc.extraction_metadata, Files, is_metadata=True)


def test_extraction_schema_simple(sample_image_path):
    extraction_schema = {
        "type": "object",
        "properties": {"eye_color": {"type": "string", "description": "Eye color"}},
    }

    result = parse(sample_image_path, extraction_schema=extraction_schema)

    assert len(result) == 1
    extraction_result = result[0]
    assert extraction_result.extraction is not None
    assert isinstance(extraction_result.extraction, dict)
    assert extraction_result.extraction["eye_color"] == "green"
    assert isinstance(extraction_result.extraction_metadata, dict)


def test_extraction_schema_nested(sample_pdf_path):
    extraction_schema = {
        "type": "object",
        "properties": {
            "sample_bookmark_file": {
                "type": "object",
                "properties": {
                    "invoices": {
                        "type": "object",
                        "properties": {
                            "invoices_by_date": {
                                "type": "integer",
                                "description": "Invoices by date",
                            },
                            "trans_date": {
                                "type": "string",
                                "description": "Transaction date",
                            },
                        },
                    },
                    "type": {
                        "type": "object",
                        "properties": {
                            "invoices_by_type": {
                                "type": "integer",
                                "description": "Invoices by type",
                            },
                            "trans_type": {
                                "type": "string",
                                "description": "Transaction type",
                            },
                        },
                    },
                    "amount": {
                        "type": "object",
                        "properties": {
                            "invoices_by_trans_amount": {
                                "type": "integer",
                                "description": "Invoices by transaction amount",
                            },
                            "trans_amount": {
                                "type": "string",
                                "description": "Transaction amount",
                            },
                        },
                    },
                },
            },
            "sample_data_file": {
                "type": "object",
                "properties": {
                    "invoices": {
                        "type": "object",
                        "properties": {
                            "invoices_by_date": {
                                "type": "integer",
                                "description": "Invoices by date",
                            },
                            "trans_date": {
                                "type": "string",
                                "description": "Transaction date",
                            },
                        },
                    },
                    "type": {
                        "type": "object",
                        "properties": {
                            "invoices_by_type": {
                                "type": "integer",
                                "description": "Invoices by type",
                            },
                            "trans_type": {
                                "type": "string",
                                "description": "Transaction type",
                            },
                        },
                    },
                    "amount": {
                        "type": "object",
                        "properties": {
                            "invoices_by_trans_amount": {
                                "type": "integer",
                                "description": "Invoices by transaction amount",
                            },
                            "trans_amount": {
                                "type": "string",
                                "description": "Transaction amount",
                            },
                        },
                    },
                },
            },
        },
    }

    result = parse(sample_pdf_path, extraction_schema=extraction_schema)

    assert len(result) == 1
    extraction_result = result[0]
    assert extraction_result.extraction is not None
    assert isinstance(extraction_result.extraction, dict)
    assert "sample_bookmark_file" in extraction_result.extraction
    assert "sample_data_file" in extraction_result.extraction
    assert (
        "invoices_by_date"
        in extraction_result.extraction["sample_bookmark_file"]["invoices"]
    )
    assert (
        "invoices_by_type"
        in extraction_result.extraction["sample_bookmark_file"]["type"]
    )
    assert (
        "invoices_by_trans_amount"
        in extraction_result.extraction["sample_bookmark_file"]["amount"]
    )
    assert (
        "invoices_by_date"
        in extraction_result.extraction["sample_data_file"]["invoices"]
    )
    assert (
        "invoices_by_type" in extraction_result.extraction["sample_data_file"]["type"]
    )
    assert (
        "invoices_by_trans_amount"
        in extraction_result.extraction["sample_data_file"]["amount"]
    )
    assert isinstance(extraction_result.extraction_metadata, dict)


@pytest.mark.skipif(
    not get_settings().vision_agent_api_key,
    reason="API key not set, skipping integration test that requires actual API call",
)
def test_xml_output_integration(sample_image_path, results_dir):
    """Integration test for XML output functionality."""
    # Test with XML output enabled
    xml_config = ParseConfig(output_xml=True)
    xml_result = parse(sample_image_path, result_save_dir=results_dir, config=xml_config)
    
    # Test with JSON output (default)  
    json_config = ParseConfig(output_xml=False)
    json_result = parse(sample_image_path, result_save_dir=results_dir, config=json_config)
    
    # Both should return valid ParsedDocument objects with same structure
    assert len(xml_result) == 1
    assert len(json_result) == 1
    
    xml_doc = xml_result[0]
    json_doc = json_result[0]
    
    # Core structure should be identical
    assert xml_doc.doc_type == json_doc.doc_type
    assert xml_doc.start_page_idx == json_doc.start_page_idx 
    assert xml_doc.end_page_idx == json_doc.end_page_idx
    assert len(xml_doc.chunks) == len(json_doc.chunks)
    assert len(xml_doc.errors) == len(json_doc.errors)
    
    # Chunk structure should match
    for xml_chunk, json_chunk in zip(xml_doc.chunks, json_doc.chunks):
        assert xml_chunk.chunk_type == json_chunk.chunk_type
        assert len(xml_chunk.grounding) == len(json_chunk.grounding)