import json
from agentic_doc.parse import parse_and_save_documents
from agentic_doc.common import ParsedDocument


def test_parse_and_save_documents_single_image(sample_image_path, results_dir):
    # Arrange
    input_file = sample_image_path

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
    assert len(parsed_doc.chunks) > 0
    assert parsed_doc.start_page_idx == 0
    assert parsed_doc.end_page_idx == 0
    assert parsed_doc.doc_type == "image"


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

    # Verify that grounding images were saved
    for chunk in parsed_doc.chunks:
        for grounding in chunk.grounding:
            assert grounding.image_path.exists()
