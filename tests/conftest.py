import tempfile
from pathlib import Path

import httpx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    ParsedDocument,
)


@pytest.fixture(scope="session")
def sample_pdf_path():
    # Uncomment below to test a more complex pdf
    # file_url = "https://upload.wikimedia.org/wikipedia/commons/8/85/I-20-sample.pdf"
    file_url = "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
    file_path = Path(__file__).parent.parent / "temp_test_data" / "sample.pdf"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        file_path.unlink()

    with httpx.stream("GET", file_url) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)

    return file_path


@pytest.fixture(scope="session")
def sample_image_path():
    file_url = "https://upload.wikimedia.org/wikipedia/commons/3/34/Sample_web_form.png"
    file_path = Path(__file__).parent.parent / "temp_test_data" / "sample.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        file_path.unlink()

    with httpx.stream("GET", file_url) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)

    return file_path


@pytest.fixture
def results_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return results


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def multi_page_pdf(temp_dir):
    """Create a multi-page PDF with text."""
    pdf_path = temp_dir / "multi_page.pdf"
    num_pages = 5
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    for i in range(num_pages):
        elements.append(
            Paragraph(
                f"This is page {i + 1} of a multi-page document.", styles["Normal"]
            )
        )
        if i < num_pages - 1:  # Don't add page break after the last page
            elements.append(PageBreak())

    doc.build(elements)
    return pdf_path


@pytest.fixture
def complex_pdf(temp_dir):
    """Create a complex PDF with text, table, and image."""
    # First create a simple test image
    from PIL import Image as PILImage
    from PIL import ImageDraw

    img_path = temp_dir / "complex_image.png"
    img = PILImage.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], outline=(0, 0, 0), fill=(200, 200, 200))
    draw.text((40, 90), "Complex PDF", fill=(0, 0, 0))
    img.save(img_path)

    # Now create PDF with mixed content
    pdf_path = temp_dir / "complex.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("This is a complex PDF with multiple elements", styles["Heading1"]),
        Paragraph("This page contains text, a table, and an image.", styles["Normal"]),
        Table(
            data=[
                ["Type", "Description"],
                ["Text", "Regular paragraphs"],
                ["Table", "Structured data"],
                ["Image", "Visual element"],
            ],
            style=TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 1, (0, 0, 0)),
                    ("BACKGROUND", (0, 0), (-1, 0), (0.8, 0.8, 0.8)),
                ]
            ),
        ),
        Paragraph("Below is an image:", styles["Normal"]),
        Image(str(img_path), width=300, height=200),
        PageBreak(),
        Paragraph("This is page 2 of the complex document", styles["Heading2"]),
        Paragraph("This demonstrates a multi-page complex document.", styles["Normal"]),
    ]

    doc.build(elements)
    return pdf_path


@pytest.fixture
def mock_parsed_document():
    """Return a mock ParsedDocument object."""
    return ParsedDocument(
        markdown="# Test Document\n\nThis is a test document.",
        chunks=[
            Chunk(
                text="Test Document",
                chunk_type=ChunkType.text,
                chunk_id="11111",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            ),
            Chunk(
                text="This is a test document.",
                chunk_type=ChunkType.text,
                chunk_id="22222",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                    )
                ],
            ),
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )


@pytest.fixture
def mock_multi_page_parsed_document():
    """Return a mock ParsedDocument object for a multi-page document."""
    return ParsedDocument(
        markdown="# Multi-page Document\n\nPage 1 content.\n\n## Page 2\n\nPage 2 content.\n\nPage 3 content.",
        chunks=[
            Chunk(
                text="Multi-page Document",
                chunk_type=ChunkType.text,
                chunk_id="11111",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            ),
            Chunk(
                text="Page 1 content.",
                chunk_type=ChunkType.text,
                chunk_id="22222",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                    )
                ],
            ),
            Chunk(
                text="Page 2",
                chunk_type=ChunkType.text,
                chunk_id="33333",
                grounding=[
                    ChunkGrounding(
                        page=1, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            ),
            Chunk(
                text="Page 2 content.",
                chunk_type=ChunkType.figure,
                chunk_id="44444",
                grounding=[
                    ChunkGrounding(
                        page=1, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                    )
                ],
            ),
            Chunk(
                text="Page 3 content.",
                chunk_type=ChunkType.text,
                chunk_id="55555",
                grounding=[
                    ChunkGrounding(
                        page=2, box=ChunkGroundingBox(l=0.1, t=0.3, r=0.9, b=0.4)
                    )
                ],
            ),
        ],
        start_page_idx=0,
        end_page_idx=2,
        doc_type="pdf",
    )
