from pathlib import Path

import httpx
import pytest


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
