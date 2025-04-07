[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![ci_status](https://github.com/landing-ai/agentic-doc/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/agentic-doc.svg)](https://badge.fury.io/py/agentic-doc)

# Agentic Document Extraction Python Library

The LandingAI [Agentic Document Extraction tool](https://va.landing.ai/demo/doc-extraction) extracts structured information from visually complex documents with text, tables, pictures, charts, and other information. The API returns the extracted data in a hierarchical format and pinpoints the exact location of each element.

This `agentic-doc` Python library wraps around the Agentic Document Extraction API to add more features and support to the document extraction process. For example, using this library allows you to process much longer documents.

For advanced users or for troubleshooting purposes, you can refer to the Agentic Document Extraction API [here](https://support.landing.ai/docs/document-extraction).

## Quick Start

### Installation

```bash
pip install agentic-doc
```

### Requirements
- Python version 3.9, 3.10, 3.11 or 3.12
- LandingAI agentic AI API key (get the key [here](https://va.landing.ai/account/api-key))

### Set the API Key as an Environment Variable
After you get the LandingAI agentic AI API key, set the key as an environment variable (or put it in a `.env` file):

```bash
export VISION_AGENT_API_KEY=<your-api-key>
```

### Supported Files
The library can extract data from:
- PDFs (any length)
- Images that are supported by OpenCV-Python (i.e. the `cv2` library)
- URLs pointing to PDF or image files

### Basic Usage

#### Extract Data from One Document
Run the following script to extract data from one document and return the results in both markdown and structured chunks.

```python
from agentic_doc.parse import parse_documents

# Parse a local file
results = parse_documents(["path/to/image.png"])
parsed_doc = results[0]
print(parsed_doc.markdown)  # Get the extracted data as markdown
print(parsed_doc.chunks)  # Get the extracted data as structured chunks of content

# Parse a document from a URL
results = parse_documents(["https://example.com/document.pdf"])
parsed_doc = results[0]
print(parsed_doc.markdown)
```

#### Extract Data from Multiple Documents and Save the Results to a Directory
Run the following script to extract data from multiple documents. The results will be saved as structured chunks in JSON files in the specified directory.

```python
from agentic_doc.parse import parse_and_save_documents

# Parse local files
file_paths = ["path/to/your/document1.pdf", "path/to/another/document2.pdf"]
result_save_dir = "path/to/save/results"

result_paths = parse_and_save_documents(file_paths, result_save_dir=result_save_dir)
# result_paths: ["path/to/save/results/document1_20250313_070305.json", "path/to/save/results/document2_20250313_070408.json"]

# Parse documents from URLs
urls = ["https://example.com/doc1.pdf", "https://example.com/doc2.pdf"]
result_paths = parse_and_save_documents(urls, result_save_dir=result_save_dir)
```

## Why Use It?

- **Simplified Setup:** No need to manage API keys or handle low-level REST calls.
- **Automatic Large File Processing:** Splits large PDFs into manageable parts and processes them in parallel.
- **Built-In Error Handling:** Automatically retries requests with exponential backoff and jitter for common HTTP errors.
- **Parallel Processing:** Efficiently parse multiple documents at once with configurable parallelism.

## Main Features

With this library, you can do things that are otherwise hard to do with the Agentic Document Extraction API alone.
This section describes some of the key features this library offers.

### Parse Large PDF Files

**A single REST API call can only handle up to 2 pages at a time.** This library automatically splits a large PDF into multiple calls, uses a thread pool to process the calls in parallel, and stitches the results back together as a single result.

We've used this library to successfully parse PDFs that are 1000+ pages long.

### Parse Multiple Files in a Batch

You can parse multiple files in a single function call with this library. The library processes files in parallel.

> **NOTE:** You can change the parallelism by setting the `batch_size` setting.

### Save Groundings as Images

The library can extract and save the visual regions (groundings) of the document where each chunk of content was found. This is useful for visualizing exactly what parts of the document were extracted and for debugging extraction issues.

Each grounding represents a bounding box in the original document, and the library can save these regions as individual PNG images. The images are organized by page number and chunk ID.

Here's how to use this feature:

```python
from agentic_doc.parse import parse_documents

# Save groundings when parsing a document
results = parse_documents(
    ["path/to/document.pdf"],
    grounding_save_dir="path/to/save/groundings"
)

# The grounding images will be saved to:
# path/to/save/groundings/document_TIMESTAMP/page_X/CHUNK_ID_Y.png
# Where X is the page number, CHUNK_ID is the unique ID of each chunk,
# and Y is the index of the grounding within the chunk

# Each chunk's grounding in the result will have the image_path set
for chunk in results[0].chunks:
    for grounding in chunk.grounding:
        if grounding.image_path:
            print(f"Grounding saved to: {grounding.image_path}")
```

This feature works with all parsing functions: `parse_documents`, `parse_and_save_documents`, and `parse_and_save_document`.

### Visualize Parsing Result

The library provides a visualization utility that creates annotated images showing where each chunk of content was extracted from the document. This is useful for:
- Verifying the accuracy of the extraction
- Debugging extraction issues

Here's how to use the visualization feature:

```python
from agentic_doc.parse import parse_documents
from agentic_doc.utils import viz_parsed_document
from agentic_doc.config import VisualizationConfig

# Parse a document
results = parse_documents(["path/to/document.pdf"])
parsed_doc = results[0]

# Create visualizations with default settings
# The output images have a PIL.Image.Image type
images = viz_parsed_document(
    "path/to/document.pdf",
    parsed_doc,
    output_dir="path/to/save/visualizations"
)

# Or customize the visualization appearance
viz_config = VisualizationConfig(
    thickness=2,  # Thicker bounding boxes
    text_bg_opacity=0.8,  # More opaque text background
    font_scale=0.7,  # Larger text
    # Custom colors for different chunk types
    color_map={
        ChunkType.TITLE: (0, 0, 255),  # Red for titles
        ChunkType.TEXT: (255, 0, 0),  # Blue for regular text
        # ... other chunk types ...
    }
)

images = viz_parsed_document(
    "path/to/document.pdf",
    parsed_doc,
    output_dir="path/to/save/visualizations",
    viz_config=viz_config
)

# The visualization images will be saved as:
# path/to/save/visualizations/document_viz_page_X.png
# Where X is the page number
```

The visualization shows:
- Bounding boxes around each extracted chunk
- Chunk type and index labels
- Different colors for different types of content (titles, text, tables, etc.)
- Semi-transparent text backgrounds for better readability

### Automatically Handle API Errors and Rate Limits with Retries

The REST API endpoint imposes rate limits per API key. This library automatically handles the rate limit error or other intermittent HTTP errors with retries.

For more information, see [Error Handling](#error-handling) and [Configuration Options](#configuration-options).

### Error Handling

This library implements a retry mechanism for handling API failures:

- Retries are performed for these HTTP status codes: 408, 429, 502, 503, 504.
- Exponential backoff with jitter is used for retry wait time.
- The initial retry wait time is 1 second, which increases exponentially.
- Retry will stop after `max_retries` attempts. Exceeding the limit raises an exception and results in a failure for this request.
- Retry wait time is capped at `max_retry_wait_time` seconds.
- Retries include a random jitter of up to 10 seconds to distribute requests and prevent the thundering herd problem.

### Parsing Errors

If the REST API encounters an unrecoverable error during parsing, the library includes an [error chunk](./agentic_doc/common.py#L45) in the final result for the affected page.
Each error chunk contains the error message and corresponding page index.
Error chunks can be identified in the `ParsedDocument` by checking for `chunk_type=ChunkType.error`.

## Configuration Options

The library uses a [`Settings`](./agentic_doc/config.py) object to manage configuration. You can customize these settings either through environment variables or a `.env` file:

Below is an example `.env` file that customizes the configurations:

```bash
# Number of files to process in parallel, defaults to 4
BATCH_SIZE=4
# Number of threads used to process parts of each file in parallel, defaults to 5.
MAX_WORKERS=2
# Maximum number of retry attempts for failed intermittent requests, defaults to 100
MAX_RETRIES=80
# Maximum wait time in seconds for each retry, defaults to 60
MAX_RETRY_WAIT_TIME=30
# Logging style for retry, defaults to log_msg
RETRY_LOGGING_STYLE=log_msg
```

### Max Parallelism

The maximum number of parallel requests is determined by multiplying `BATCH_SIZE` × `MAX_WORKERS`.

> **NOTE:** The maximum parallelism allowed by this library is 100.

Specifically, increasing `MAX_WORKERS` can speed up the processing of large individual files, while increasing `BATCH_SIZE` improves throughput when processing multiple files.

> **NOTE:** Your job's maximum processing throughput may be limited by your API rate limit. If your rate limit isn't high enough, you may encounter rate limit errors, which the library will automatically handle through retries.

The optimal values for `MAX_WORKERS` and `BATCH_SIZE` depend on your API rate limit and the latency of each REST API call. For example, if your account has a rate limit of 5 requests per minute, and each REST API call takes approximately 60 seconds to complete, and you're processing a single large file, then `MAX_WORKERS` should be set to 5 and `BATCH_SIZE` to 1.

You can find your REST API latency in the logs. If you want to increase your rate limit, schedule a time to meet with us [here](https://scheduler.zoom.us/d/56i81uc2/landingai-document-extraction).

### Set `RETRY_LOGGING_STYLE`

The `RETRY_LOGGING_STYLE` setting controls how the library logs the retry attempts.

- `log_msg`: Log the retry attempts as a log messages. Each attempt is logged as a separate message. This is the default setting.
- `inline_block`: Print a yellow progress block ('█') on the same line. Each block represents one retry attempt. Choose this if you don't want to see the verbose retry logging message and still want to track the number of retries that have been made.
- `none`: Do not log the retry attempts.

## API Reference

### Main Functions

#### `parse_documents(documents: list[str | Path | AnyHttpUrl], *, grounding_save_dir: str | Path | None = None) -> list[ParsedDocument]`

Parse multiple documents and return their parsed results.

- **Parameters:**
  - `documents`: List of paths to documents (PDFs or images) or URLs pointing to documents
  - `grounding_save_dir`: Optional directory to save the grounding images
- **Returns:**
  - List of `ParsedDocument` objects containing parsed results
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist
  - `ValueError`: If any URL is invalid or points to an unsupported file type

#### `parse_and_save_documents(documents: list[str | Path | AnyHttpUrl], *, result_save_dir: str | Path, grounding_save_dir: str | Path | None = None) -> list[Path]`

Parse multiple documents and save results to the specified directory.

- **Parameters:**
  - `documents`: List of paths to documents or URLs pointing to documents
  - `result_save_dir`: Directory to save parsed results
  - `grounding_save_dir`: Optional directory to save the grounding images
- **Returns:**
  - A list of JSON file paths to the saved results. The file paths are sorted by the order of the input file paths. The file name is the original file name with a timestamp appended. For example, the input file "document.pdf" could have this output file: "document_20250313_070305.json".
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist
  - `ValueError`: If any URL is invalid or points to an unsupported file type

#### `parse_and_save_document(document: str | Path | AnyHttpUrl, *, result_save_dir: str | Path | None = None, grounding_save_dir: str | Path | None = None) -> Path | ParsedDocument`

Parse a single document and optionally save results.

- **Parameters:**
  - `document`: Path to document or URL pointing to a document
  - `result_save_dir`: Optional directory to save results
  - `grounding_save_dir`: Optional directory to save the grounding images
- **Returns:**
  - If `result_save_dir` provided: Path to saved result file  
  - If no `result_save_dir`: ParsedDocument object
- **Raises:**
  - `FileNotFoundError`: If input file doesn't exist  
  - `ValueError`: If file type is not supported or URL is invalid

## Result Schema

#### ParsedDocument

Represents a parsed document with the following attributes:

- `markdown`: str - Markdown representation of the document
- `chunks`: list[Chunk] - List of parsed content chunks, sorted by page index, then the layout of the content on the page
- `start_page_idx`: Optional[int] - Starting page index for PDFs
- `end_page_idx`: Optional[int] - Ending page index for PDFs
- `doc_type`: Literal["pdf", "image"] - Type of document

#### Chunk

Represents a parsed content chunk with the following attributes:

- `text`: str - Extracted text content
- `grounding`: list[Grounding] - List of content locations in document
- `chunk_type`: Literal["text", "error"] - Type of chunk
- `chunk_id`: Optional[str] - ID of the chunk

## Troubleshooting & FAQ

### Common Issues
- **API Key Errors:**  
  Ensure your API key is correctly set as an environment variable.
- **Rate Limits:**  
  The library automatically retries requests if you hit the API rate limit. Adjust `BATCH_SIZE` or `MAX_WORKERS` if you encounter frequent rate limit errors.
- **Parsing Failures:**  
  If a document fails to parse, an error chunk will be included in the result, detailing the error message and page index.
- **URL Access Issues:**
  If you're having trouble accessing documents from URLs, check that the URLs are publicly accessible and point to supported file types (PDF or images).

---
