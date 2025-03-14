# agentic-doc

The LandingAI [Agentic Document Extraction](https://va.landing.ai/demo/doc-extraction) tool extracts structured information from visually complex documents with text, tables, pictures, charts, and other information. The API returns the extracted data in a hierarchical format and pinpoints the exact location of each element.

This Python library wraps around the [Agentic Document Extraction](https://va.landing.ai/demo/doc-extraction) API to add more features and support to the document extraction process. For example, using this library allows you to process much longer documents.

Learn more about the Agentic Document Extraction API [here](https://support.landing.ai/docs/document-extraction).

## Quick Start

### Installation

```bash
pip install git+https://github.com/landing-ai/agentic-doc.git
```

### Requirements
- Python version 3.9, 3.10, or 3.11
- LandingAI agentic AI API key (get the key [here](https://va.landing.ai/account/api-key))

### Set the API Key as an Environment Variable
After you get the LandingAI agentic AI API key, set the key as an environment variable (or put it in a `.env` file):

```bash
export VISION_AGENT_API_KEY=<your-api-key>
```

### Supported Files
The library can extract data from:
- PDFs (any length)
- Images that are supported by OpenCV (the `agentic-doc` library imports the `cv2` library)

### Basic Usage

#### Extract Data from One Document
Run the following script to extract data from one document and return the results in both markdown and structured chunks.

```python
from agentic_doc.parse import parse_documents

results = parse_documents(["path/to/image.png"])
parsed_doc = results[0]
print(parsed_doc.markdown)  # Get the extracted data as markdown
print(parsed_doc.chunks)  # Get the extracted data as structured chunks of content
```

#### Extract Data from Multiple Documents and Save the Results to a Directory
Run the following script to extract data from multiple documents. The results will be saved as structured chunks in JSON files in the specified directory.

```python
from agentic_doc.parse import parse_and_save_documents

file_paths = ["path/to/your/document1.pdf", "path/to/another/document2.pdf"]
result_save_dir = "path/to/save/results"

result_paths = parse_and_save_documents(file_paths, result_save_dir=result_save_dir)
# result_paths: ["path/to/save/results/document1_20250313_070305.json", "path/to/save/results/document2_20250313_070408.json"]
```


## Main Features

With this library, you can do things that are otherwise hard to do with the REST API alone.
Below are some of the highlighted features.

### Parse a large PDF file (e.g. 500 pages)

A single REST API call can only handle up to 2 pages at a time. This library will automatically split a large file into multiple calls, using a thread pool to process the calls in parallel, and stitching the results back together as a single result.


### Parse multiple documents in a batch

You can parse multiple documents in a single function call with this library. The library will process those documents in parallel.


### Automatically handle API errors and rate limits with retries

The REST API endpoint imposes rate limits per API key. This library automatically handles the rate limit error or other intermittent HTTP errors with retries.

See [Error Handling](#error-handling) and [Configuration Options](#configuration-options) for more details.


### File type and size support

PDF and common image files are supported.
The library should be able to process a single PDF file of 1000+ pages.

NOTE: if anything is not working, please let us know by opening an issue or a PR.


### Error Handling

The library implements a retry mechanism for handling API failures:

- Retries are performed for HTTP status codes: 408, 429, 502, 503, 504
- Exponential backoff with jitter is used for retry wait time
- Initial retry wait time is 1 second, increasing exponentially
- Retry will stop after `max_retries` attempts. Exceeding the limit will raise an exception and result in a failure for this request.
- Retry wait time is capped at `max_retry_wait_time` seconds
- Jitter of 10 seconds is added to prevent thundering herd


## Configuration Options

The library uses a [`Settings` object](./agentic_doc/config.py) to manage configuration. You can customize these settings either through environment variables or a `.env` file:

Below is an example `.env` file:

```bash
MAX_WORKERS=4 # Number of worker threads for parallel processing for each file, defaults to 10
MAX_RETRIES=80 # Maximum number of retry attempts for failed intermittent requests, defaults to 100
MAX_RETRY_WAIT_TIME=30 # Maximum wait time in seconds for each retry, defaults to 60
RETRY_LOGGING_STYLE=log_msg # Logging style for retry, defaults to log_msg
```

### Setting `MAX_WORKERS`

Increasing `MAX_WORKERS` will increase the number of concurrent requests, which can speed up the processing of large files if you have a enough API rate limit. Otherwise, you hit the rate limit error and the library just keeps retrying for you.

The best `MAX_WORKERS` value depends on your API rate limit and the latency of each REST API call. For example, if your account has a rate limit of 5 requests per minute, and each REST API call takes on average 60 seconds to complete, then `MAX_WORKERS` should be set to 5.

NOTE: you can find out your REST API latency from logs, and reach out to us if you want to increase your rate limit.


### Setting `RETRY_LOGGING_STYLE`

This setting controls how the library logs the retry attempts.

- `log_msg`: Log the retry attempts as a log messages. Each attempt is logged as a separate message.
- `inline_blobk`: Print a yellow progress block ('█') on the same line. Each block represents one retry attempt. Choose this if you don't want to see the verbose retry logging message and still want to keep an eye on the number of retries has been made.
- `none`: Do not log the retry attempts.


## API Reference

### Main Functions

#### `parse_documents(file_paths: list[str | Path]) -> list[ParsedDocument]`

Parse multiple documents and return their parsed results.

- **Parameters:**
  - `file_paths`: List of paths to documents (PDF or images)
- **Returns:**
  - List of `ParsedDocument` objects containing parsed results
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist

#### `parse_and_save_documents(file_paths: list[str | Path], *, result_save_dir: str | Path) -> list[Path]`

Parse multiple documents and save results to specified directory.

- **Parameters:**
  - `file_paths`: List of paths to documents
  - `result_save_dir`: Directory to save parsed results
- **Returns:**
  - A list of json file paths to the saved results. The file paths are sorted by the order of the input file paths. The file name is the original file name with a timestamp appended. E.g. "document.pdf" -> "document_20250313_070305.json".
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist

#### `parse_and_save_document(file_path: str | Path, *, result_save_dir: str | Path = None) -> Path | ParsedDocument`

Parse a single document and optionally save results.

- **Parameters:**
  - `file_path`: Path to document
  - `result_save_dir`: Optional directory to save results
- **Returns:**
  - If `result_save_dir` provided: Path to saved result file
  - If no `result_save_dir`: ParsedDocument object
- **Raises:**
  - `FileNotFoundError`: If input file doesn't exist
  - `ValueError`: If file type is not supported

### Result Schema

#### ParsedDocument

Represents a parsed document with the following attributes:

- `markdown`: str - Markdown representation of the document
- `chunks`: list[Chunk] - List of parsed content chunks, sorted by page index, then the layout of the content in the page
- `start_page_idx`: Optional[int] - Starting page index for PDFs
- `end_page_idx`: Optional[int] - Ending page index for PDFs
- `doc_type`: Literal["pdf", "image"] - Type of document

#### Chunk

Represents a parsed content chunk with the following attributes:

- `text`: str - Extracted text content
- `grounding`: list[Grounding] - List of content locations in document
- `chunk_type`: Literal["text", "error"] - Type of chunk
- `chunk_id`: Optional[str] - ID of the chunk
