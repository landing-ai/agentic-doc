<div align="center">
[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![ci_status](https://github.com/landing-ai/agentic-doc/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/agentic-doc.svg)](https://badge.fury.io/py/agentic-doc)
</div>


# agentic-doc

The LandingAI [Agentic Document Extraction](https://va.landing.ai/demo/doc-extraction) tool extracts structured information from visually complex documents with text, tables, pictures, charts, and other information. The API returns the extracted data in a hierarchical format and pinpoints the exact location of each element.

This `agentic-doc` Python library wraps around the [Agentic Document Extraction](https://va.landing.ai/demo/doc-extraction) API to add more features and support to the document extraction process. For example, using this library allows you to process much longer documents.

Learn more about the Agentic Document Extraction API [here](https://support.landing.ai/docs/document-extraction).

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

With this library, you can do things that are otherwise hard to do with the Agentic Document Extraction API alone.
This section describes some of the key features this library offers.

### Parse Large PDF Files

A single REST API call can only handle up to 2 pages at a time. This library automatically splits a large PDF into multiple calls, uses a thread pool to process the calls in parallel, and stitches the results back together as a single result.

We've used this library to successfully parse PDFs that are 1000+ pages long.


### Parse Multiple Files in a Batch

You can parse multiple files in a single function call with this library. The library processes files in parallel.

NOTE: You can change the parallelism by setting the `batch_size` setting.

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


## Configuration Options

The library uses a [`Settings`](./agentic_doc/config.py) object to manage configuration. You can customize these settings either through environment variables or a `.env` file:

Below is an example `.env` file that customizes the configurations:

```bash
BATCH_SIZE=4 # Number of files to process in parallel, defaults to 4
MAX_WORKERS=2 # Number of threads used to process parts of each file in parallel, defaults to 5.
MAX_RETRIES=80 # Maximum number of retry attempts for failed intermittent requests, defaults to 100
MAX_RETRY_WAIT_TIME=30 # Maximum wait time in seconds for each retry, defaults to 60
RETRY_LOGGING_STYLE=log_msg # Logging style for retry, defaults to log_msg
```

### Set `MAX_WORKERS`

Increasing `MAX_WORKERS` increases the number of concurrent requests, which can speed up the processing of large files if you have a high enough API rate limit. Otherwise, you hit the rate limit error and the library just keeps retrying for you.

The optimal `MAX_WORKERS` value depends on your API rate limit and the latency of each REST API call. For example, if your account has a rate limit of 5 requests per minute, and each REST API call takes about 60 seconds to complete, then `MAX_WORKERS` should be set to 5.

You can find your REST API latency in the logs. If you want to increase your rate limit, schedule a time to meet with us [here](https://scheduler.zoom.us/d/56i81uc2/landingai-document-extraction).


### Set `RETRY_LOGGING_STYLE`

The `RETRY_LOGGING_STYLE` setting controls how the library logs the retry attempts.

- `log_msg`: Log the retry attempts as a log messages. Each attempt is logged as a separate message. This is the default setting.
- `inline_blobk`: Print a yellow progress block ('█') on the same line. Each block represents one retry attempt. Choose this if you don't want to see the verbose retry logging message and still want to track the number of retries has been made.
- `none`: Do not log the retry attempts.


## API Reference

### Main Functions

#### `parse_documents(file_paths: list[str | Path]) -> list[ParsedDocument]`

Parse multiple documents and return their parsed results.

- **Parameters:**
  - `file_paths`: List of paths to documents (PDFs or images)
- **Returns:**
  - List of `ParsedDocument` objects containing parsed results
- **Raises:**
  - `FileNotFoundError`: If any input file doesn't exist

#### `parse_and_save_documents(file_paths: list[str | Path], *, result_save_dir: str | Path) -> list[Path]`

Parse multiple documents and save results to the specified directory.

- **Parameters:**
  - `file_paths`: List of paths to documents
  - `result_save_dir`: Directory to save parsed results
- **Returns:**
  - A list of JSON file paths to the saved results. The file paths are sorted by the order of the input file paths. The file name is the original file name with a timestamp appended. For example, the input file "document.pdf" could have this output file: "document_20250313_070305.json".
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
