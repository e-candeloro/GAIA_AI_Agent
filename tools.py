import base64
import cmath
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

# from duckduckgo_search import DDGS
from groq import Groq
from langchain.agents import tool
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from markitdown import MarkItDown
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage  # Added for Groq summarization

load_dotenv()
### =============== MATHEMATICAL TOOLS =============== ###

SAFE_GLOBALS = {"__builtins__": {}, "math": math}
# one Groq client reused for all calls
_GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEMP_DIR = os.getenv("TEMP_DIR", "./tmp")  # Default temp directory
QUESTIONS_FILES_DIR = os.path.join(TEMP_DIR, "questions_files")
os.makedirs(QUESTIONS_FILES_DIR, exist_ok=True)


@tool
def calculator(expr: str) -> float:
    """
    Calculate a basic arithmetic or math expression.

    Accepted syntax
    ---------------
    • Literals: integers or floats (e.g. ``2``, ``3.14``)
    • Operators: ``+``, ``-``, ``*``, ``/``, ``**``
    • Unary minus (``-5``)
    • Functions/consts from ``math`` (e.g. ``sin(0.5)``, ``pi``)
    • Parentheses for grouping

    Parameters
    ----------
    expr : str
        The expression to evaluate.

    Returns
    -------
    float
        Result of the computation.

    Raises
    ------
    ValueError
        If the expression contains unsupported syntax or names.
    """
    try:
        if "." in expr or "__" in expr:
            raise ValueError("Attribute access not allowed")
        return eval(expr, SAFE_GLOBALS)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid expression '{expr}': {exc}") from exc


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """
    Adds two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a + b


@tool
def subtract(a: float, b: float) -> int:
    """
    Subtracts two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """
    Get the modulus of two numbers.
    Args:
        a (int): the first number
        b (int): the second number
    """
    return a % b


@tool
def power(a: float, b: float) -> float:
    """
    Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a**b


@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): the number to get the square root of
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)


### =============== GENERIC SEARCH TOOLS =============== ###
_SEPARATOR = "\\n\\n---\\n\\n"


def _format_docs(docs: Sequence, max_chars: int = 5000) -> str:
    """Uniformly format loader docs for the LLM / calling agent."""
    if not docs:
        return "No results found."
    chunks = []
    for doc in docs:
        meta = doc.metadata
        snippet = doc.page_content[:max_chars].strip()
        chunks.append(
            f'<Document source="{meta.get("source")}" page="{meta.get("page", "")}">\n'
            f"{snippet}\n</Document>"
        )
    return _SEPARATOR.join(chunks)


# ─────────────────────────  wiki_search  ──────────────────────────


@tool
def wiki_search(query: str) -> str:
    """Return up to 2 Wikipedia pages about *query*."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return _format_docs(docs)


# ─────────────────────────  web_search   ──────────────────────────

# ``Document`` and ``_format_docs`` are provided by the host application.


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Web search powered by Tavily.
    Requires `TAVILY_API_KEY` in env.
    Returns up to `max_results` results formatted with `_format_docs`.
    """
    docs: list[Document] = []
    try:
        tavily = TavilySearch(max_results=max_results)
        hits = tavily.run(query)  # -> list[dict]
        for hit in hits:
            docs.append(
                Document(
                    page_content=hit.get("snippet", ""),
                    metadata={"source": hit.get("url"), "page": ""},
                )
            )
    except Exception as exc:
        docs.append(
            Document(
                page_content=f"[web_search error] {exc}",
                metadata={"source": "tavily", "page": ""},
            )
        )
    return _format_docs(docs)


@tool
def comprehensive_web_search(query: str, num_pages_to_browse: int = 3) -> str:
    """
    Performs a web search using Tavily, then downloads, reads, and summarizes content
    from the top N pages to provide a comprehensive answer to the query.
    Requires TAVILY_API_KEY and GROQ_API_KEY in env.

    Args:
        query (str): The search query.
        num_pages_to_browse (int): The number of top search results to browse for content. Defaults to 3.

    Returns:
        str: A compiled string of relevant information from the browsed pages,
             or an error message if issues occur.
    """
    all_processed_contents = []
    try:
        # Fetch a bit more to choose from
        tavily = TavilySearch(max_results=num_pages_to_browse + 2)
        # Use .search to get structured results
        hits = tavily.search(query=query)

        if not hits:
            return "No search results found by Tavily."

        urls_to_browse = [hit["url"] for hit in hits[:num_pages_to_browse]]

        for i, hit_url in enumerate(urls_to_browse):
            all_processed_contents.append(f"Browsing page {i+1}: {hit_url}")
            download_status = download_file_from_url(
                url=hit_url)  # Let the tool handle filename

            if download_status.startswith("File downloaded to "):
                local_path = download_status.replace(
                    "File downloaded to ", "").split(". You can read")[0]

                page_content = read_document(file_path=local_path)

                if page_content.startswith("[read_document error]") or page_content == "[read_document] no text found":
                    processed_content = f"Could not retrieve content from {hit_url}. Error: {page_content}"
                else:
                    # Summarize if content is too long (e.g., > 15000 chars, roughly 3k-4k tokens)
                    if len(page_content) > 15000:
                        try:
                            summarization_prompt_text = (
                                f"Given the user's search query \'{query}\', please provide a concise summary of the following text. "
                                f"Focus on the information most relevant to answering the query. The text is from {hit_url}. "
                                f"If the text seems irrelevant, state that. Output only the summary or the statement of irrelevance."
                                # Limit input to summarizer
                                f"\\n\\nText to summarize:\\n\\n{page_content[:25000]}"
                            )

                            resp = _GROQ_CLIENT.chat.completions.create(
                                model="llama3-8b-8192",  # Use a cost-effective model for summarization
                                messages=[HumanMessage(
                                    content=summarization_prompt_text)],
                                temperature=0.2,
                            )
                            summary = resp.choices[0].message.content.strip()
                            processed_content = f"Summary from {hit_url}:\\n{summary}"
                        except Exception as e:
                            processed_content = f"Could not summarize content from {hit_url}. Error: {e}. Raw content snippet (first 500 chars):\\n{page_content[:500]}"
                    else:
                        processed_content = f"Content from {hit_url}:\\n{page_content}"
                all_processed_contents.append(processed_content)
                # Consider deleting the local_path file here if it's temporary and no longer needed
                # try:
                #     os.remove(local_path)
                # except OSError:
                #     pass # Ignore if deletion fails
            else:
                all_processed_contents.append(
                    f"Failed to download {hit_url}: {download_status}")
            all_processed_contents.append("---")

    except Exception as e:
        return f"An error occurred during comprehensive web search: {str(e)}"

    return "\\n\\n".join(all_processed_contents)

# @tool
# def web_search(query: str, max_results: int = 3) -> str:
#     """
#     Return up to `max_results` DuckDuckGo search results for *query*.

#     The output is formatted by `_format_docs`, so it matches the schema your
#     other tools already use.
#     """
#     docs = []
#     with DDGS() as ddgs:
#         for hit in ddgs.text(query, max_results=max_results):
#             docs.append(
#                 Document(
#                     page_content=hit.get("body") or hit.get("snippet") or "",
#                     metadata={"source": hit.get(
#                         "href") or hit.get("url"), "page": ""},
#                 )
#             )

#     return _format_docs(docs)

# ─────────────────────────  arxiv_search ──────────────────────────


@tool
def arxiv_search(query: str) -> str:
    """Return up to 3 recent ArXiv papers about *query*."""
    docs = ArxivLoader(query=query, load_max_docs=3).load()
    return _format_docs(docs)


# ---------- Search → list of links -----------------------


@tool
def list_webpage_links(url: str, same_domain_only: bool = False) -> list[str]:
    """
    Return all unique <a href="..."> links found in the HTML at `url`.

    Parameters
    ----------
    url : str
        Page to scrape.
    same_domain_only : bool, optional
        If True, keep only links on the same domain as `url`.  Default = False.

    Returns
    -------
    list[str]
        Absolute URLs, deduplicated and sorted.
    """
    try:
        html = requests.get(url, timeout=10).text
    except Exception as exc:
        return [f"ERROR: fetch failed – {exc}"]

    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
    soup = BeautifulSoup(html, "html.parser")

    links: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"].strip()
        # Convert relative → absolute
        full = urljoin(base, href)
        if same_domain_only and urlparse(full).netloc != urlparse(url).netloc:
            continue
        links.add(full)

    return sorted(links)


### =============== DOCUMENT PROCESSING TOOLS =============== ###

# ─────────────────────────────────────────────────────────────────────────────
# MarkItDown initialisation
#   • Works out-of-the-box for PDFs, Word, PowerPoint, Excel, images, etc.
#   • If DOCINTEL_ENDPOINT is set, heavy lifting (scanned PDFs, OCR tables…)
#     is delegated to Azure Document Intelligence.
# ─────────────────────────────────────────────────────────────────────────────
_DOCINTEL_ENDPOINT = os.getenv("DOCINTEL_ENDPOINT")  # set in env if needed
_MD = MarkItDown(enable_plugins=False,
                 docintel_endpoint=_DOCINTEL_ENDPOINT or None)


@tool("read_document", return_direct=True)
def read_document(file_path: str, max_pages: Optional[int] = 10) -> str:
    """
    Extract plain text from **any** local document supported by MarkItDown
    (PDF, DOCX, PPTX, XLSX, images, .HTML,.py, etc...).

    Parameters
    ----------
    file_path : str
        Path to the document on disk.
    max_pages : int, optional
        Truncate output after this many pages/slides (only applies to
        paginated formats).  If omitted, return the full text.

    Returns
    -------
    str
        The extracted text, or an error string that starts with
        “[read_document error] …”.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"[read_document error] file not found: {file_path}"

    try:
        result = _MD.convert(str(path))
        text = result.text_content or ""

        # For paginated formats MarkItDown uses form-feed (\f) between pages
        if max_pages and max_pages > 0:
            pages = text.split("\f")
            text = "\f".join(pages[:max_pages])

        cleaned = text.strip()
        return cleaned if cleaned else "[read_document] no text found"

    except Exception as err:
        return f"[read_document error] {err}"


@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = Path(QUESTIONS_FILES_DIR)
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = Path(QUESTIONS_FILES_DIR)
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


# @tool
# def extract_text_from_image(image_path: str) -> str:
#     """
#     Extract text from an image using OCR library pytesseract (if available).
#     Args:
#         image_path (str): the path to the image file.
#     """
#     try:
#         # Open the image
#         image = Image.open(image_path)

#         # Extract text from the image
#         text = pytesseract.image_to_string(image)

#         return f"Extracted text from image:\n\n{text}"
#     except Exception as e:
#         return f"Error extracting text from image: {str(e)}"


@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"


@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"


# ─────────── vision tool ────────────────────────────
_VISION_PROMPT = """
You are a GAIA-benchmark vision assistant. Return **exactly three sections**:

1. Description – ≤40-word caption of the whole scene.
2. Objects – JSON array of {"name": str, "bbox": [x0,y0,x1,y1]} for each visible item.
3. Extracted text – verbatim text in the image or “[none]”.

No extra commentary.
""".strip()


def _b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


@tool("describe_image", return_direct=True)
def describe_image(local_path: str) -> str:
    """
    Caption a **local** image + list objects + OCR text using Groq’s
    meta-llama/llama-4-scout-17b-16e-instruct vision model.

    Steps for the agent:
      • If you only have a URL, first call `download_file_from_url`
        (that tool returns the tmp path). Then call this tool.

    Returns
    -------
    str
        Three-section GAIA-formatted answer, or an error string.
    """
    p = Path(local_path).expanduser()
    if not p.exists():
        return f"[describe_image] file not found: {local_path}"

    try:
        data_uri = f"data:image/{p.suffix.lstrip('.').lower()};base64,{_b64(p)}"
        resp = _GROQ_CLIENT.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content.strip()

    except Exception as err:
        return f"[describe_image error] {err}"


# ──────────────────────── audio tool ──────────────────────────────


@tool("transcribe_audio", return_direct=True)
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe spoken content from a local audio file using Groq Whisper-large-v3.

    Parameters
    ----------
    audio_path : str
        Path to a .wav/.mp3/.m4a/.flac file on disk.

    Returns
    -------
    str
        The plain-text transcription, or an error string if something fails.
    """
    p = Path(audio_path).expanduser()
    if not p.exists():
        return f"[transcribe_audio] file not found: {audio_path}"

    try:
        with p.open("rb") as fh:
            resp = _GROQ_CLIENT.audio.transcriptions.create(
                file=(p.name, fh.read()),
                model="whisper-large-v3",
                response_format="text",  # “text” → plain string in .text
            )
        txt = resp.text.strip() if hasattr(resp, "text") else str(resp).strip()
        return txt or "[empty transcription]"

    except Exception as err:
        return f"[transcribe_audio error] {err}"


tools = [
    # math & utils
    calculator,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    # retrieval
    wiki_search,
    web_search,
    arxiv_search,
    list_webpage_links,
    # file IO
    save_and_read_file,
    download_file_from_url,
    read_document,
    analyze_csv_file,
    analyze_excel_file,
    # vision / audio
    describe_image,
    transcribe_audio,
]


def get_tools() -> list:
    """Return the curated list of LangChain tools."""
    # Add the new tool to the list
    updated_tools = tools[:]  # Create a copy
    # Ensure it's not duplicated if script is run multiple times
    if comprehensive_web_search not in updated_tools:
        # Insert comprehensive_web_search after web_search for logical grouping
        try:
            web_search_index = updated_tools.index(web_search)
            updated_tools.insert(web_search_index + 1,
                                 comprehensive_web_search)
        except ValueError:
            # If web_search is not found for some reason, just append
            updated_tools.append(comprehensive_web_search)
    return updated_tools
