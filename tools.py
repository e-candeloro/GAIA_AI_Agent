from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
import base64
import cmath
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import pytesseract
import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.agents import tool
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEmbeddings,
                                   HuggingFaceEndpoint)
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from code_interpreter import CodeInterpreter  # <-- the class you pasted above
from image_processing import *


interpreter_instance = CodeInterpreter()

load_dotenv()
### =============== MATHEMATICAL TOOLS =============== ###

SAFE_GLOBALS = {"__builtins__": {}, "math": math}
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# one Groq client reused for all calls
_GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")  # Default temp directory
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


# ────────────────────────  generic search utils  ───────────────────────
_SEPARATOR = "\n\n---\n\n"


def _format_docs(docs: Sequence, max_chars: int = 5000) -> str:
    """Uniformly format loader docs for the LLM / calling agent."""
    if not docs:
        return "No results found."
    chunks = []
    for doc in docs:
        meta = doc.metadata
        snippet = doc.page_content[:max_chars].strip()
        chunks.append(
            f'<Document source="{meta.get("source")}" page="{meta.get("page","")}">\n'
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


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Return up to `max_results` DuckDuckGo search results for *query*.

    The output is formatted by `_format_docs`, so it matches the schema your
    other tools already use.
    """
    docs = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            docs.append(
                Document(
                    page_content=hit.get("body") or hit.get("snippet") or "",
                    metadata={"source": hit.get(
                        "href") or hit.get("url"), "page": ""},
                )
            )

    return _format_docs(docs)

# ─────────────────────────  arxiv_search ──────────────────────────


@tool
def arxiv_search(query: str) -> str:
    """Return up to 3 recent ArXiv papers about *query*."""
    docs = ArxivLoader(query=query, load_max_docs=3).load()
    return _format_docs(docs)

# ---------- 1. Search → list of links -----------------------


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


# ---------- 2. Browse → cleaned article text ----------------
@tool
def browse_webpage_link(url: str) -> str:
    """
    Download `url` and return the main readable text (no html, ads, nav bars).
    Relies on trafilatura’s article extractor.
    """
    raw = trafilatura.fetch_url(url)
    if raw is None:
        return "🛑 Could not fetch the page."

    text = trafilatura.extract(
        raw,
        include_comments=False,
        include_tables=False,
        include_links=False,
    )
    return text or "🛑 Page fetched but no readable text found."


@tool
def search_links_for_match(
    url: str,
    keyword: str,
    max_links: int = 100,
    same_domain_only: bool = True,
    case_sensitive: bool = False,
) -> list[str]:
    """
    Search the content of up to `max_links` found on a webpage, and return URLs that contain the given keyword.

    Parameters:
    ----------
    url : str
        The starting webpage to extract links from.
    keyword : str
        The keyword or phrase to match inside linked pages.
    max_links : int, optional
        Number of links to follow (default: 10).
    same_domain_only : bool, optional
        Only consider links from the same domain (default: True).
    case_sensitive : bool, optional
        Whether the keyword match should be case-sensitive.

    Returns:
    -------
    list[str]
        List of URLs whose content contains the keyword.
    """

    # Use the tool's .func() to access base function
    all_links = list_webpage_links.func(
        url=url, same_domain_only=same_domain_only)
    matched_links = []

    # Normalize keyword
    kw = keyword if case_sensitive else keyword.lower()

    for link in all_links[:max_links]:
        try:
            text = browse_webpage_link.func(link)
            if not case_sensitive:
                text = text.lower()
            if kw in text:
                matched_links.append(link)
        except Exception:
            continue

    return matched_links or ["No matches found."]


# code_interpreter_tools.py

# 1. create ONE sandbox instance that sticks around
sandbox = CodeInterpreter(
    allowed_modules=[
        "numpy", "pandas", "matplotlib", "scipy", "sklearn",
        "math", "random", "statistics", "datetime", "collections",
        "itertools", "functools", "operator", "re", "json",
        "sympy", "networkx", "nltk", "PIL", "pytesseract", "uuid", "tempfile", "requests", "urllib"
    ],  # trim to what you need
    max_execution_time=10                               # seconds; tune as required
)

# 2. LangChain tool: secure Python


@tool("run_python_safe", return_direct=True)
def run_python_safe(code: str) -> str:
    """
    Execute a Python snippet inside the CodeInterpreter sandbox.

    Args:
        code (str): Pure Python (no back-ticks); avoid long-running loops.
    Returns:
        str: Std-out if success, or formatted error message.
    """
    res = sandbox.execute_code(code, language="python")
    if res["status"] == "success":
        out = res["stdout"].strip()
        return out if out else "✅ Code ran with no output."
    return f"❌ Python error:\n{res['stderr']}"

# 3. LangChain tool: secure SQL (SQLite)


@tool("run_sql_safe", return_direct=True)
def run_sql_safe(sql: str) -> str:
    """
    Execute an SQL statement against the sandbox’s temporary SQLite database.

    Args:
        sql (str): SQL query or DDL/DML statement.
    Returns:
        str: Query head (for SELECT) or a success message.
    """
    res = sandbox.execute_code(sql, language="sql")
    if res["status"] == "success":
        if res["dataframes"]:
            # Pretty-print the first five rows of the result
            df_head = res["dataframes"][0]["head"]
            rows = [" | ".join(map(str, r.values())) for r in df_head.values()]
            return "\n".join(rows) if rows else "(0 rows)"
        return res["stdout"]
    return f"❌ SQL error:\n{res['stderr']}"


### =============== DOCUMENT PROCESSING TOOLS =============== ###


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


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    try:
        # Open the image
        image = Image.open(image_path)

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


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


### ============== IMAGE PROCESSING AND GENERATION TOOLS =============== ###


@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}

        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)

        return {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def transform_image(
    image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            img = img.resize(
                (
                    params.get("width", img.width // 2),
                    params.get("height", img.height // 2),
                )
            )
        elif operation == "rotate":
            img = img.rotate(params.get("angle", 90), expand=True)
        elif operation == "crop":
            img = img.crop(
                (
                    params.get("left", 0),
                    params.get("top", 0),
                    params.get("right", img.width),
                    params.get("bottom", img.height),
                )
            )
        elif operation == "flip":
            if params.get("direction", "horizontal") == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif operation == "adjust_brightness":
            img = ImageEnhance.Brightness(
                img).enhance(params.get("factor", 1.5))
        elif operation == "adjust_contrast":
            img = ImageEnhance.Contrast(img).enhance(params.get("factor", 1.5))
        elif operation == "blur":
            img = img.filter(ImageFilter.GaussianBlur(params.get("radius", 2)))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif operation == "grayscale":
            img = img.convert("L")
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"transformed_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def draw_on_image(
    image_base64: str, drawing_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Draw shapes (rectangle, circle, line) or text onto an image.
    Args:
        image_base64 (str): Base64 encoded input image
        drawing_type (str): Drawing type
        params (Dict[str, Any]): Drawing parameters
    Returns:
        Dictionary with result image (base64)
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        color = params.get("color", "red")

        if drawing_type == "rectangle":
            draw.rectangle(
                [params["left"], params["top"], params["right"], params["bottom"]],
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "circle":
            x, y, r = params["x"], params["y"], params["radius"]
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "line":
            draw.line(
                (
                    params["start_x"],
                    params["start_y"],
                    params["end_x"],
                    params["end_y"],
                ),
                fill=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "text":
            font_size = params.get("font_size", 20)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(
                (params["x"], params["y"]),
                params.get("text", "Text"),
                fill=color,
                font=font,
            )
        else:
            return {"error": f"Unknown drawing type: {drawing_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"result_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def generate_simple_image(
    image_type: str,
    width: int = 500,
    height: int = 500,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a simple image (gradient, noise, pattern, chart).
    Args:
        image_type (str): Type of image
        width (int), height (int)
        params (Dict[str, Any], optional): Specific parameters
    Returns:
        Dictionary with generated image (base64)
    """
    try:
        params = params or {}

        if image_type == "gradient":
            direction = params.get("direction", "horizontal")
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))

            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            if direction == "horizontal":
                for x in range(width):
                    r = int(
                        start_color[0] +
                        (end_color[0] - start_color[0]) * x / width
                    )
                    g = int(
                        start_color[1] +
                        (end_color[1] - start_color[1]) * x / width
                    )
                    b = int(
                        start_color[2] +
                        (end_color[2] - start_color[2]) * x / width
                    )
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            else:
                for y in range(height):
                    r = int(
                        start_color[0] + (end_color[0] -
                                          start_color[0]) * y / height
                    )
                    g = int(
                        start_color[1] + (end_color[1] -
                                          start_color[1]) * y / height
                    )
                    b = int(
                        start_color[2] + (end_color[2] -
                                          start_color[2]) * y / height
                    )
                    draw.line([(0, y), (width, y)], fill=(r, g, b))

        elif image_type == "noise":
            noise_array = np.random.randint(
                0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")

        else:
            return {"error": f"Unsupported image_type {image_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"generated_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def combine_images(
    images_base64: List[str], operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Combine multiple images (collage, stack, blend).
    Args:
        images_base64 (List[str]): List of base64 images
        operation (str): Combination type
        params (Dict[str, Any], optional)
    Returns:
        Dictionary with combined image (base64)
    """
    try:
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}

        if operation == "stack":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                new_img = Image.new("RGB", (total_width, max_height))
                x = 0
                for img in images:
                    new_img.paste(img, (x, 0))
                    x += img.width
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                new_img = Image.new("RGB", (max_width, total_height))
                y = 0
                for img in images:
                    new_img.paste(img, (0, y))
                    y += img.height
        else:
            return {"error": f"Unsupported combination operation {operation}"}

        result_path = save_image(new_img)
        result_base64 = encode_image(result_path)
        return {"combined_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------
# 100 % ChatGoogleGenerativeAI-based tools (no GoogleGenerativeAIImage import)
# -------------------------------------------------------------------

# ───────────────────────── helpers ────────────────────────────────


def _download(url: str) -> Optional[Path]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=Path(url).suffix or ".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return Path(tmp)
    except Exception:
        return None


def _local(path_or_url: str) -> Optional[Path]:
    p = Path(path_or_url).expanduser()
    if p.exists():
        return p
    if path_or_url.startswith(("http://", "https://")):
        return _download(path_or_url)
    return None


def _b64(path: Path) -> tuple[str, str]:
    mime = (
        "image/png" if path.suffix.lower() == ".png" else
        "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else
        "image/webp"
    )
    return mime, base64.b64encode(path.read_bytes()).decode()


def _usage(msg: AIMessage, tag: str) -> None:
    meta = msg.additional_kwargs.get("usage_metadata", {})
    if meta:
        print(f"[TOKENS {tag}] input={meta.get('input_tokens')} "
              f"output={meta.get('output_tokens')} total={meta.get('total_tokens')}")


# ───────────────────────── vision tool ────────────────────────────
_VISION_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

_VISION_PROMPT = """
You are a GAIA-benchmark vision assistant. Return exactly three sections:

1. Description: ≤40-word paragraph describing the scene.
2. Extracted text: verbatim text from the image or “[none]”.

No extra commentary.
""".strip()


@tool("describe_image", return_direct=True)
def describe_image(local_path: str) -> str:
    """
    **Local-file–only** vision tool.

    Steps for the agent:
      1. If the user gives a URL, first call `download_file_from_url`
         (that tool returns the temp path).
      2. Call this tool with that **local** path.

    Returns two sections:
      1. Description – ≤40-word caption.
      2. Extracted text – OCR result or “[none]”.

    Args
    ----
    local_path : str
        Absolute path to a PNG/JPG/WebP on disk (e.g. /tmp/xyz.png).

    Notes
    -----
    • The base-64 payload sent to Gemini never goes into the agent’s message
      history; only Gemini’s textual answer is returned.
    • If `local_path` does not exist, the tool replies with an error string.
    """
    p = Path(local_path).expanduser()
    if not p.exists():
        return f"[describe_image] file not found: {local_path}"

    # encode image for Gemini call
    # helper you already have
    mime, data = _b64(p)
    image_part = {"inline_data": {"mime_type": mime, "data": data}}

    try:
        resp: AIMessage = _VISION_LLM.invoke(
            [
                image_part,
                "Describe this image (≤40 words), then OCR any text or "
                "write “[none]”. Output exactly two labelled sections."
            ]
        )
        return resp.content.strip()                    # plain text only
    except Exception as err:
        return f"[describe_image error] {err}"


# ──────────────────────── audio tool ──────────────────────────────
_AUDIO_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

_AUDIO_PROMPT = (
    "Transcribe the spoken content in the provided audio **verbatim**. "
    "Return ONLY the transcription text."
)


def _audio_b64(path: Path) -> tuple[str, str]:
    mime = (
        "audio/wav" if path.suffix.lower() == ".wav" else
        "audio/mp3" if path.suffix.lower() in {".mp3", ".mpeg"} else
        "audio/flac"
    )
    return mime, base64.b64encode(path.read_bytes()).decode()


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
    calculator,
    wiki_search,
    web_search,
    arxiv_search,
    list_webpage_links,
    browse_webpage_link,
    search_links_for_match,
    run_python_safe,
    save_and_read_file,
    download_file_from_url,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    # analyze_image,
    # transform_image,
    # draw_on_image,
    # generate_simple_image,
    # combine_images,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    # describe_image,
    transcribe_audio
]


def get_tools() -> list:
    """
    Return the list of tools available for the agent.
    This can be used to dynamically load tools in the agent.
    """
    return tools


if __name__ == "__main__":
    # Example usage
    # describe image

    describe_image.invoke(
        {"path_or_url": "https://www.wikiwand.com/en/articles/Cat#/media/File:Cat_August_2010-4.jpg"})
