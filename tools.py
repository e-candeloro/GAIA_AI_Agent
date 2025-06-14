from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_core.tools import tool
from langchain_core.documents import Document

from duckduckgo_search import DDGS
import trafilatura
from langchain_core.tools import tool
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from typing import Sequence
import math

SAFE_GLOBALS = {"__builtins__": {}, "math": math}


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

tools = [
    calculator,
    wiki_search,
    web_search,
    arxiv_search,
    list_webpage_links,
    browse_webpage_link,
    search_links_for_match,
]

def get_tools() -> list:
    """
    Return the list of tools available for the agent.
    This can be used to dynamically load tools in the agent.
    """
    return tools
