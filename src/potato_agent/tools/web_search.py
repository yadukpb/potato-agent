"""Web search via DuckDuckGo (no API key required)."""
from ddgs import DDGS

NAME = "web_search"
DESCRIPTION = 'Search the web. Args: {"query": "latest news on x", "max_results": 3}'


def run(args: dict) -> str:
    query = args.get("query", "")
    max_results = int(args.get("max_results", 3))
    if not query:
        return "Error: 'query' is required."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Error searching web: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)
