# tools.py

from langchain_core.tools import tool
import os
from textwrap import dedent


@tool
def search_tool(query: str) -> str:
    """
    Simulated web search for information about a topic.
    Returns a short "search results" style summary.
    In a real app, you'd call a search API here.
    """
    query_lower = query.lower()

    return f"No specific search results stored for: {query}"


@tool
def wiki_tool(topic: str) -> str:
    """
    Simulated Wikipedia lookup.
    Returns a short encyclopedic explanation if we know the topic.
    """
    topic_lower = topic.lower()

    return f"No Wikipedia-style summary stored for topic: {topic}"


@tool
def save_tool(content: str, filename: str = "research_note.txt") -> str:
    """
    Save research content to a local text file.
    Returns the absolute path or an error message.
    """
    try:
        path = os.path.abspath(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n\n" + "=" * 80 + "\n\n")
        return f"Content saved to {path}"
    except Exception as e:
        return f"Failed to save content: {e}"
