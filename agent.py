"""
Research agent. Hand-rolled tool loop using the anthropic SDK.

Features:
- Planning step: generate an editable plan before execution.
- Adversarial mode: stress-test the question and surface counterevidence.
- Private corpus: search user-uploaded documents alongside the web.
- Citation formatter: post-format sources in APA, MLA, Chicago, etc.

Web search and page extraction both go through Tavily.
"""

import os
from typing import Callable, Optional

from anthropic import Anthropic, RateLimitError
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 4096
MAX_STEPS = 10
SEARCH_SNIPPET_CHARS = 250
PAGE_EXTRACT_CHARS = 8000

BASE_SYSTEM = """You are a careful research agent.

For each question:
1. Use web_search to find authoritative, recent sources. Prefer multiple
   focused searches over one broad one when the question has sub-parts.
2. Use fetch_page on the most promising URLs to read the full content.
   Don't trust search snippets alone for anything that matters.
3. Cross-check important claims across at least two sources when possible.
4. Write a clear briefing - 3 to 6 short paragraphs OR a bulleted summary,
   whichever fits. Use inline citations like [1], [2] pointing at the URLs
   you actually relied on.
5. End with a "Sources" section listing each cited URL, numbered to match.

If sources disagree, say so. If you can't find good evidence, say that
explicitly - don't guess."""

ADVERSARIAL_PROMPT = """

ADVERSARIAL MODE: Stress-test the question. Prioritize finding
counterevidence, weak assumptions, and the strongest case AGAINST the
prevailing view. Surface credible dissenting sources. Be sharp, not
balanced - the user wants their thesis attacked, not validated."""

CORPUS_PROMPT = """

PRIVATE CORPUS: The user has uploaded private documents. Use the
corpus_search tool to find relevant passages from their docs. Check the
corpus for question-specific context before or alongside web searches."""

PLAN_PROMPT = """

APPROVED RESEARCH PLAN:
{plan}

Follow this plan, but adapt if you find a better path mid-research."""

PLANNING_USER = """Write a brief research plan for this question:

{question}

The plan should include:
- 3-5 specific sub-questions to answer
- Key search queries to run
- What sources or evidence would be most valuable
- {focus}
{corpus_line}

Keep it concise (5-12 lines, prose or bullets). The user will review and
edit it before execution. Output only the plan - no preamble."""

CITATION_USER = """Format the following URLs as a numbered bibliography in {style} style.

For each URL, infer reasonable bibliographic fields (author, site name,
title, year) from the URL structure. When a field is unknown, use the
conventional fallback (e.g. "n.d." for missing dates, the site name
for a missing author).

URLs:
{urls}

Output ONLY the formatted citation list, numbered to match the input
order. No preamble, no commentary."""


def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _build_system(plan: Optional[str], adversarial: bool, has_corpus: bool) -> str:
    s = BASE_SYSTEM
    if plan:
        s += PLAN_PROMPT.format(plan=plan)
    if has_corpus:
        s += CORPUS_PROMPT
    if adversarial:
        s += ADVERSARIAL_PROMPT
    return s


def _tools(has_corpus: bool) -> list[dict]:
    tools = [
        {
            "name": "web_search",
            "description": "Search the web. Returns a numbered list of results with title, URL, and snippet.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "max_results": {"type": "integer", "description": "1-10.", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_page",
            "description": "Fetch a URL and return its readable text. Use after web_search to read a page.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL including https://"}},
                "required": ["url"],
            },
        },
    ]
    if has_corpus:
        tools.append({
            "name": "corpus_search",
            "description": "Search the user's uploaded private documents for passages relevant to a query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "description": "1-10.", "default": 5},
                },
                "required": ["query"],
            },
        })
    return tools


def _web_search(query: str, max_results: int = 5) -> tuple[str, list[str]]:
    max_results = max(1, min(int(max_results), 10))
    resp = _tavily().search(query=query, max_results=max_results)
    results = resp.get("results", [])
    urls = [r["url"] for r in results if r.get("url")]
    if not results:
        return "No results.", []
    lines = [
        f"[{i}] {r.get('title', '(no title)')}\n    URL: {r.get('url', '')}\n    {r.get('content', '')[:SEARCH_SNIPPET_CHARS]}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n".join(lines), urls


def _fetch_page(url: str) -> str:
    try:
        resp = _tavily().extract(urls=[url])
    except Exception as e:
        return f"Error extracting {url}: {e}"
    results = resp.get("results", [])
    if not results:
        return f"No content extracted from {url}."
    raw = results[0].get("raw_content", "") or ""
    if len(raw) > PAGE_EXTRACT_CHARS:
        raw = raw[:PAGE_EXTRACT_CHARS] + "\n\n[...truncated...]"
    return raw or f"Empty content from {url}."


def _corpus_search(query: str, corpus: dict, top_k: int = 5) -> str:
    """Naive keyword scoring over 1000-char chunks. Good enough for v1."""
    top_k = max(1, min(int(top_k), 10))
    chunks = []
    for filename, text in corpus.items():
        for i in range(0, len(text), 800):
            chunks.append((filename, i, text[i:i + 1000]))

    query_terms = {w for w in query.lower().split() if len(w) > 2}
    if not query_terms:
        return "Query too short to match."

    scored = []
    for filename, offset, chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(t) for t in query_terms)
        if score > 0:
            scored.append((score, filename, offset, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]
    if not top:
        return f"No matches for '{query}' in uploaded documents."

    return "\n\n".join(
        f"[{filename}, offset {offset}, score {score}]\n{chunk.strip()}"
        for score, filename, offset, chunk in top
    )


def plan_research(question: str, adversarial: bool = False, has_corpus: bool = False) -> str:
    """Generate an editable research plan for a question."""
    client = Anthropic()
    focus = (
        "Counterevidence and skeptical angles to investigate"
        if adversarial
        else "Specific facts and claims to verify"
    )
    corpus_line = (
        "- The user has uploaded private documents; mention checking them as part of the plan"
        if has_corpus
        else ""
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": PLANNING_USER.format(question=question, focus=focus, corpus_line=corpus_line),
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run_research(
    question: str,
    plan: Optional[str] = None,
    adversarial: bool = False,
    corpus: Optional[dict] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run the agent loop. Returns {briefing, sources, searches, corpus_hits}."""
    client = Anthropic()
    has_corpus = bool(corpus)
    system = _build_system(plan, adversarial, has_corpus)
    tools = _tools(has_corpus)

    user_msg = question
    if plan:
        user_msg = f"Question: {question}\n\n(The approved research plan is in the system prompt.)"

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    sources: list[str] = []
    searches: list[str] = []
    corpus_hits: list[str] = []
    final_text_parts: list[str] = []

    def emit(ev: dict) -> None:
        if on_event is not None:
            on_event(ev)

    for _ in range(MAX_STEPS):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if getattr(delta, "type", None) == "text_delta":
                            emit({"type": "text_delta", "text": delta.text})
                final = stream.get_final_message()
        except RateLimitError as e:
            note = (
                "\n\n[Hit the Anthropic API rate limit before finishing. "
                "Sources gathered so far are listed below. Wait a minute and re-run, "
                "or upgrade your tier at https://console.anthropic.com/settings/billing.]"
            )
            return {
                "briefing": "".join(final_text_parts) + note,
                "sources": sources,
                "searches": searches,
                "corpus_hits": corpus_hits,
            }

        messages.append({"role": "assistant", "content": final.content})
        for block in final.content:
            if block.type == "text":
                final_text_parts.append(block.text)

        if final.stop_reason == "end_turn":
            emit({"type": "done"})
            return {
                "briefing": "".join(final_text_parts),
                "sources": sources,
                "searches": searches,
                "corpus_hits": corpus_hits,
            }

        if final.stop_reason != "tool_use":
            return {
                "briefing": "".join(final_text_parts) or f"Agent stopped unexpectedly (stop_reason={final.stop_reason}).",
                "sources": sources,
                "searches": searches,
                "corpus_hits": corpus_hits,
            }

        tool_results = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            args = dict(block.input)

            if block.name == "web_search":
                query = args.get("query", "")
                emit({"type": "search_query", "query": query})
                text, urls = _web_search(query, args.get("max_results", 5))
                searches.append(query)
                for u in urls:
                    if u not in sources:
                        sources.append(u)
                emit({"type": "search_results", "urls": urls})
                content = text
            elif block.name == "fetch_page":
                url = args.get("url", "")
                emit({"type": "fetch", "url": url})
                if url and url not in sources:
                    sources.append(url)
                content = _fetch_page(url)
            elif block.name == "corpus_search" and has_corpus:
                query = args.get("query", "")
                emit({"type": "corpus_search", "query": query})
                corpus_hits.append(query)
                content = _corpus_search(query, corpus, args.get("top_k", 5))
            else:
                content = f"Unknown tool: {block.name}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "briefing": "Agent hit the step limit before finishing.",
        "sources": sources,
        "searches": searches,
        "corpus_hits": corpus_hits,
    }


def format_citations(sources: list[str], style: str) -> str:
    """Format a list of source URLs as a bibliography in the given style."""
    if not sources:
        return ""
    client = Anthropic()
    urls = "\n".join(f"{i}. {u}" for i, u in enumerate(sources, 1))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": CITATION_USER.format(style=style, urls=urls),
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    print(f"Q: {q}\n")

    plan = plan_research(q)
    print("--- PLAN ---")
    print(plan)
    print()

    def cli_event(ev: dict) -> None:
        t = ev["type"]
        if t == "text_delta":
            print(ev["text"], end="", flush=True)
        elif t == "search_query":
            print(f"\n[search] {ev['query']}", flush=True)
        elif t == "search_results":
            print(f"[{len(ev['urls'])} results]", flush=True)
        elif t == "fetch":
            print(f"[fetch] {ev['url']}", flush=True)

    out = run_research(q, plan=plan, on_event=cli_event)
    print("\n\n--- SOURCES ---")
    for u in out["sources"]:
        print(f"- {u}")
    print("\n--- CITATIONS (APA) ---")
    print(format_citations(out["sources"], "APA (7th)"))
