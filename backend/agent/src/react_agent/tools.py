"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, cast

from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from .context import Context

_FAISS_VS = None
_FAISS_VS_KEY = None


def _default_faiss_index_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "faiss_store"


def _get_embeddings_class() -> Type[Any]:
    try:
        module = importlib.import_module("langchain_huggingface")
        return cast(Type[Any], getattr(module, "HuggingFaceEmbeddings"))
    except Exception:
        module = importlib.import_module("langchain_community.embeddings")
        return cast(Type[Any], getattr(module, "HuggingFaceEmbeddings"))


def _ascii_safe_index_dir(index_dir: Path) -> Path:
    index_str = str(index_dir)
    if index_str.isascii():
        return index_dir
    fallback = Path(os.environ.get("FAISS_INDEX_DIR_ASCII", r"D:\faiss_store"))
    fallback.mkdir(parents=True, exist_ok=True)
    for name in ("index.faiss", "index.pkl"):
        src = index_dir / name
        dst = fallback / name
        if src.exists():
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copyfile(src, dst)
    return fallback


def _load_faiss_vectorstore(index_dir: Path, model_name: str) -> FAISS:
    global _FAISS_VS, _FAISS_VS_KEY

    embeddings = _get_embeddings_class()(model_name=model_name)

    def _has_index_files(path: Path) -> bool:
        return (path / "index.faiss").exists() and (path / "index.pkl").exists()

    fallback_env = os.environ.get("FAISS_INDEX_DIR_FALLBACK", r"D:\ai大模型应用开发\师智分身\faiss_store")
    candidates = [
        index_dir,
        _default_faiss_index_dir(),
        Path(fallback_env),
    ]

    resolved_dir = None
    for candidate in candidates:
        if _has_index_files(candidate):
            resolved_dir = candidate
            break
    if resolved_dir is None:
        raise FileNotFoundError(f"Index dir not found: {index_dir}")

    resolved_dir = _ascii_safe_index_dir(resolved_dir)

    key = (str(resolved_dir), model_name)
    if _FAISS_VS is not None and _FAISS_VS_KEY == key:
        return _FAISS_VS

    original_cwd = os.getcwd()
    try:
        os.chdir(resolved_dir.parent)
        _FAISS_VS = FAISS.load_local(
            resolved_dir.name,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    finally:
        os.chdir(original_cwd)

    _FAISS_VS_KEY = key
    return _FAISS_VS


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    try:
        runtime = get_runtime(Context)
        max_results = runtime.context.max_search_results if runtime.context else 10
    except Exception:
        max_results = 10

    wrapped = TavilySearch(max_results=max_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def faiss_search_local(query: str, k: int = 5) -> dict[str, Any]:
    """Search for documents in the local FAISS index with score filtering.

    Args:
        query: The search query string.
        k: The number of documents to retrieve. Defaults to 5.

    Returns:
        A dictionary containing the search results and metadata.
    """
    index_dir = Path(os.environ.get("FAISS_INDEX_DIR", str(_default_faiss_index_dir())))
    if index_dir.name == "data" and not (index_dir / "index.faiss").exists():
        candidate = index_dir.parent / "faiss_store"
        if (candidate / "index.faiss").exists() and (candidate / "index.pkl").exists():
            index_dir = candidate
    if not (index_dir / "index.faiss").exists():
        fallback = _default_faiss_index_dir()
        if (fallback / "index.faiss").exists() and (fallback / "index.pkl").exists():
            index_dir = fallback
    embedding_model = os.environ.get(
        "FAISS_EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5",
    )

    vectorstore = _load_faiss_vectorstore(index_dir, embedding_model)
    
    # Use similarity_search_with_score to get scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    threshold_raw = os.environ.get("FAISS_SCORE_THRESHOLD", "5.0").strip().lower()
    threshold = None if threshold_raw in {"", "none", "null", "off"} else float(threshold_raw)

    results = []
    for doc, score in docs_and_scores:
        if threshold is None or score <= threshold:
            results.append({
                "content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
                "score": float(score)
            })

    return {
        "index_dir": str(index_dir),
        "k": k,
        "results": results,
    }


TOOLS: List[Callable[..., Any]] = [search, faiss_search_local]
