"""
S2QA Core Module

Consolidated utilities for the S2QA research question answering system.
"""

from .data_loader import SemanticScholarReader
from .pdf_handler import PDFHandler
from .search import search_papers, rerank_papers, preprocess_query, get_results
from .embeddings import get_specter_embeddings
from .citations import (
    get_citation,
    extract_numbers_in_brackets,
    generate_used_reference_display,
    generate_reference_display,
)
from .ui_helpers import (
    get_twitter_badge,
    get_github_badge,
    get_link_tree_badge,
    display_questions,
    documents_to_df,
)
from .prompts import generate_prompt, create_context_chatgpt

__all__ = [
    "SemanticScholarReader",
    "PDFHandler",
    "search_papers",
    "rerank_papers",
    "preprocess_query",
    "get_results",
    "get_specter_embeddings",
    "get_citation",
    "extract_numbers_in_brackets",
    "generate_used_reference_display",
    "generate_reference_display",
    "get_twitter_badge",
    "get_github_badge",
    "get_link_tree_badge",
    "display_questions",
    "documents_to_df",
    "generate_prompt",
    "create_context_chatgpt",
]
