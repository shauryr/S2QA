"""
S2QA Core Module

Consolidated utilities for the S2QA research question answering system.

This module uses lazy imports to avoid requiring all dependencies upfront.
Individual modules can be imported directly if needed.
"""

# Lazy imports to avoid requiring all dependencies at once
def __getattr__(name):
    """Lazy import mechanism for module attributes."""

    # Map attribute names to their modules
    _module_map = {
        # data_loader.py
        'SemanticScholarReader': 'data_loader',
        # pdf_handler.py
        'PDFHandler': 'pdf_handler',
        # search.py
        'search_papers': 'search',
        'rerank_papers': 'search',
        'preprocess_query': 'search',
        'get_results': 'search',
        # embeddings.py
        'get_specter_embeddings': 'embeddings',
        # citations.py
        'get_citation': 'citations',
        'extract_numbers_in_brackets': 'citations',
        'generate_used_reference_display': 'citations',
        'generate_reference_display': 'citations',
        # ui_helpers.py
        'get_twitter_badge': 'ui_helpers',
        'get_github_badge': 'ui_helpers',
        'get_link_tree_badge': 'ui_helpers',
        'display_questions': 'ui_helpers',
        'documents_to_df': 'ui_helpers',
        # prompts.py
        'generate_prompt': 'prompts',
        'create_context_chatgpt': 'prompts',
    }

    if name in _module_map:
        module_name = _module_map[name]
        module = __import__(f'core.{module_name}', fromlist=[name])
        return getattr(module, name)

    raise AttributeError(f"module 'core' has no attribute '{name}'")


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
