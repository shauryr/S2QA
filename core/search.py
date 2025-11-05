"""
Search Module

Handles paper search, ranking, and query preprocessing.
"""

import requests
import nltk
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import get_specter_embeddings, get_tokenizer


def search_papers(
    query: str,
    limit: int = 20,
    fields: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> dict:
    """
    Search for papers on Semantic Scholar.

    Args:
        query: Search query
        limit: Maximum number of results
        fields: Fields to return from the API
        api_key: Optional Semantic Scholar API key

    Returns:
        Dictionary containing search results
    """
    if fields is None:
        fields = ["title", "abstract", "venue", "year", "paperId", "citationCount", "openAccessPdf"]

    # Format query for URL
    query = query.replace(" ", "+")
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields={",".join(fields)}'

    headers = {"Accept": "*/*"}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(url, headers=headers, timeout=30)
    return response.json()


def preprocess_query(query: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess a search query.

    Args:
        query: Raw query string
        remove_stopwords: Whether to remove stopwords

    Returns:
        Preprocessed query string
    """
    query = query.lower()

    if remove_stopwords:
        try:
            stopwords = set(nltk.corpus.stopwords.words("english"))
            # Add custom stopwords
            stopwords.update(["please", "review"])
            query = " ".join([word for word in query.split() if word not in stopwords])
        except LookupError:
            # If stopwords not downloaded, skip stopword removal
            pass

    return query


def get_results(query: str, limit: int = 20, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Get search results as a DataFrame.

    Args:
        query: Search query
        limit: Maximum number of results
        api_key: Optional Semantic Scholar API key

    Returns:
        DataFrame containing search results
    """
    search_results = search_papers(preprocess_query(query), limit, api_key=api_key)

    if search_results.get("total", 0) == 0:
        print("No results found - Try another query")
        return pd.DataFrame()

    # Create DataFrame and drop rows with missing titles
    df = pd.DataFrame(search_results["data"])
    df = df.dropna(subset=["title"])

    return df


def rerank_papers(
    df: pd.DataFrame,
    query: str,
    column_name: str = "title_abs"
) -> Tuple[pd.DataFrame, str]:
    """
    Re-rank papers using SPECTER2 embeddings for semantic similarity.

    Args:
        df: DataFrame containing papers
        query: Query string for ranking
        column_name: Name of column to create/use for ranking

    Returns:
        Tuple of (re-ranked DataFrame, processed query)
    """
    tokenizer = get_tokenizer()

    # Merge title and abstract into a single column
    df[column_name] = [
        d["title"] + tokenizer.sep_token + (d.get("abstract") or "")
        for d in df.to_dict("records")
    ]

    # Count tokens for each paper
    df["n_tokens"] = df[column_name].apply(lambda x: len(tokenizer.encode(x)))

    # Get embeddings for papers and query
    doc_embeddings = get_specter_embeddings(list(df[column_name]))
    query_embeddings = get_specter_embeddings(query)

    # Store embeddings and calculate similarity
    df["specter_embeddings"] = list(doc_embeddings)
    df["similarity"] = cosine_similarity(query_embeddings, doc_embeddings).flatten()

    # Sort by similarity
    df.sort_values(by="similarity", ascending=False, inplace=True)

    return df, query


def create_context(
    question: str,
    df: pd.DataFrame,
    max_len: int = 3800,
    column_name: str = "title_abs"
) -> str:
    """
    Create a context string for a question from a DataFrame of papers.

    Args:
        question: Question to create context for
        df: DataFrame containing papers (should be pre-ranked)
        max_len: Maximum token length for context
        column_name: Column name to use for text

    Returns:
        Context string
    """
    returns = []
    cur_len = 0

    # Add papers until we reach max length
    for i, row in df.iterrows():
        # Add the length of the text to the current length
        cur_len += row.get("n_tokens", 0) + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Add text to context
        returns.append(row[column_name])

    # Return the context
    return "\n\n###\n\n".join(returns)
