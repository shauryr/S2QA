from langchain.schema import Document
from typing import Any, Dict, List, Optional
import requests
import logging
import ast

# TODO full text capability


def generate_search_queries_prompt(question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return (
        f'Write 4 short google search queries to search online that form an objective opinion from the following: "{question}"'
        f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", "query 4"]'
    )


def load_data(
    query,
    limit,
    full_text=False,
    returned_fields=[
        "title",
        "abstract",
        "venue",
        "year",
        "paperId",
        "citationCount",
        "openAccessPdf",
        "authors",
        "externalIds",
    ],
) -> List[Document]:
    """
    Loads data from Semantic Scholar based on the entered query and returned_fields

    Parameters
    ----------
    query: str
        The search query for the paper
    limit: int, optional
        The number of maximum results returned (default is 10)
    returned_fields: list, optional
        The list of fields to be returned from the search

    Returns
    -------
    list
        The list of Document object that contains the search results

    Raises
    ------
    Exception
        If there is an error while performing the search

    """
    documents = []
    results=[]
    try:
        from semanticscholar import SemanticScholar

        s2 = SemanticScholar()
        results = s2.search_paper(query, limit=limit, fields=returned_fields)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
        logging.error(
            "Failed to fetch data from Semantic Scholar with exception: %s", e
        )
        raise
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
    

    for item in results[:limit]:
        openAccessPdf = getattr(item, "openAccessPdf", None)
        abstract = getattr(item, "abstract", None)
        title = getattr(item, "title", None)
        text = None
        # concat title and abstract
        if abstract and title:
            text = title + ";\n ABSTRACT:" + abstract
        elif not abstract:
            text = title

        metadata = {
            "title": title,
            "venue": getattr(item, "venue", None),
            "year": getattr(item, "year", None),
            "paperId": getattr(item, "paperId", None),
            "citationCount": getattr(item, "citationCount", None),
            "openAccessPdf": openAccessPdf.get("url") if openAccessPdf else None,
            "authors": [author["name"] for author in getattr(item, "authors", [])],
            "externalIds": getattr(item, "externalIds", None),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    if full_text:
        logging.info("Getting full text documents...")
        full_text_documents = self._get_full_text_docs(documents)
        documents.extend(full_text_documents)
    return documents


def get_questions(response_text):
    data = response_text.split("\n")
    data = [ast.literal_eval(item)[0] for item in data]
    return data
