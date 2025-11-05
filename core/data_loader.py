"""
Data Loader Module

Semantic Scholar data loading and document processing.
"""

import logging
import os
import requests
from typing import List, Optional
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

from .pdf_handler import PDFHandler


class SemanticScholarReader(BaseReader):
    """
    A class to read and process data from Semantic Scholar API.

    This class handles searching for papers, downloading PDFs, and extracting
    full text content from papers available on Semantic Scholar.
    """

    def __init__(self, timeout: int = 10, api_key: Optional[str] = None, base_dir: str = "pdfs"):
        """
        Initialize the SemanticScholar Reader.

        Args:
            timeout: Request timeout in seconds
            api_key: Optional Semantic Scholar API key
            base_dir: Base directory for storing PDFs
        """
        from semanticscholar import SemanticScholar

        self.s2 = SemanticScholar(timeout=timeout, api_key=api_key)
        self.pdf_handler = PDFHandler(base_dir=base_dir)
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)

    def _get_full_text_docs(self, documents: List[Document]) -> List[Document]:
        """
        Get full text documents by downloading and parsing PDFs.

        Args:
            documents: List of Document objects with metadata

        Returns:
            List of Document objects with full text content
        """
        full_text_docs = []

        for paper in documents:
            metadata = paper.extra_info
            url = metadata.get("openAccessPdf")
            external_ids = metadata.get("externalIds")
            paper_id = metadata["paperId"]
            file_path = None
            persist_dir = os.path.join(self.base_dir, f"{paper_id}.pdf")

            # Download from open access URL if available
            if url and not os.path.exists(persist_dir):
                file_path = self.pdf_handler.download_pdf(paper_id, url)

            # Try arXiv if no open access URL
            if (
                not url
                and external_ids
                and "ArXiv" in external_ids
                and not os.path.exists(persist_dir)
            ):
                file_path = self.pdf_handler.download_pdf_from_arxiv(
                    paper_id, external_ids["ArXiv"]
                )

            # Extract text from PDF
            if file_path or os.path.exists(persist_dir):
                text = self.pdf_handler.extract_text_from_pdf(
                    file_path if file_path else persist_dir
                )
                if text:
                    full_text_docs.append(Document(text=text, extra_info=metadata))

        return full_text_docs

    def load_data(
        self,
        query: str,
        limit: int,
        full_text: bool = False,
        returned_fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data from Semantic Scholar based on the query.

        Args:
            query: The search query for papers
            limit: Maximum number of results to return
            full_text: Whether to download and extract full text from PDFs
            returned_fields: List of fields to return from Semantic Scholar API

        Returns:
            List of Document objects containing paper data

        Raises:
            Exception: If there's an error performing the search
        """
        if returned_fields is None:
            returned_fields = [
                "title",
                "abstract",
                "venue",
                "year",
                "paperId",
                "citationCount",
                "openAccessPdf",
                "authors",
                "externalIds",
            ]

        results = []
        queries = [query]  # Can be extended to support multiple related queries

        try:
            for question in queries:
                self.logger.info(f"Searching for {question}")
                _results = self.s2.search_paper(question, limit=limit, fields=returned_fields)
                results.extend(_results[:limit])
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            self.logger.error(
                "Failed to fetch data from Semantic Scholar with exception: %s", e
            )
            raise
        except Exception as e:
            self.logger.error("An unexpected error occurred: %s", e)
            raise

        # Convert results to Document objects
        documents = []
        for item in results[: limit * len(queries)]:
            open_access_pdf = getattr(item, "openAccessPdf", None)
            abstract = getattr(item, "abstract", None)
            title = getattr(item, "title", None)

            # Concatenate title and abstract for text content
            text = None
            if abstract and title:
                text = title + " " + abstract
            elif title:
                text = title

            metadata = {
                "title": title,
                "venue": getattr(item, "venue", None),
                "year": getattr(item, "year", None),
                "paperId": getattr(item, "paperId", None),
                "citationCount": getattr(item, "citationCount", None),
                "openAccessPdf": open_access_pdf.get("url") if open_access_pdf else None,
                "authors": [author["name"] for author in getattr(item, "authors", [])],
                "externalIds": getattr(item, "externalIds", None),
            }
            documents.append(Document(text=text, extra_info=metadata))

        # Download and extract full text if requested
        if full_text:
            self.logger.info("Getting full text documents...")
            full_text_documents = self._get_full_text_docs(documents)
            documents.extend(full_text_documents)

        # Remove duplicates based on paperId
        documents = self._get_unique_docs(documents)

        return documents

    @staticmethod
    def _get_unique_docs(docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on paperId.

        Args:
            docs: List of Document objects

        Returns:
            List of unique Document objects
        """
        unique_docs_id = []
        unique_docs = []
        for doc in docs:
            paper_id = doc.extra_info.get('paperId')
            if paper_id not in unique_docs_id:
                unique_docs_id.append(paper_id)
                unique_docs.append(doc)
        return unique_docs

    def _clear_cache(self):
        """Delete the citation cache folders."""
        import shutil
        import glob

        for folder in glob.glob("./.citation*"):
            shutil.rmtree(folder)
