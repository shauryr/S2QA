"""
PDF Handler Module

Consolidated PDF download, parsing, and text extraction utilities.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional, List
from PyPDF2 import PdfReader


class PDFHandler:
    """Handles PDF download and text extraction operations."""

    def __init__(self, base_dir: str = "pdfs"):
        """
        Initialize PDF Handler.

        Args:
            base_dir: Base directory for storing downloaded PDFs
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)

        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def download_pdf(self, paper_id: str, url: str) -> Optional[str]:
        """
        Download a PDF from a given URL.

        Args:
            paper_id: Unique identifier for the paper
            url: URL to download the PDF from

        Returns:
            Path to the downloaded PDF file, or None if download failed
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            content_type = response.headers.get("Content-Type", "")

            # Check if the response is a PDF
            if "application/pdf" in content_type:
                os.makedirs(self.base_dir, exist_ok=True)
                file_path = os.path.join(self.base_dir, f"{paper_id}.pdf")

                # Check if file already exists
                if os.path.exists(file_path):
                    self.logger.info(f"{file_path} already exists")
                    return file_path

                # Write PDF to file
                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)

                self.logger.info(f"Downloaded PDF from {url}")
                return file_path
            else:
                self.logger.warning(f"{url} was not downloaded: protected or not a PDF")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {e}")
            return None

    def download_pdf_from_arxiv(self, paper_id: str, arxiv_id: str) -> Optional[str]:
        """
        Download a PDF from arXiv.

        Args:
            paper_id: Unique identifier for the paper
            arxiv_id: arXiv ID of the paper

        Returns:
            Path to the downloaded PDF file, or None if download failed
        """
        try:
            import arxiv

            paper = next(arxiv.Search(id_list=[arxiv_id], max_results=1).results())
            paper.download_pdf(dirpath=self.base_dir, filename=f"{paper_id}.pdf")
            return os.path.join(self.base_dir, f"{paper_id}.pdf")
        except Exception as e:
            self.logger.error(f"Error downloading PDF from arXiv {arxiv_id}: {e}")
            return None

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text, or None if extraction failed
        """
        if not Path(file_path).exists():
            self.logger.warning(f"PDF file not found: {file_path}")
            return None

        try:
            with open(file_path, "rb") as fp:
                pdf = PdfReader(fp)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            self.logger.error(f"Failed to read PDF {file_path}: {e}")
            return None

    def get_pdfs_batch(self, papers_df, folder: Optional[str] = None):
        """
        Download multiple PDFs from a DataFrame.

        Args:
            papers_df: DataFrame with 'paperId' and 'openAccessPdf' columns
            folder: Optional folder override for downloads
        """
        download_dir = folder if folder else self.base_dir

        for i in range(len(papers_df)):
            url = papers_df.iloc[i].get('openAccessPdf')
            if url is not None:
                filename = papers_df.iloc[i]['paperId'] + '.pdf'

                # Handle dict or direct URL
                if isinstance(url, dict):
                    url = url.get('url')

                if url and not os.path.exists(os.path.join(download_dir, filename)):
                    try:
                        r = requests.get(url, allow_redirects=True, timeout=30)
                        os.makedirs(download_dir, exist_ok=True)
                        with open(os.path.join(download_dir, filename), 'wb') as f:
                            f.write(r.content)
                    except Exception as e:
                        self.logger.error(f'Error downloading PDF for {filename}: {e}')


def split_text_to_chunks(text: str, model_name: str = "allenai/specter2",
                        tokens_per_chunk: int = 512) -> List[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: Text to split
        model_name: Model name for tokenizer
        tokens_per_chunk: Number of tokens per chunk

    Returns:
        List of text chunks
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_text = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    num_chunks = (len(encoded_text[0]) + tokens_per_chunk - 1) // tokens_per_chunk

    chunks = []
    for i in range(num_chunks):
        start = i * tokens_per_chunk
        end = (i + 1) * tokens_per_chunk

        chunk = encoded_text[0, start:end]
        decoded_chunk = tokenizer.decode(chunk, clean_up_tokenization_spaces=True)
        chunks.append(decoded_chunk)

    return chunks
