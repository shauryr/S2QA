"""
Embeddings Module

Handles text embedding generation for semantic search and ranking.
"""

from typing import List, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel


# Initialize SPECTER2 model and tokenizer (lazy loading)
_tokenizer = None
_model = None


def _get_specter_model():
    """Lazy load SPECTER2 model and tokenizer."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
    if _model is None:
        _model = AutoModel.from_pretrained("allenai/specter2")
    return _tokenizer, _model


def get_specter_embeddings(text: Union[str, List[str]]) -> np.ndarray:
    """
    Generate SPECTER2 embeddings for the given text.

    SPECTER2 is a transformer-based model specifically designed for
    generating embeddings for scientific papers.

    Args:
        text: Text or list of texts to embed

    Returns:
        Numpy array of embeddings
    """
    tokenizer, model = _get_specter_model()

    # Tokenize the text
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # Get the embeddings
    embeddings = model(**tokens).pooler_output

    # Return the embeddings as numpy array
    return embeddings.detach().numpy()


def get_tokenizer():
    """
    Get the SPECTER2 tokenizer.

    Returns:
        SPECTER2 tokenizer
    """
    tokenizer, _ = _get_specter_model()
    return tokenizer
