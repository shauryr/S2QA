"""
UI Helpers Module

Utility functions for UI display and formatting.
"""

import pandas as pd
from typing import List, Any


# Configuration
TWITTER_USERNAME = "shauryr"
GITHUB_REPO = "shauryr/s2qa"


def get_twitter_badge() -> str:
    """
    Construct the Markdown code for the Twitter badge.

    Returns:
        HTML string for Twitter badge
    """
    return f'<a href="https://twitter.com/{TWITTER_USERNAME}" target="_blank"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" /></a>'


def get_link_tree_badge() -> str:
    """
    Construct the Markdown code for the LinkTree badge.

    Returns:
        HTML string for LinkTree badge
    """
    return f'<a href="https://linktr.ee/shauryr" target="_blank"><img src="https://img.shields.io/badge/Linktree-39E09B?style=for-the-badge&logo=linktree&logoColor=white" /></a>'


def get_github_badge() -> str:
    """
    Construct the Markdown code for the GitHub badge.

    Returns:
        HTML string for GitHub badge
    """
    return f'<a href="https://github.com/{GITHUB_REPO}" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" /></a>'


def display_questions(sample_questions: List[str]) -> str:
    """
    Format sample questions for display.

    Args:
        sample_questions: List of question strings

    Returns:
        Markdown-formatted string of questions
    """
    s = "#### 🧐 More questions? \n"
    for question in sample_questions:
        s += "- " + question + "\n"
    return s


def documents_to_df(documents: List[Any]) -> pd.DataFrame:
    """
    Convert Document objects to a pandas DataFrame.

    Args:
        documents: List of Document objects with extra_info metadata

    Returns:
        DataFrame containing document metadata
    """
    list_data = []
    for i, doc in enumerate(documents):
        if hasattr(doc, 'extra_info'):
            list_data.append(doc.extra_info.copy())

    df = pd.DataFrame(list_data)
    return df


def print_papers_streamlit(df: pd.DataFrame, k: int = 8):
    """
    Print papers in Streamlit format.

    Args:
        df: DataFrame containing paper information
        k: Number of papers to display
    """
    import streamlit as st

    count = 1
    for i in range(min(k, len(df))):
        title = df.iloc[i]["title"]
        link = f"https://www.semanticscholar.org/paper/{df.iloc[i]['paperId']}"
        venue = df.iloc[i].get("venue", "Unknown")
        year = df.iloc[i].get("year", "Unknown")
        st.markdown(f"{[count]} [{title}]({link}) - {venue}, {year}")
        count += 1
