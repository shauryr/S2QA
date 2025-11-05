"""
Prompts Module

Handles prompt generation for LLM queries.
"""

import pandas as pd
from typing import Optional


def generate_search_queries_prompt(question: str) -> str:
    """
    Generate a prompt for creating related search queries.

    Args:
        question: The main question

    Returns:
        Prompt string for generating related queries
    """
    return (
        f'Please generate four related search queries that align with the initial query: "{question}"'
        f'Each variation should be presented as a list of strings, following this format: ["query 1", "query 2", "query 3", "query 4"]'
    )


def create_context_chatgpt(question: str, df: pd.DataFrame, k: int = 5) -> str:
    """
    Create a context string for ChatGPT from top-k papers.

    Args:
        question: Question being asked
        df: DataFrame containing papers with 'tldr' and 'paperId' columns
        k: Number of papers to include

    Returns:
        Formatted context string with citations
    """
    returns = []
    count = 1

    # Take top k papers
    for i, row in df[:k].iterrows():
        returns.append(
            "["
            + str(count)
            + "] "
            + row["tldr"]
            + "\nURL: "
            + "https://www.semanticscholar.org/paper/"
            + row["paperId"]
        )
        count += 1

    # Return the context
    return "\n\n".join(returns)


def generate_prompt(
    df: pd.DataFrame,
    query: str,
    k: int = 5,
    instructions: Optional[str] = None
) -> str:
    """
    Generate a complete prompt for answering a question with papers.

    Args:
        df: DataFrame containing papers
        query: User's question
        k: Number of papers to include in context
        instructions: Optional custom instructions

    Returns:
        Complete prompt string
    """
    if instructions is None:
        instructions = (
            "Instructions: Using the provided web search results, write a comprehensive reply to the given query. "
            "If you find a result relevant, make sure to cite the result using [[number](URL)] notation after the reference. "
            "End your answer with a summary.\nQuery:"
        )

    context = create_context_chatgpt(query, df, k=k)
    prompt = f"{context} \n\n{instructions} {query}\nAnswer:"

    return prompt


def answer_question_chatgpt(
    df: pd.DataFrame,
    question: str = "What is the impact of creatine on cognition?",
    k: int = 5,
    instructions: Optional[str] = None,
    max_len: int = 3000,
    debug: bool = False,
) -> str:
    """
    Generate a prompt for answering a question with ChatGPT.

    This is a wrapper around generate_prompt for backward compatibility.

    Args:
        df: DataFrame containing papers
        question: Question to answer
        k: Number of papers to include
        instructions: Optional custom instructions
        max_len: Maximum context length (not currently used)
        debug: Enable debug mode (not currently used)

    Returns:
        Complete prompt string
    """
    if instructions is None:
        instructions = (
            "Instructions: Using the provided web search results, write a comprehensive reply to the given query. "
            "If you find a result relevant definitely make sure to cite the result using [[number](URL)] notation after the reference. "
            "End your answer with a summary.\nQuery:"
        )

    return generate_prompt(df, question, k=k, instructions=instructions)
