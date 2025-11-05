"""
Citations Module

Handles citation formatting, extraction, and display.
"""

import re
from typing import List, Dict, Any


def get_citation(metadata: Dict[str, Any]) -> str:
    """
    Generate an APA-style citation from paper metadata.

    Args:
        metadata: Dictionary containing paper metadata

    Returns:
        Formatted citation string
    """
    # Extract details from metadata
    title = metadata.get("title", "No Title")
    venue = metadata.get("venue", "No Venue")
    year = metadata.get("year", "No Year")
    authors = metadata.get("authors", [])

    # Generate author names in APA format
    author_names = []
    for author in authors[:5]:  # Limit to first 5 authors
        parts = author.split(" ")
        if len(parts) > 1:
            last_name = parts[0]
            first_initials = " ".join(name[0] + "." for name in parts[1:])
            author_names.append(f"{last_name}, {first_initials}")
        else:
            author_names.append(author)

    authors_string = ", & ".join(author_names)

    # APA citation format: Author1, Author2, & Author3. (Year). Title. Venue.
    citation = f"{authors_string}. ({year}). **{title}**. {venue}."

    return citation


def extract_numbers_in_brackets(input_string: str) -> List[int]:
    """
    Extract citation numbers from text in [number] format.

    This function extracts reference indices from LLM-generated text
    that includes citations in the format [1], [2], etc.

    Args:
        input_string: Text containing citation references

    Returns:
        Sorted list of unique citation numbers
    """
    # Find all occurrences of [content]
    numbers_in_brackets = re.findall(r"\[(.*?)\]", input_string)

    # Convert to integers, skipping non-numeric values
    cleaned_numbers = []
    for n in numbers_in_brackets:
        try:
            cleaned_numbers.append(int(n))
        except ValueError:
            continue

    # Return sorted unique numbers
    return sorted(list(set(cleaned_numbers)))


def generate_used_reference_display(source_nodes: List[Any], used_nodes: List[int]) -> str:
    """
    Generate a formatted display of used references.

    Args:
        source_nodes: List of source nodes from query response
        used_nodes: List of citation indices that were used

    Returns:
        Markdown-formatted reference display string
    """
    reference_display = "\n #### 📚 References: \n"

    # For each used index, get the source node and add to display
    for index in used_nodes:
        try:
            source_node = source_nodes[index - 1]
        except IndexError:
            return "\n #### 😞 Couldn't Parse References \n"

        metadata = source_node.node.metadata
        reference_display += (
            "[["
            + str(source_nodes.index(source_node) + 1)
            + "]"
            + "("
            + "https://www.semanticscholar.org/paper/"
            + metadata["paperId"]
            + ")] "
            + "\n `. . ."
            + str(source_node.node.text)[100:290]
            + ". . .`"
            + get_citation(metadata)
            + " \n\n"
        )

    return reference_display


def generate_reference_display(source_nodes: List[Any]) -> str:
    """
    Generate a formatted display of all references.

    Args:
        source_nodes: List of all source nodes from query response

    Returns:
        Markdown-formatted reference display string
    """
    reference_display = "\n ### References: \n"

    for source_node in source_nodes:
        metadata = source_node.node.metadata
        reference_display += (
            "[["
            + str(source_nodes.index(source_node) + 1)
            + "]"
            + "("
            + "https://www.semanticscholar.org/paper/"
            + metadata["paperId"]
            + ")] "
            + get_citation(metadata)
            + " \n\n"
        )

    return reference_display
