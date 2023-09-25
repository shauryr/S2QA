import logging
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import requests
from typing import List
import re
import os
import logging
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import requests
from typing import List
import os
import pandas as pd
import openai
import ast

TWITTER_USERNAME = "shauryr"

def generate_search_queries_prompt(question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """

    return (
        f'Please generate four related search queries that align with the initial query: "{question}"'
        f'Each variation should be presented as a list of strings, following this format: ["query 1", "query 2", "query 3", "query 4"]'
    )


def get_related_questions(query):
    research_template = """You are a search engine expert"""
            
    messages = [{
                "role": "system",
                "content": research_template
            }, {
                "role": "user",
                "content": generate_search_queries_prompt(query),
            }]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.5,
    max_tokens=256
    )
    related_questions = get_questions(response.choices[0].message.content)
    related_questions.append(query)
    return related_questions

def get_questions(response_text):
    data = response_text.split("\n")
    data = [ast.literal_eval(item)[0] for item in data]
    return data

def get_unique_docs(docs):
    unique_docs_id = []
    unique_docs = []
    for doc in docs:
        if doc.extra_info['paperId'] not in unique_docs:
            unique_docs_id.append(doc.extra_info['paperId'])
            unique_docs.append(doc)
    return unique_docs

class SemanticScholarReader(BaseReader):
    """
    A class to read and process data from Semantic Scholar API
    ...

    Methods
    -------
    __init__():
       Instantiate the SemanticScholar object

    load_data(query: str, limit: int = 10, returned_fields: list = ["title", "abstract", "venue", "year", "paperId", "citationCount", "openAccessPdf", "authors"]) -> list:
        Loads data from Semantic Scholar based on the query and returned_fields

    """

    def __init__(self, timeout=10, api_key=None, base_dir="pdfs"):
        """
        Instantiate the SemanticScholar object
        """
        from semanticscholar import SemanticScholar
        import arxiv

        self.arxiv = arxiv
        self.base_dir = base_dir
        self.s2 = SemanticScholar(timeout=timeout)
        # check for base dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _clear_cache(self):
        """
        delete the .citation* folder
        """
        import shutil

        shutil.rmtree("./.citation*")

    def _download_pdf(self, paper_id, url: str, base_dir="pdfs"):
        logger = logging.getLogger()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        # Making a GET request
        response = requests.get(url, headers=headers, stream=True)
        content_type = response.headers["Content-Type"]

        # As long as the content-type is application/pdf, this will download the file
        if "application/pdf" in content_type:
            os.makedirs(base_dir, exist_ok=True)
            file_path = os.path.join(base_dir, f"{paper_id}.pdf")
            # check if the file already exists
            if os.path.exists(file_path):
                logger.info(f"{file_path} already exists")
                return file_path
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            logger.info(f"Downloaded pdf from {url}")
            return file_path
        else:
            logger.warning(f"{url} was not downloaded: protected")
            return None

    def _get_full_text_docs(self, documents: List[Document]) -> List[Document]:
        from PyPDF2 import PdfReader

        """
        Gets the full text of the documents from Semantic Scholar

        Parameters
        ----------
        documents: list
            The list of Document object that contains the search results

        Returns
        -------
        list
            The list of Document object that contains the search results with full text

        Raises
        ------
        Exception
            If there is an error while getting the full text

        """
        full_text_docs = []
        for paper in documents:
            metadata = paper.extra_info
            url = metadata["openAccessPdf"]
            externalIds = metadata["externalIds"]
            paper_id = metadata["paperId"]
            file_path = None
            persist_dir = os.path.join(self.base_dir, f"{paper_id}.pdf")
            if url and not os.path.exists(persist_dir):
                # Download the document first
                file_path = self._download_pdf(metadata["paperId"], url, persist_dir)

            if (
                not url
                and externalIds
                and "ArXiv" in externalIds
                and not os.path.exists(persist_dir)
            ):
                # download the pdf from arxiv
                file_path = self._download_pdf_from_arxiv(
                    paper_id, externalIds["ArXiv"]
                )

            # Then, check if it's a valid PDF. If it's not, skip to the next document.
            if file_path:
                try:
                    pdf = PdfReader(open(file_path, "rb"))
                except Exception as e:
                    logging.error(
                        f"Failed to read pdf with exception: {e}. Skipping document..."
                    )
                    continue

                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                full_text_docs.append(Document(text=text, extra_info=metadata))

        return full_text_docs

    def _download_pdf_from_arxiv(self, paper_id, arxiv_id):
        paper = next(self.arxiv.Search(id_list=[arxiv_id], max_results=1).results())
        paper.download_pdf(dirpath=self.base_dir, filename=paper_id + ".pdf")
        return os.path.join(self.base_dir, f"{paper_id}.pdf")

    def load_data(
        self,
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
        results = []
        query = get_related_questions(query)
        try:
            for question in query:
                logging.info(f"Searching for {question}")
                _results = self.s2.search_paper(question, limit=limit, fields=returned_fields)
                results.extend(_results[:limit])
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            logging.error(
                "Failed to fetch data from Semantic Scholar with exception: %s", e
            )
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

        documents = []
        
        for item in results[:limit*len(query)]:
            openAccessPdf = getattr(item, "openAccessPdf", None)
            abstract = getattr(item, "abstract", None)
            title = getattr(item, "title", None)
            text = None
            # concat title and abstract
            if abstract and title:
                text = title + " " + abstract
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
            documents.append(Document(text=text, extra_info=metadata))

        if full_text:
            logging.info("Getting full text documents...")
            full_text_documents = self._get_full_text_docs(documents)
            documents.extend(full_text_documents)
        
        documents = get_unique_docs(documents)
        
        return documents


def get_twitter_badge():
    """Constructs the Markdown code for the Twitter badge."""
    return f'<a href="https://twitter.com/{TWITTER_USERNAME}" target="_blank"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" /></a>'


def get_link_tree_badge():
    return f'<a href="https://linktr.ee/shauryr" target="_blank"><img src="https://img.shields.io/badge/Linktree-39E09B?style=for-the-badge&logo=linktree&logoColor=white" /></a>'

def get_github_badge():
    return f'<a href="https://github.com/shauryr/s2qa" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" /></a>'

def display_questions(sample_questions):
    s = "#### 🧐 More questions? \n"
    for i in sample_questions:
        s += "- " + i + "\n"

    return s


def get_citation(metadata):
    # Extract details from metadata
    title = metadata.get("title", "No Title")
    venue = metadata.get("venue", "No Venue")
    year = metadata.get("year", "No Year")
    authors = metadata.get("authors", [])

    # Generate author names in correct format
    author_names = []
    for author in authors[:5]:
        last_name, *first_names = author.split(" ")
        first_initials = " ".join(name[0] + "." for name in first_names)
        author_names.append(f"{last_name}, {first_initials}")

    authors_string = ", & ".join(author_names)

    # APA citation format: Author1, Author2, & Author3. (Year). Title. Venue.
    citation = f"{authors_string}. ({year}). **{title}**. {venue}."

    return citation


def extract_numbers_in_brackets(input_string):
    # use regular expressions to find all occurrences of [number]
    # numbers_in_brackets = re.findall(r"\[(\d+)\]", input_string)
    numbers_in_brackets = re.findall(r"\[(.*?)\]", input_string)
    # numbers_in_brackets = [int(i) for num in numbers_in_brackets for i in num.split(",")]
    # convert all numbers to int and remove duplicates by converting list to set and then back to list
    cleaned_numbers = []
    for n in numbers_in_brackets:
    # Try to convert the value to an integer
        try:
            cleaned_numbers.append(int(n))
        # If it fails (throws a ValueError), just ignore and continue with the next value
        except ValueError:
            continue

    # Apply the rest of your code on the cleaned list
    return sorted(list(set(cleaned_numbers)))


def generate_used_reference_display(source_nodes, used_nodes):
    reference_display = "\n #### 📚 References: \n"
    # for index in used_nodes get the source node and add it to the reference display
    for index in used_nodes:
        source_node = source_nodes[index - 1]
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

def documents_to_df(documents):
    # convert document objects to dataframe
    list_data = []
    for i, doc in enumerate(documents):
        list_data.append(doc.extra_info.copy())
    
    df = pd.DataFrame(list_data)
    return df
        

def generate_reference_display(source_nodes):
    reference_display = "\n ### References: \n"
    for source_node in source_nodes:
        metadata = source_node.node.metadata
        # add number infront of citation to make it easier to reference
        # reference_display += (
        #     "[["
        #     + str(source_nodes.index(source_node) + 1)
        #     + "]"
        #     + "("
        #     + "https://www.semanticscholar.org/paper/"
        #     + metadata["paperId"]
        #     + ")] "
        #     + '\n "`. . .'
        #     + str(source_node.node.text)[100:290]
        #     + ". . .` - **"
        #     + get_citation(metadata)
        #     + "** \n\n"
        # )
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
