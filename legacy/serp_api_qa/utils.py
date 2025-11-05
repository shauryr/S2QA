import constants
import requests
import nltk
from transformers import AutoTokenizer
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from IPython.display import display, Markdown
import streamlit as st
import tqdm as tqdm
from pathlib import Path
import json
from urllib.parse import urlparse

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
K = 10

URL = "http://localhost:5002/predict"

def get_pdfs(df, folder):
    import requests
    import os
    for i in range(len(df)):
        url = df.iloc[i]['openAccessPdf']
        if url is not None:
            filename =df.iloc[i]['paperId']+ '.pdf'
            url = url['url']
            if not os.path.exists(folder + filename):
                try:
                    r = requests.get(url, allow_redirects=True)
                    open(folder + filename, 'wb').write(r.content)
                except Exception as e:
                    print(e)
                    print('error downloading pdf')

def load_data(file) :
    """Parse file."""
    import PyPDF2

    text_list = []
    # check if file exists
    if not Path(file).exists():
        return None
    
    with open(file, "rb") as fp:
        # Create a PDF object
        try:
            pdf = PyPDF2.PdfReader(fp)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # Iterate over every page
            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                text_list.append(page_text)
        except:
            return None
    text = "\n".join(text_list)

    return text



def split_text_to_chunks(text, model_name="allenai/specter2", tokens_per_chunk=512):
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

# function to convert paragraphs in to (paperid, paragraph) tuples
def get_paragraphs(df):
    paragraphs = []
    for i in range(len(df)):
        if df.iloc[i]['paragraphs'] is not None:
            for paragraph in df.iloc[i]['paragraphs']:
                paragraphs.append((df.iloc[i]['paperId'], paragraph))
    return paragraphs

# function to get (paperid, title+abstract) tuples
def get_titles_abstracts(df):
    titles_abstracts = []
    for i in range(len(df)):
        if df.iloc[i]['title'] is not None and df.iloc[i]['abstract'] is not None:
            titles_abstracts.append((df.iloc[i]['paperId'], df.iloc[i]['title']+ tokenizer.sep_token + df.iloc[i]['abstract']))
    return titles_abstracts

def get_embeddings(text):
    """Sends a request to the server to get the summary of the given text."""
    data = {
    "text": text
    }

    response = requests.post(URL, json=data)
    return  json.loads(response.text).get("embeddings")

def get_tldr(text):
    """Sends a request to the server to get the summary of the given text."""
    data = {"text": text}
    URL = "http://localhost:5001/predict"
    response = requests.post(URL, json=data)
    return response.json()["tldr"]


def generate_prompt(df, query):
    """Generates a prompt for the model to answer the question."""
    return answer_question_chatgpt(
        df,
        query,
        k=K,
        instructions="Instructions: Using the provided web search results, write a comprehensive reply to the given query. If you find a result relevant make sure to cite the result using [[number](URL)] notation after the reference. End your answer with a summary.\nQuery:",
    )


def generate_answer(prompt):
    """Generates an answer using ChatGPT."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to a researcher. You are helping them write a paper. You are given a prompt and a list of references. You are asked to write a summary of the references if they are related to the question. You should not include any personal opinions or interpretations in your answer, but rather focus on objectively presenting the information from the search results.",
            },
            {"role": "user", "content": prompt},
        ],
        api_key=constants.OPENAI_API_KEY,
    )
    return response.choices[0].message.content


def search(query, limit=20, fields=["title", "abstract", "venue", "year", "openAccessPdf"]):
    # space between the  query to be removed and replaced with +
    query = query.replace(" ", "+")
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields={",".join(fields)}'
    headers = {"Accept": "*/*", "x-api-key": constants.S2_KEY}

    response = requests.get(url, headers=headers, timeout=30)
    return response.json()


def get_results(query, limit=20):
    """ """
    search_results = search(preprocess_query(query), limit)

    if search_results["total"] == 0:
        print("No results found - Try another query")
    else:
        # drop rows with missing abstracts and titles
        df = pd.DataFrame(search_results["data"])
        df = df.dropna(subset=["title"])
        # replace NA with empty string
        
        
    return df


def get_doc_objects_from_df(df):
    """
    Get a list of Document objects from a dataframe
    """
    doc_objects = []
    for i, row in df.iterrows():
        doc_object = langchain.docstore.document.Document(
            page_content=row["abstract"],
            metadata={"source": row["paperId"]},
            lookup_index=i,
        )
        doc_objects.append(doc_object)
    return doc_objects


def rerank(df, query, column_name="title_abs"):
    # merge columns title and abstract into a string separated by tokenizer.sep_token and store it in a list
    df["title_abs"] = [
        d["title"] + tokenizer.sep_token + (d.get("abstract") or "")
        for d in df.to_dict("records")
    ]

    df["n_tokens"] = df.title_abs.apply(lambda x: len(tokenizer.encode(x)))
    doc_embeddings = get_specter_embeddings(list(df[column_name]))
    query_embeddings = get_specter_embeddings(query)
    df["specter_embeddings"] = list(doc_embeddings)
    df["similarity"] = cosine_similarity(query_embeddings, doc_embeddings).flatten()

    # sort the dataframe by similarity
    df.sort_values(by="similarity", ascending=False, inplace=True)
    return df, query


# function to preprocess the query and remove the stopwords before passing it to the search function
def preprocess_query(query):
    query = query.lower()
    # remove stopwords from the query
    stopwords = set(nltk.corpus.stopwords.words("english"))
    # add words to the stopwords list
    stopwords.update(["please", "review"])
    query = " ".join([word for word in query.split() if word not in stopwords])
    return query


def get_specter_embeddings(text):
    # tokenize the text
    tokens = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    # get the embeddings
    embeddings = model(**tokens).pooler_output
    # return the embeddings
    return embeddings.detach().numpy()


def create_context(question, df, max_len=3800, size="davinci"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.iterrows():

        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["title_abs"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="text-davinci-003",
    question="What is the impact of creatine on cognition?",
    max_len=3800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f'Answer the question based on the context below"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:',
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


def get_langchain_response(docs, query, k=5):
    """
    Get the langchain response for a query. Here we are using the langchain mapreduce function to get the response.
    Prompts here should be played around with. These are the prompts that worked best for us.
    """
    question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 

    {context}
    Question: {question}
    Relevant text, if any:"""
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """Given the following extracted parts of a scientific paper and a question.  
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Create a final answer with references ("SOURCES")
    ALWAYS return a "SOURCES" part at the end of your answer. Return sources as a list of strings, e.g. ["source1", "source2", ...]

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0, openai_api_key=constants.OPENAI_API_KEY),
        chain_type="map_reduce",
        return_intermediate_steps=True,
        question_prompt=QUESTION_PROMPT,
        combine_prompt=COMBINE_PROMPT,
    )
    chain_out = chain(
        {"input_documents": docs[:k], "question": query}, return_only_outputs=True
    )
    return chain_out


def return_answer_markdown(chain_out, df, query):
    """
    Parse the output_text and sources from the chain_out JSON and return a markdown string
    """
    output_text = chain_out["output_text"].split("\n\nSOURCES: ")[0].strip()
    if chain_out["output_text"].endswith("]"):
        sources = eval(chain_out["output_text"].split("SOURCES:")[1].strip())
    else:
        sources = eval(chain_out["output_text"].split("SOURCES:")[1].strip() + '"]')

    # Creating a new JSON with the extracted output_text and sources
    output_text = {"output_text": output_text, "sources": sources}

    # Printing the new JSON
    display(Markdown(f"## Question\n\n"))

    display(Markdown(f"### {query}\n\n"))

    display(Markdown(f"## Answer\n\n"))

    display(Markdown(f"### {output_text['output_text']}\n\n"))

    display(Markdown(f"## Sources: \n\n"))

    # markdown headings for each source
    for source in output_text["sources"]:
        try:
            title = df[df["paperId"] == source]["title"].values[0]
            link = f"https://www.semanticscholar.org/paper/{source}"
            venue = df[df["paperId"] == source]["venue"].values[0]
            year = df[df["paperId"] == source]["year"].values[0]
            display(Markdown(f"* #### [{title}]({link}) - {venue}, {year}"))
        except:
            display(Markdown(f"Source not found: {source}"))

def print_papers(df, k=8):
    count = 1
    for i in range(k):
        # add index
        title = df.iloc[i]["title"]
        link = f"https://www.semanticscholar.org/paper/{df.iloc[i]['paperId']}"
        venue = df.iloc[i]["venue"]
        year = df.iloc[i]["year"]
        display(Markdown(f"#### {[count]} [{title}]({link}) - {venue}, {year}"))
        count+=1

def print_papers_streamlit(df, k=8):
    count = 1
    for i in range(k):
        # add index
        title = df.iloc[i]["title"]
        link = f"https://www.semanticscholar.org/paper/{df.iloc[i]['paperId']}"
        venue = df.iloc[i]["venue"]
        year = df.iloc[i]["year"]
        st.markdown(f"{[count]} [{title}]({link}) - {venue}, {year}")
        count+=1


def answer_question_chatgpt(
    df,
    question="What is the impact of creatine on cognition?",
    k=5,
    instructions="Instructions: Using the provided web search results, write a comprehensive reply to the given query. If you find a result relevant definitely make sure to cite the result using [[number](URL)] notation after the reference. End your answer with a summary in a new line. A\nQuery:",
    max_len=3000,
    debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context_chatgpt(question, df, k=k)

    try:
        # Create a completions using the question and context
        # prompt = f'''{context} \n\n Instructions: Using the provided literature with sources, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. You can skip a citation which you dont find relevant to the query. \nQuery:{question}\nAnswer:'''
        prompt = f"""{context} \n\n{instructions} {question}\nAnswer:"""
        return prompt
    except Exception as e:
        print(e)
        return ""


def create_context_chatgpt(question, df, k=5):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    returns = []
    count = 1
    # Sort by distance and add the text to the context until the context is too long
    for i, row in df[:k].iterrows():

        # Else add it to the text that is being returned
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

def get_google_results(query):
    # url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}'
    url = f"https://www.googleapis.com/customsearch/v1?key={constants.google_api_key}&cx={constants.search_engine_id}&q={query}&num=10"

    # Make the API request and retrieve the results
    response = requests.get(url).json()
    results = response.get("items", [])
    df = pd.DataFrame(results)
    return df

def get_semantic_scholar_id(url):
    path = urlparse(url).path
    try:
        return path.split("/")[3]
    except:
        return path.split("/")[2]

# a fuction which can do this https://pdfs.semanticscholar.org/7d95/0518907ef1b1027aed8479601fa11a02883e.pdf -> 7d950518907ef1b1027aed8479601fa11a02883e
def get_semantic_scholar_pdf_id(url):
    path = urlparse(url).path
    # keep the last 2 parts of the path and strip the .pdf
    return path.split("/")[-2] + path.split("/")[-1][:-4]


def controller_id_function(url):
    # if url starts with https://www then call get_semantic_scholar_id else call get_semantic_scholar_pdf_id
    if url.startswith("https://www"):
        return get_semantic_scholar_id(url)
    else:
        return get_semantic_scholar_pdf_id(url)
    
def get_paper_info(
    paperId, fields=["title", "abstract", "venue", "year", "openAccessPdf"]
):
    url = f'https://api.semanticscholar.org/graph/v1/paper/{paperId}?fields={",".join(fields)}'
    headers = {"Accept": "*/*", "x-api-key": constants.S2_KEY}
    response = requests.get(url, headers=headers, timeout=30)
    return response.json()

def get_cosine_similarity(query_embedding, text_embedding):
    return cosine_similarity(query_embedding, text_embedding)[0][0]

def printmd(string):
    display(Markdown(string))
