import constants
import requests
import nltk
from transformers import AutoTokenizer, AutoModel
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from IPython.display import display, Markdown
import streamlit as st


tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
model = AutoModel.from_pretrained("allenai/specter2")


def search(query, limit=20, fields=["title", "abstract", "venue", "year"]):
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
    instructions="Instructions: Using the provided web search results, write a comprehensive reply to the given query. If you find a result relevant definitely make sure to cite the result using [[number](URL)] notation after the reference. End your answer with a summary. A\nQuery:",
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
