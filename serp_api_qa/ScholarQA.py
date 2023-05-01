"""
streamlit run ScholarQA.py --server.fileWatcherType none
"""
from tqdm import tqdm
from IPython.display import Markdown, display
import openai
import constants
import time
import json
import streamlit as st
from display_utils import (
    display_badges,
    display_description,
    display_known_issues,
    display_references_temp,
    display_powered_by,
    display_why_no_hallucinations
)
from IPython.display import Markdown, display
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    get_semantic_scholar_id,
    get_pdfs,
    load_data,
    split_text_to_chunks,
    get_paragraphs,
    get_titles_abstracts,
    get_embeddings,
    get_google_results,
    generate_answer,
    generate_prompt,
    get_tldr,
    create_context_chatgpt,
    answer_question_chatgpt,
    get_semantic_scholar_pdf_id,
    controller_id_function,
    get_cosine_similarity,
    get_paper_info,
    answer_question_chatgpt,
    print_papers_streamlit,
    get_results,
    rerank,
)
from urllib.parse import urlparse
import requests


# Constants
URL = "http://localhost:5001/predict"
GITHUB_URL = "https://api.github.com/repos/shauryr/S2QA"
K = 8
TWITTER_USERNAME = "shauryr"


# Call the function to get the session info
def dump_logs(query, response, success=True):

    # session = get_session_info()
    # Create a dictionary of query details
    query_details = {
        # "session": session,
        "timestamp": time.time(),
        "query": query,
        "response": response,
    }

    if success:
        # Append the query details to a JSON file
        with open("query_details.json", "a") as f:
            json.dump(query_details, f)
            f.write("\n")
    else:
        # Append the query details to a JSON file
        with open("query_details_error.json", "a") as f:
            json.dump(query_details, f)
            f.write("\n")


def extract_paper_ids(df):
    df["paperId"] = df["link"].apply(controller_id_function)
    return df


def add_paper_info(df):
    df["paperInfo"] = df["paperId"].apply(get_paper_info)
    return df


def parse_paper_info(df):
    # drop row if paperInfo is None
    df = df.dropna(subset=["paperInfo"])
    # drop row if title key is not present in paperInfo
    df = df[df["paperInfo"].apply(lambda x: "title" in x)]

    df["title"] = df["paperInfo"].apply(lambda x: x["title"])
    df["abstract"] = df["paperInfo"].apply(lambda x: x["abstract"])
    df["venue"] = df["paperInfo"].apply(lambda x: x["venue"])
    df["year"] = df["paperInfo"].apply(lambda x: x["year"])
    df["openAccessPdf"] = df["paperInfo"].apply(lambda x: x["openAccessPdf"])
    df = df[["title", "abstract", "venue", "year", "openAccessPdf", "paperId"]]
    return df


def add_pdf_paths(df, folder):
    df["pdf_paths"] = df["paperId"].apply(lambda x: folder + x + ".pdf")
    return df


def read_pdfs(df):
    df["text"] = df["pdf_paths"].apply(lambda x: load_data(x))
    return df


def preprocess_text(df):
    df["paragraphs"] = df["text"].apply(
        lambda x: split_text_to_chunks(x) if not pd.isna(x) else None
    )
    return df


def get_all_text(df):
    paragraphs = get_paragraphs(df)
    titles_abstracts = get_titles_abstracts(df)
    all_text = paragraphs + titles_abstracts
    return all_text


def create_text_df(all_text):
    df_text = pd.DataFrame(all_text, columns=["paperId", "text"])
    return df_text


def add_embeddings(df_text):
    df_text["embedding"] = df_text["text"].apply(lambda x: get_embeddings(x))
    return df_text


def get_cosine_similarities(df_text, query_embedding):
    df_text["cosine_similarity"] = df_text["embedding"].apply(
        lambda x: get_cosine_similarity(query_embedding, x)
    )
    return df_text


def sort_by_cosine_similarity(df_text):
    df_text = df_text.sort_values(by=["cosine_similarity"], ascending=False)
    return df_text


def get_top_results(df_text, K):
    df_text = df_text.head(K)
    return df_text


def add_tldrs(df_text):
    df_text["tldr"] = df_text["text"].apply(lambda x: get_tldr(x))
    return df_text


def main(query, K=10):
    df = get_google_results(query)
    df = extract_paper_ids(df)
    df = add_paper_info(df)
    df = parse_paper_info(df)
    get_pdfs(df, "pdfs/")
    df = add_pdf_paths(df, "pdfs/")
    df = read_pdfs(df)
    df = preprocess_text(df)
    all_text = get_all_text(df)
    df_text = create_text_df(all_text)
    df_text = add_embeddings(df_text)
    query_embedding = get_embeddings(query)
    df_text = get_cosine_similarities(df_text, query_embedding)
    df_text = sort_by_cosine_similarity(df_text)
    df_text = get_top_results(df_text, K)
    df_text = add_tldrs(df_text)
    return df_text

def unique_id(df):
    df['unique_id'] = df['paperId'].astype('category').cat.codes
    return df

def generate_prompt(df, query):
    """Generates a prompt for the model to answer the question."""
    return answer_question_chatgpt(
        df,
        query,
        k=K,
        instructions="Instructions: Using the provided web search results, write a comprehensive reply to the given query. If you find a result relevant make sure to cite the result using [[number](URL)] notation after the reference. End your answer with a summary.\nQuery:",
    )



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
            + str(row['unique_id'])
            + "] "
            + row["tldr"]
            + "\nURL: "
            + "https://www.semanticscholar.org/paper/"
            + row["paperId"]
        )
        count += 1
    # Return the context
    return "\n\n".join(returns)


def app():
    """Main function that runs the Streamlit app."""
    st.markdown(
        "<h2 style='text-align: left;'>🚀 S2QA (beta): ChatGPT for Research📚🤖</h2>",
        unsafe_allow_html=True,
    )
    display_powered_by()
    display_badges()
    display_description()
    display_why_no_hallucinations()
    # Get the query from the user and sumit button
    query = st.text_input(
        "Enter your research question here and press Generate Answer:"
    )

    # Add the button to the empty container
    button = st.button("Generate Answer", type="primary")
    st.markdown(
        "<h7 style='text-align: left;'>🚨 Generating an answer may take approximately 60 seconds to download and process full-text PDFs</h7>",
        unsafe_allow_html=True,
    )
    
    if query and button:
        try:
            # Get the results from Semantic Scholar
            with st.spinner("⏳ Getting papers from Semantic Scholar ..."):
                df = get_google_results(query)
                df = extract_paper_ids(df)
                df = add_paper_info(df)
            st.success(f"Got {df.shape[0]} related papers from Semantic Scholar 🎉")
            with st.spinner("⏳ Downloading and parsing PDFs ..."):
                start_time = time.time()
                df = parse_paper_info(df)
                st.dataframe(df[['title', 'paperId']].head())
                get_pdfs(df, "pdfs/")
                df_text = add_pdf_paths(df, "pdfs/")
                df_text = read_pdfs(df_text)
                df_text = preprocess_text(df_text)
                all_text = get_all_text(df_text)
                df_text = create_text_df(all_text)
                df_text = add_embeddings(df_text)
                query_embedding = get_embeddings(query)
                df_text = get_cosine_similarities(df_text, query_embedding)
                df_text = sort_by_cosine_similarity(df_text)
                df_text = get_top_results(df_text, K)
                df_text = add_tldrs(df_text)
                # display head of df_text
                elapsed_time = time.time() - start_time
            st.success("Time taken to read PDFs and abstracts: {:.2f} seconds".format(elapsed_time))
        except Exception as e:
            # print the error
            print(e)
            st.write(
                "No results found for the query. Please try again with a different query 🏖️ OR the search API is down. Please try again later."
            )
            dump_logs(query, "", success=False)
            if st.button("Reload"):
                st.experimental_rerun()
            st.stop()

        # Generate prompt for the model to answer the question
        df_text = unique_id(df_text)
        prompt = generate_prompt(df_text, query)
        st.markdown("### ❓Question:")
        st.markdown(f"#### {query}")
        st.markdown("### 🤖 Generated Answer:")
        try:
            # with st.spinner("GPT-4 is generating your answer ..."):
            res_box = st.empty()

            report = []
            # Looping over the response
            for resp in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant to a researcher. You are helping them write a paper. You are given a prompt and a list of references. You are asked to write a summary of the references if they are related to the question. You should not include any personal opinions or interpretations in your answer, but rather focus on objectively presenting the information from the search results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                api_key=constants.OPENAI_API_KEY,
                stream=True,
                temperature = 0.8
            ):

                token = ""
                response_obj = dict(resp.choices[0].delta)
                if "content" in response_obj:
                    token = response_obj["content"]
                    report.append(token)
                    result = "".join(report).strip()
                    result = result.replace("\n", "")
                    res_box.markdown(f"{result}")

            dump_logs(query, report, success=True)
        except:
            st.write(
                "Error generating answer using ChatGPT. Please reload below and try again"
            )
            dump_logs(query, "", success=False)
            if st.button("Reload"):
                st.experimental_rerun()
            st.stop()

        df_text = df_text["paperId"].unique()
        # only keep the papers that are in the df_text in df
        df = df[df["paperId"].isin(df_text)]
        # print(df.columns)
        display_references_temp(df)

    # Display sample questions
    with st.expander("❓❓ Here are some questions which you can ask:", expanded=False):
        st.markdown(
            """
            - How can we improve the interpretability and transparency of complex machine learning models, and what are the implications for ensuring ethical and responsible AI development?
            - Do Language Models Plagiarize?
            - How does iron supplementation affect anemia?
            - Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
            """
        )

    display_known_issues()

    st.write(
        "Made with ❤️ by [Shaurya Rohatgi](https://linktr.ee/shauryr) 📜 [Privacy Policy](https://www.termsfeed.com/live/5864cf7e-39e9-4e48-a014-c16ba54e08ea)"
    )


if __name__ == "__main__":
    app()
