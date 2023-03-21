
import streamlit as st
import requests
from tqdm import tqdm
from IPython.display import Markdown, display
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import openai
from utils import *
import constants
import time
import json
import streamlit as st
import requests

tqdm.pandas()

# Constants
URL = "http://localhost:5001/predict"
GITHUB_URL = "https://api.github.com/repos/shauryr/S2QA"
K = 8
TWITTER_USERNAME = "shauryr"

def get_github_badge():
    """Sends a GET request to the GitHub API to retrieve information about the repository 
    and constructs the Markdown code for the GitHub badge with the number of stars."""
    response = requests.get(GITHUB_URL)
    return f"[![GitHub stars](https://img.shields.io/github/stars/shauryr/S2QA?style=social)](https://github.com/shauryr/S2QA)"

def get_twitter_badge():
    """Constructs the Markdown code for the Twitter badge."""
    return f'<a href="https://twitter.com/{TWITTER_USERNAME}" target="_blank"><img src="https://img.shields.io/twitter/follow/{TWITTER_USERNAME}?style=social&logo=twitter" /></a>'

def display_badges():
    """Displays the GitHub and Twitter badges in Streamlit."""
    github_badge = get_github_badge()
    twitter_badge = get_twitter_badge()
    st.markdown(f"{github_badge}{twitter_badge}", unsafe_allow_html=True)

def display_description():
    """Displays the description of the app."""
    # st.markdown("<h4 style='text-align: left;'>Get answers to your questions from 200M+ research papers from Semantic Scholar, summarized by ChatGPT</h4>", unsafe_allow_html=True)
    st.write("<h5 style='text-align: left;'>🏖️ Relax while a robot writes your lit review</h5>", unsafe_allow_html=True)

    st.write("<h5 style='text-align: left; '>✨The citations here are not hallucinated.</h5>", unsafe_allow_html=True)
    st.write("""
        Why use this tool?
        - 👉 Get research overview of a topic
        - 👉 Find papers relevant to your research
        """)

def display_warning():
    """Displays a warning message in small text"""
    st.write("<h8 style='text-align: left;'>⚠️ Warning: This is a research prototype hosted on a graduate student's local machine. It is not meant for production use.</h8>", unsafe_allow_html=True)

def get_response(text):
    """Sends a request to the server to get the summary of the given text."""
    data = {"text": text}
    response = requests.post(URL, json=data)
    return response.json()["tldr"]

def generate_prompt(df, query):
    """Generates a prompt for the model to answer the question."""
    return answer_question_chatgpt(
        df,
        query,
        k=K,
        instructions="Instructions: Using the provided web search results, write a comprehensive reply to the given query. If you find a result relevant make sure to cite the result using [[number](URL)] notation after the reference. End your answer with a summary.\nQuery:"
    )

def generate_answer(prompt):
    """Generates an answer using ChatGPT."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
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


def display_references(df):
    """Displays the references."""
    st.markdown("<h4 style='text-align: left;'>📚 Used References: </h4>", unsafe_allow_html=True)
    print_papers_streamlit(df, k=df.shape[0]) if df.shape[0] < K else print_papers_streamlit(df, k=K)

def get_session_info():
    # get session info
    session = requests.get("http://ip-api.com/json").json()
    return session

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


def display_known_issues():
    """Displays the known issues"""
    with st.expander("⚠️ Known Issues:", expanded=False):
        st.markdown("""
            - 🚨 If search fails to get relevant papers, the model may not be able to generate a good answer
            - 🚨 If the model generates a bad answer, try rephrasing the question
            - 🚨 Please verify the answer before using it in your paper, although real sources are cited the text itself might be hallucinated(rare)
            """)

def app():
    """Main function that runs the Streamlit app."""
    st.markdown("<h2 style='text-align: left;'>🚀 S2QA (beta): ChatGPT for Research 📚🤖</h2>", unsafe_allow_html=True)


    display_badges()
    display_description()

    # Get the query from the user and sumit button 
    query = st.text_input("Enter your research question here and press Generate Answer:")
   
    # Add the button to the empty container
    button = st.button("Generate Answer")
    st.markdown("<h7 style='text-align: left;'>🚨 Generating an answer may take approximately 60 seconds as GPT-4 is slow</h7>", unsafe_allow_html=True)

    if query and button:
        try:
            # Get the results from Semantic Scholar
            with st.spinner("⏳ Getting papers from Semantic Scholar ..."):
                df = get_results(query, limit=20)
            st.success(f"Got {df.shape[0]} related papers from Semantic Scholar 🎉")
            # st.dataframe(df[["title", "abstract", "venue"]].head())
            df = df.dropna(subset=["abstract"])
            df = df.fillna("")
        except:
            st.write("No results found for the query. Please try again with a different query 🏖️ OR the search API is down. Please try again later.") 
            dump_logs(query, '',success=False)
            if st.button("Reload"):
                st.experimental_rerun()
            st.stop()
            
        start_time = time.time()
        # st.success(f"Removing papers with no abstracts 🗑️")
        with st.spinner("⏳ Re-ranking search results ..."):
            df, query = rerank(df, query)

        # elapsed_time = time.time() - start_time
        # st.success(f"🙌 Re-ranking finished! 🕰️ Time taken {elapsed_time:.2f} seconds.")

        df = df[:K]
        with st.spinner(f"⏳ Generating summaries for top-{df.shape[0]} abstracts ..."):
            df["tldr"] = df["title_abs"].progress_apply(get_response)
        # elapsed_time = time.time() - start_time
        # st.success(f"🙌 Summaries extracted successfully! 🕰️ Time taken {elapsed_time:.2f} seconds.")

        # Generate prompt for the model to answer the question
        prompt = generate_prompt(df, query)

        try:
            with st.spinner("GPT-4 is generating your answer ..."):
                response = generate_answer(prompt)
                elapsed_time = time.time() - start_time
                st.success(f"🙌 ChatGPT finished generating an answer!  🕰️ Time taken {elapsed_time:.2f} seconds.")
                st.markdown("## ❓Question:")
                st.markdown(f"### {query}")
                st.markdown("## 🤖 Generated Answer:")
                st.markdown(response)
                dump_logs(query, response, success=True)
        except:
            st.write("Error generating answer using ChatGPT. Please reload below and try again")
            dump_logs(query, '',success=False)
            if st.button("Reload"):
                st.experimental_rerun()
            st.stop()


        display_references(df)

    # Add content to the footer
    # footer_container = st.container()

    # Add two columns to the footer container
    # left_column = footer_container.columns(1)[0]

    # Display sample questions
    with st.expander("❓❓ Here are some questions which you can ask:", expanded=False):
        st.markdown("""
            - How can we improve the interpretability and transparency of complex machine learning models, and what are the implications for ensuring ethical and responsible AI development?
            - Do Language Models Plagiarize?
            - How does iron supplementation affect anemia?
            - Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
            """)

    # encourage people to share their feedback and open issues on the following link 

    display_known_issues()
    display_warning()

    st.write("Made with ❤️ by [Shaurya Rohatgi](https://linktr.ee/shauryr) 📜 [Privacy Policy](https://www.termsfeed.com/live/5864cf7e-39e9-4e48-a014-c16ba54e08ea)")

if __name__ == '__main__':
    app()
