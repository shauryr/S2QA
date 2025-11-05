# Constants
import requests
import streamlit as st
from utils import print_papers_streamlit

URL = "http://localhost:5001/predict"
GITHUB_URL = "https://api.github.com/repos/shauryr/S2QA"
K = 10
TWITTER_USERNAME = "shauryr"

def get_github_badge():
    """Sends a GET request to the GitHub API to retrieve information about the repository
    and constructs the Markdown code for the GitHub badge with the number of stars."""
    response = requests.get(GITHUB_URL)
    return f"[![GitHub stars](https://img.shields.io/github/stars/shauryr/S2QA?style=social)](https://github.com/shauryr/S2QA)"


def get_twitter_badge():
    """Constructs the Markdown code for the Twitter badge."""
    return f'<a href="https://twitter.com/{TWITTER_USERNAME}" target="_blank"><img src="https://img.shields.io/twitter/follow/{TWITTER_USERNAME}?style=social&logo=twitter" /></a>'

# show powered by icons
def display_powered_by():
    """Displays the powered by Streamlit and Hugging Face badges."""
    st.markdown(
        "<h5 style='text-align: left;'>Powered by: </h5>", unsafe_allow_html=True
    )
    st.markdown(
        """
        <a href='https://programmablesearchengine.google.com/about/' target='https://programmablesearchengine.google.com/about/'><img src='https://seeklogo.com/images/G/google-custom-search-logo-DD7DDA2440-seeklogo.com.png' alt='Powered by Google Search' width='35' style='margin-right: 20px;' /></a>
        <a href='https://semanticscholar.org' target='https://semanticscholar.org'><img src='https://pbs.twimg.com/profile_images/1304515818219216897/ns73Z_GS_400x400.png' alt='Powered by semanticscholar' width='35' style='margin-right: 20px;' /></a>
        <a href='https://openai.com' target='https://openai.com'><img src='https://www.drupal.org/files/project-images/openai-avatar.png' alt='Powered by openai' width='35' style='margin-right: 20px;' /></a>
        """,
        unsafe_allow_html=True,
    )

def display_badges():
    """Displays the GitHub and Twitter badges in Streamlit."""
    github_badge = get_github_badge()
    twitter_badge = get_twitter_badge()
    st.markdown(f"{github_badge}{twitter_badge}", unsafe_allow_html=True)

def display_description():
    """Displays the description of the app."""
    # st.markdown("<h4 style='text-align: left;'>Get answers to your questions from 200M+ research papers from Semantic Scholar, summarized by ChatGPT</h4>", unsafe_allow_html=True)
    
    st.write(
        "<h5 style='text-align: left;'>🤩 Offering Google custom search ranking and full text search!</h5>",
        unsafe_allow_html=True,
    )

    st.write(
        "<h5 style='text-align: left;'>🏖️ Relax while a robot writes your lit review</h5>",
        unsafe_allow_html=True,
    )

    st.write(
        "<h5 style='text-align: left; '>✨The citations here are not hallucinated.</h5>",
        unsafe_allow_html=True,
    )
    # st.write(
    #     """
    #     Why use this tool?
    #     - 👉 Get research overview of a topic
    #     - 👉 Find papers relevant to your research
    #     """
    # )

def display_warning():
    """Displays a warning message in small text"""
    st.write(
        "<h8 style='text-align: left;'>⚠️ Warning: This is a research prototype hosted on a graduate student's local machine. It is not meant for production use.</h8>",
        unsafe_allow_html=True,
    )

def display_references(df):
    """Displays the references."""
    st.markdown(
        "<h4 style='text-align: left;'>📚 Used References: </h4>", unsafe_allow_html=True
    )
    print_papers_streamlit(df, k=df.shape[0]) if df.shape[
        0
    ] < K else print_papers_streamlit(df, k=K)

def display_references_temp(df):
    """Displays the references."""
    st.markdown(
        "<h4 style='text-align: left;'>📚 Used References: </h4>", unsafe_allow_html=True
    )
    print_papers_streamlit_temp(df, k=df.shape[0]) if df.shape[
        0
    ] < K else print_papers_streamlit_temp(df, k=K)

def print_papers_streamlit_temp(df, k=8):
    count = 1
    for i in range(k):
        # add index
        title = df.iloc[i]["title"]
        link = f"https://www.semanticscholar.org/paper/{df.iloc[i]['paperId']}"
        venue = df.iloc[i]["venue"]
        year = df.iloc[i]["year"]
        st.markdown(f" [{title}]({link}) - {venue}, {year}")
        count+=1

def get_session_info():
    # get session info
    session = requests.get("http://ip-api.com/json").json()
    return session

def display_known_issues():
    """Displays the known issues"""
    with st.expander("⚠️ Known Issues:", expanded=False):
        st.markdown(
            """
            - 🚨 If search fails to get relevant papers, the model may not be able to generate a good answer
            - 🚨 Full-text of arxiv papers is not available for now
            - 🚨 If the model generates a bad answer, try rephrasing the question
            - 🚨 Please verify the answer before using it in your paper, although real sources are cited the text itself might be hallucinated(rare)
            """
        )

def display_why_no_hallucinations():
    """Displays the known issues"""
    with st.expander("🤔 How is it working? ", expanded=False):
        st.markdown(
            """
                - compile a list of papers using Google search and Semantic Scholar
                - download open access papers from the list
                - summarize the content of the downloaded papers
                - select the top ten summaries that are most relevant
                - provide these summaries as context to ChatGPT for generating an answer"""
        )