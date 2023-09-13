import streamlit as st
from backend import (
    create_index,
    get_chat_engine,
    citation_query_engine,
    generate_sample_questions,
)
from utils import (
    get_twitter_badge,
    get_link_tree_badge,
    display_questions,
    extract_numbers_in_brackets,
    generate_used_reference_display,
    documents_to_df
)
import openai
import pandas as pd

# ?show_map=True&selected=asia&selected=america
# {"show_map": ["True"], "selected": ["asia", "america"]}
# ?query=deep%20learning%20for%20nlp&num_papers=50&full_text=True
# TODO update URL with query and num_papers after button press: add share button
# TODO add github logo in same style as twitter and linktree
# TODO [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-custom-subdomain>.streamlit.app)

url_params = st.experimental_get_query_params()
num_papers = 30
query = "large language models"
if "query" in url_params:
    query = url_params["query"][0]
if "num_papers" in url_params:
    num_papers = int(url_params["num_papers"][0])

with st.sidebar:
    st.title("📚🤖 S2QA: Query your Research Space")

    st.markdown(
        f"{get_twitter_badge()}  {get_link_tree_badge()}", unsafe_allow_html=True
    )
    st.markdown("Ask deeper questions about your research space")
    openai_api_key = st.text_input(
        "OpenAI API Key", key="OPENAI_API_KEY", type="password"
    )
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    research_space_box = st.empty()
    # hidden text input placeholder
    research_space_query = st.text_input(
        "Enter your research space, \n e.g. machine learning, large language models, covid 19 vaccine",
        query,
    )
    num_papers = st.slider(
        "Number of papers you want to chat with ", 20, 50, num_papers
    )
    full_text = st.toggle(
        "Full Text (initial setup is slow as we first download the pdfs: default set to 10 papers)"
    )
    if full_text:
        full_text = True
        num_papers = 10
    button = st.button("Set Research Space", type="primary")

if button and research_space_query:
    st.session_state["show_chat"] = True
    with st.sidebar:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        with st.status("🦙🦙 LlaMa's are working together . . ."):
            st.write("Creating Index for research space: " + research_space_query)
            try:
                index, documents = create_index(
                    research_space_query.lower(), num_papers, full_text
                )
            except:
                st.error(
                    "Error creating index. Please check your API key or try reducing the number of papers."
                )
                st.stop()
            st.write("Getting Query Engine ready . . .")
            sample_questions = generate_sample_questions(documents)
            chat_engine = citation_query_engine(index, 10, True, 512)
            st.session_state["chat_engine"] = chat_engine
            st.session_state["documents"] = documents
        st.markdown(display_questions(sample_questions))
        with st.expander("📚 Papers in the index: ", expanded=False):
            st.dataframe(documents_to_df(documents))
        
    st.success(
        "###### 🤖 Summary of Research Space *"
        + research_space_query.lower()
        + "* with "
        + str(num_papers)
        + " papers is ready 🚀"
    )
        
    with st.chat_message("assistant"):
        response = chat_engine.query("elaborate on " + research_space_query)
        full_response = ''
        placeholder = st.empty()
        for text in response.response_gen:
            # Appending response content if available
            full_response += text
            # Displaying the response to the user
            placeholder.markdown(full_response + "▌")
        used_nodes = extract_numbers_in_brackets(full_response)
        if used_nodes:
            list_titles = generate_used_reference_display(
                response.source_nodes, used_nodes
            )
            full_response = str(full_response) + list_titles
            documents = st.session_state["documents"]
            questions = display_questions(generate_sample_questions(documents))
        else:
            questions = ""
        placeholder.markdown(full_response + "\n" + questions)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    
if st.session_state.get("show_chat", False):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about " + research_space_query):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            last_query = st.session_state.messages[-1]["content"]
            chat_engine = st.session_state["chat_engine"]
            response = chat_engine.query(last_query)
            for text in response.response_gen:
                # Appending response content if available
                full_response += text
                # Displaying the response to the user
                message_placeholder.markdown(full_response + "▌")

            used_nodes = extract_numbers_in_brackets(full_response)
            if used_nodes:
                list_titles = generate_used_reference_display(
                    response.source_nodes, used_nodes
                )
                full_response = str(full_response) + list_titles
                documents = st.session_state["documents"]
                questions = display_questions(generate_sample_questions(documents))
            else:
                questions = ""
            message_placeholder.markdown(full_response + "\n" + questions)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
