import streamlit as st
from backend import (
    create_index,
    get_chat_engine,
    citation_query_engine,
    generate_sample_questions,
)
import datetime
import pytz
from utils import (
    get_twitter_badge,
    get_link_tree_badge,
    get_github_badge,
    display_questions,
    extract_numbers_in_brackets,
    generate_used_reference_display,
    documents_to_df,
)
import openai

# ?query=deep%20learning%20for%20nlp&num_papers=50&full_text=True
# TODO update URL with query and num_papers after button press: add share button
# TODO add github logo in same style as twitter and linktree
# TODO [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-custom-subdomain>.streamlit.app)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


# if not firebase_admin._apps:
#     cred = dict(st.secrets["firebase"]['my_project_settings'])
#     # cred = credentials.Certificate("/Users/shaurya/Desktop/s2qa-0512-firebase-adminsdk-fc3mi-066e0bd473.json")
#     default_app = firebase_admin.initialize_app(cred, {'databaseURL' : 'https://s2qa-0512-default-rtdb.firebaseio.com/'})
# else:
#     default_app = firebase_admin.get_app()

# # Get a database reference
# ref = db.reference("logs", app=default_app)

url_params = st.experimental_get_query_params()
num_papers = 30
query = "large language models"
full_text = False
if "query" in url_params:
    query = url_params["query"][0]
if "num_papers" in url_params:
    num_papers = int(url_params["num_papers"][0])
if "full_text" in url_params:
    full_text = bool(url_params["full_text"][0])


with st.sidebar:
    st.title("📚🤖 S2QA: Query your Research Space")

    st.markdown(
        f" {get_github_badge()} {get_twitter_badge()}  {get_link_tree_badge()}",
        unsafe_allow_html=True,
    )
    st.markdown("Ask deeper questions about your research space")
    openai_api_key = st.text_input("OpenAI API Key", "OPENAI_API_KEY", type="password")
    openai.api_key = openai_api_key
    "🔑 [Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    research_space_box = st.empty()
    # hidden text input placeholder
    research_space_query = st.text_input(
        "Enter your research space, \n e.g. machine learning, large language models, covid 19 vaccine",
        query,
    )
    num_papers = st.slider(
        "Number of papers you want to chat with ", 5, 30, num_papers
    )
    full_text = st.toggle(
        "Full Text (initial setup is slow as we first download the pdfs: default set to 10 papers)"
    )
    if full_text:
        full_text = True
        num_papers = 5
    button = st.button("Set Research Space", type="primary")
    num_papers = int(num_papers/2)

# Welcome to S2QA
st.markdown(
    """
    # 🚀 Welcome to S2QA 🚀
    ## 🧩 Ask deeper questions about your research

    ### How to use this tool? 🧐
    
    📌 **Step One:** Enter your research space (For example: machine learning, large language models, covid 19 vaccine)
    
    📌 **Step Two:** Select the number of papers you want to get the answers from
    
    📌 **Step Three:** Click on the button "Set Research Space" 🖱️. Please be patient, this could take a few minutes to set up the index
    
    📌 **Step Four:** Congrats🎉! Now you have set your research space, you can delve into and ask questions about it
    
    📌 **Step Five:** Once a research space is established, you can share the URL with your colleagues 👥 to collaborate on the research space
    
    Happy Exploring! 🕵️‍♂️
    """
)


if button and research_space_query:
    st.session_state["show_chat"] = True
    with st.sidebar:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        with st.status("🦙🦙 LlaMa's are working together . . ."):
            st.write("Fetching papers for research space: " + research_space_query)
            try:
                index, documents = create_index(
                    research_space_query.lower(), num_papers, full_text
                )
                st.experimental_set_query_params(
                    query=[research_space_query],
                    num_papers=[num_papers],
                    full_text=[full_text],
                )
            except Exception as e:
                st.error("Error creating index: " + str(e))
                st.error(
                    "Please check your API key or try reducing the number of papers."
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
        if "messages" not in st.session_state:
            st.session_state.messages = []
        response = chat_engine.query("elaborate on " + research_space_query)
        full_response = ""
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
            # questions = ""
        #     ref.push(
        #     {
        #         "full_response": full_response,
        #         "query": research_space_query,
        #         "research_space": research_space_query,
        #         "num_papers": num_papers,
        #         "full_text": full_text,
        #         "timestamp": datetime.datetime.now(pytz.utc).isoformat(),            }
        # )

        else:
            questions = ""
        placeholder.markdown(full_response + "\n" + questions)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # st.session_state.messages = []


if st.session_state.get("show_chat", False):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages[1:]:
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
                # questions = ""
            else:
                questions = ""
            message_placeholder.markdown(full_response + "\n" + questions)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # Push the response and query to the database
        # ref.push(
        #     {
        #         "full_response": full_response,
        #         "query": last_query,
        #         "research_space": research_space_query,
        #         "num_papers": num_papers,
        #         "full_text": full_text,
        #         "timestamp": datetime.datetime.now(pytz.utc).isoformat(),            }
        # )

