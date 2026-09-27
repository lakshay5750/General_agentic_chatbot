from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st


st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 Agentic Chatbot with LangGraph")
st.caption("Groq + LangGraph + MemorySaver")


# -----------------------------
# Thread ID
# -----------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_1"


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# -----------------------------
# Display previous messages
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Type your message...")


if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # -----------------------------
    # Generate streaming response
    # -----------------------------
    with st.chat_message("assistant"):

        def generate_response():

            for chunk in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages"
            ):

                message_chunk, metadata = chunk

                if message_chunk.content:
                    yield message_chunk.content


        response = st.write_stream(generate_response())


    # -----------------------------
    # Save AI response
    # -----------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })