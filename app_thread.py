from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid


# =========================================================
# Thread / Chat Functions
# =========================================================

# Generate a unique thread ID
def generate_thread_id():
    return str(uuid.uuid4())


# Add a thread to the conversation list
def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a new chat
def reset_chat():

    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id

    st.session_state["message_history"] = []

    # Give the new chat a default title
    st.session_state["chat_titles"][new_thread_id] = "New Chat"

    add_thread(new_thread_id)


# Load conversation from LangGraph checkpointer
def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# =========================================================
# Generate Chat Title
# =========================================================

def generate_chat_title(user_message):

    # Take first 30 characters of user's first message
    title = user_message.strip()

    if len(title) > 30:
        title = title[:30] + "..."

    return title


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Agentic Chatbot with LangGraph")


# =========================================================
# Session State Initialization
# =========================================================

# Store messages of currently selected conversation
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Store current thread ID
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Store all thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


# Store human-readable names
#
# Example:
#
# {
#     "uuid-123": "Explain RAG",
#     "uuid-456": "Python Interview",
# }
#
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}


# Add current thread
add_thread(st.session_state["thread_id"])


# If current thread doesn't have a title
if st.session_state["thread_id"] not in st.session_state["chat_titles"]:

    st.session_state["chat_titles"][
        st.session_state["thread_id"]
    ] = "New Chat"


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("💬 My Conversations")


# ---------------------------------------------------------
# New Chat Button
# ---------------------------------------------------------

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


# ---------------------------------------------------------
# Display Conversations
# ---------------------------------------------------------

st.sidebar.markdown("---")


for thread_id in st.session_state["chat_threads"][::-1]:

    # Get human-readable title
    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat"
    )

    # Highlight current conversation
    if thread_id == st.session_state["thread_id"]:

        button_label = f"🟢 {title}"

    else:

        button_label = f"💬 {title}"


    # Create sidebar button
    if st.sidebar.button(
        button_label,
        key=f"chat_{thread_id}",
        use_container_width=True
    ):

        # Change current thread
        st.session_state["thread_id"] = thread_id


        # Load conversation from LangGraph
        messages = load_conversation(thread_id)


        # Temporary UI message list
        temp_messages = []


        # Convert LangChain messages
        # into Streamlit format
        for message in messages:

            if isinstance(message, HumanMessage):

                role = "user"

            elif isinstance(message, AIMessage):

                role = "assistant"

            else:

                # Ignore ToolMessage etc.
                continue


            temp_messages.append(
                {
                    "role": role,
                    "content": message.content
                }
            )


        # Replace current UI history
        st.session_state["message_history"] = temp_messages


        # Refresh UI
        st.rerun()


# =========================================================
# Main Chat Interface
# =========================================================


# Display previous messages
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "Type your message..."
)


# =========================================================
# Process User Message
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Get Current Thread
    # -----------------------------------------------------

    current_thread_id = st.session_state["thread_id"]


    # -----------------------------------------------------
    # Generate Chat Title
    # -----------------------------------------------------

    # Only generate title for the first user message
    if (
        st.session_state["chat_titles"][current_thread_id]
        == "New Chat"
    ):

        title = generate_chat_title(user_input)

        st.session_state["chat_titles"][
            current_thread_id
        ] = title


    # -----------------------------------------------------
    # Save User Message in UI
    # -----------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # -----------------------------------------------------
    # Display User Message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(user_input)


    # -----------------------------------------------------
    # LangGraph Configuration
    # -----------------------------------------------------

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }


    # -----------------------------------------------------
    # Stream AI Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata
            in chatbot.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },

                config=CONFIG,

                stream_mode="messages"
            )

            if isinstance(
                message_chunk,
                AIMessage
            )
        )


    # -----------------------------------------------------
    # Save AI Response
    # -----------------------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )