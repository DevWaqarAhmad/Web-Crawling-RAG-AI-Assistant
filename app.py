import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory  # Required for chat history
from backend import rag_response  # Import the updated rag_response function

# Set page config
st.set_page_config(
    page_title="Property Finder AI Agent",
    layout="wide"
)

# Supported Languages
SUPPORTED_LANGUAGES = {
    "Auto-Detect": None,
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "Chinese": "zh-cn",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Turkish": "tr",
    "Filipino (Tagalog)": "tl"
}

# Sidebar UI
st.sidebar.title("Settings")
selected_lang_name = st.sidebar.selectbox(
    "Select Language",
    options=SUPPORTED_LANGUAGES.keys(),
    index=0  # Default to Auto-Detect
)
selected_lang_code = SUPPORTED_LANGUAGES[selected_lang_name]

if st.sidebar.button("Refresh Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()  # Reset LangChain history

# Contact Info Section
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-family: Arial, sans-serif; padding: 10px;'>
        <h4 style='margin-bottom: 5px;'>📞 Contact Us</h4>
        <p style='margin: 0;'><strong>Phone:</strong> 0900 786 01</p>
        <p style='margin: 0;'><strong>Email:</strong> <a href="mailto:info@demo.ae">info@demo.ae</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Title
st.title("PropertyParams Finder AI Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Use LangChain's ChatMessageHistory

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to UI and memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    target_lang = selected_lang_code

    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            answer = rag_response(
                prompt,
                message_history=st.session_state.chat_history,  # Pass LangChain history
                target_lang=target_lang
            )
        st.markdown(answer)

    # Add assistant response only to visible messages
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Initial greeting
if len(st.session_state.messages) == 0:
    greeting = "Hello! How can I help you today?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.messages.append({"role": "assistant", "content": greeting})