import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory  
from backend import rag_response  

st.set_page_config(
    page_title="Property Finder AI Agent",
    layout="wide"
)

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

st.sidebar.title("Settings")
selected_lang_name = st.sidebar.selectbox(
    "Select Language",
    options=SUPPORTED_LANGUAGES.keys(),
    index=0  
)
selected_lang_code = SUPPORTED_LANGUAGES[selected_lang_name]

if st.sidebar.button("Refresh Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()  # Reset chat history

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-family: Arial, sans-serif; padding: 10px;'>
        <h4 style='margin-bottom: 5px;'>📞 Contact Us</h4>
        <p style='margin: 0;'><strong>Phone:</strong> +971 000 000 000</p>
        <p style='margin: 0;'><strong>Email:</strong> <a href="mailto:info@demo.ae">info@demo.ae</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Property Finder AI Agent")

# Initialize messages and chat_history if not exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use selected language for translation etc.
    target_lang = selected_lang_code

    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            # Call your backend rag_response function with chat history & lang
            answer = rag_response(
                prompt,
                message_history=st.session_state.chat_history,
                target_lang=target_lang
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Initial greeting if no messages yet
if len(st.session_state.messages) == 0:
    greeting = "Hello! How can I help you today?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.messages.append({"role": "assistant", "content": greeting})