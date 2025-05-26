import streamlit as st
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
    st.session_state.chat_history = []

st.sidebar.markdown("---")

# 📞 Contact info section
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
st.title("Property Finder AI Agent")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask your question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    target_lang = selected_lang_code

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            answer = rag_response(prompt, chat_history=st.session_state.chat_history, target_lang=target_lang)
        st.markdown(answer)

    # Add both prompt and response to memory and visible messages
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(f"User: {prompt}")
    st.session_state.chat_history.append(f"Bot: {answer}")

# Initial greeting
if len(st.session_state.messages) == 0:
    greeting = "Hello! How can I help you today?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.messages.append({"role": "assistant", "content": greeting})
