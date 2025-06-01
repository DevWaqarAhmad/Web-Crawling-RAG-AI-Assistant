import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

my_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=my_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
translator = Translator()

# 1. Load multiple files and tag them with source
def load_knowledge_bases():
    file_list = ["Property_Finder.txt", "Bayut_Property.txt", "Find_Properties.txt"]
    knowledge_entries = []
    for file in file_list:
        try:
            with open(file, "r", encoding="utf-8") as f:
                chunks = f.read().split('\n\n')  # Split by paragraphs
                for chunk in chunks:
                    content = chunk.strip()
                    if content:
                        knowledge_entries.append({
                            "content": content,
                            "source": file.replace(".txt", "")  # e.g., Property_Finder
                        })
        except FileNotFoundError:
            print(f"❌ File not found: {file}")
    return knowledge_entries

knowledge_base = load_knowledge_bases()

# 2. Create embeddings
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([entry["content"] for entry in knowledge_base])

# 3. Retrieval with source info
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

# 4. Detect and translate language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate_text(text, src_lang='auto', target_lang='en'):
    try:
        return translator.translate(text, src=src_lang, dest=target_lang).text
    except:
        return text

# 5. RAG + Gemini Logic using LangChain Chat History
def rag_response(query, message_history=None, target_lang='en'):
    if message_history is None:
        message_history = ChatMessageHistory()

    original_lang = detect_language(query)
    translated_query = translate_text(query, src_lang=original_lang, target_lang='en') if original_lang != 'en' else query

    # Retrieve relevant chunks with source
    relevant_chunks = retrieve_relevant_chunks(translated_query, top_k=3)

    if not relevant_chunks:
        fallback = "I couldn't find relevant info. Please try rephrasing or contact us at 0900 786 01 or info@demo.ae"
        return translate_text(fallback, src_lang='en', target_lang=original_lang) if original_lang != 'en' else fallback

    # Build prompt with source tagging
    context = "\n\n".join([f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in relevant_chunks])

    # Format history
    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
        for msg in message_history.messages
    ])

    persona = (
        "You are a helpful AI assistant for UAE real estate. "
        "Provide concise and accurate answers based on the following knowledge. "
        "Clearly mention which source each piece of info is from."
    )

    prompt = f"{persona}\n\nChat History:\n{history_text}\n\nKnowledge:\n{context}\n\nUser Question:\n{translated_query}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

    # Translate response back
    answer = translate_text(answer_in_english, src_lang='en', target_lang=original_lang) if original_lang != 'en' else answer_in_english

    # Save to message history
    message_history.add_message(HumanMessage(content=translated_query))
    message_history.add_message(AIMessage(content=answer_in_english))

    return answer