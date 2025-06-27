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

def load_knowledge_bases():
    file_list = ["Property_Finder.txt", "Bayut_Property.txt", "Find_Properties.txt"]
    knowledge_entries = []
    for file in file_list:
        try:
            with open(file, "r", encoding="utf-8") as f:
                chunks = f.read().split('\n\n')  
                for chunk in chunks:
                    content = chunk.strip()
                    if content:
                        knowledge_entries.append({
                            "content": content,
                            "source": file.replace(".txt", "")  
                        })
        except FileNotFoundError:
            print(f"❌ File not found: {file}")
    return knowledge_entries

knowledge_base = load_knowledge_bases()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([entry["content"] for entry in knowledge_base])

def retrieve_relevant_chunks(query, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

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

def rag_response(query, message_history=None, target_lang='en'):
    if message_history is None:
        message_history = ChatMessageHistory()

    #Detect input language
    try:
        original_lang = detect(query)
    except:
        original_lang = 'en'  

    #Translate query to English if needed
    if original_lang != 'en':
        try:
            translated_query = translator.translate(query, src=original_lang, dest='en').text
        except:
            translated_query = query
    else:
        translated_query = query

    #Retrieve relevant knowledge chunks
    relevant_chunks = retrieve_relevant_chunks(translated_query)

    if not relevant_chunks:
        fallback_message_en = "I couldn't find relevant info. Please try rephrasing or contact us at 0900 786 01 or info@demo.ae"

        try:
            fallback_message = translate_text(fallback_message_en, src_lang='en', target_lang=original_lang)
        except:
            fallback_message = fallback_message_en

        return fallback_message

    context = "\n\n".join([f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in relevant_chunks])

    history_text = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
        for msg in message_history.messages
    ])

    persona = (
    "You are a helpful AI assistant for UAE real estate. "
    "Use the provided knowledge base to answer property-related questions, and cite the source like this: (Source: Bayut_Property). "
    "However, if the answer comes from general knowledge or chat history, do NOT include a source."
)


    prompt = f"{persona}\n\nChat History:\n{history_text}\n\nKnowledge:\n{context}\n\nUser Question:\n{translated_query}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

    if original_lang != 'en':
        try:
            final_answer = translator.translate(answer_in_english, src='en', dest=original_lang).text
        except:
            final_answer = answer_in_english
    else:
        final_answer = answer_in_english

    #Save both versions to chat history
    message_history.add_message(HumanMessage(content=translated_query))
    message_history.add_message(AIMessage(content=answer_in_english))

    return final_answer