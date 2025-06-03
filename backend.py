import os
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import asyncio

# Configure Gemini API
my_key = "AIzaSyBagwVt7YqZkpQQa_wzoEkVKxzilZTHPY8"
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
user_profile = {}

# Load knowledge base files
def load_knowledge_bases():
    file_list = ["Property_Finder.txt", "Bayut_Property.txt", "Find_Properties.txt"]
    knowledge_bases = {}
    for file in file_list:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                knowledge_bases[file] = content
        except FileNotFoundError:
            knowledge_bases[file] = ""
    return knowledge_bases

def retrieve_per_file_responses(query, knowledge_bases, top_k=2, similarity_threshold=0.1):
    responses = []
    for file, content in knowledge_bases.items():
        if not content.strip():
            continue

        paragraphs = content.split('\n\n')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        top_scores = similarities[top_indices]

        if top_scores[0] < similarity_threshold:
            continue

        best_paragraphs = [
            paragraphs[i].replace('\n', ' ').strip()
            for i in top_indices if similarities[i] >= similarity_threshold
        ]
        if not best_paragraphs:
            continue

        source_name = file.replace('.txt', '')
        response_with_source = f"[Source: {source_name}]\n" + "\n".join(best_paragraphs)
        responses.append(response_with_source)

    return responses

def is_general_query(text):
    general_patterns = [
        r"how are you", r"hi", r"hello", r"hey", r"good morning", r"good evening",
        r"what is your name", r"who are you", r"thank you", r"thanks",
        r"my name is", r"what is my name"
    ]
    text = text.lower()
    return any(re.search(pattern, text) for pattern in general_patterns)

def rag_response(query, message_history=None, target_lang='en'):
    if message_history is None:
        message_history = ChatMessageHistory()

    try:
        original_lang = detect(query)
        if len(query.split()) <= 3:
            original_lang = 'en'
    except:
        original_lang = 'en'

    translated_query = query
    if original_lang != 'en':
        try:
            translated_query = translator.translate(query, src=original_lang, dest='en').text
        except:
            translated_query = query

    chat_context = ""
    for msg in message_history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        chat_context += f"{role}: {msg.content}\n"

    # Store name if user says "my name is..."
    if "my name is" in translated_query.lower():
        parts = translated_query.lower().split("my name is")
        if len(parts) > 1:
            name = parts[1].strip().split()[0].capitalize()
            user_profile["name"] = name

    # Answer if user asks "what is my name"
    if "what is my name" in translated_query.lower() and "name" in user_profile:
        return f"Your name is {user_profile['name']}."

    # Handle greetings and general small talk
    if is_general_query(translated_query):
        persona = "You are a helpful assistant answering general questions briefly and kindly."
        prompt = f"{persona}\nConversation History:\n{chat_context}\nUser Question:\n{translated_query}\nAnswer:"
        try:
            response = model.generate_content(prompt)
            answer_in_english = response.text
        except Exception as e:
            answer_in_english = f"An error occurred: {str(e)}"

        final_answer = translator.translate(answer_in_english, src='en', dest=original_lang).text if original_lang != 'en' else answer_in_english
        message_history.add_message(HumanMessage(content=query))
        message_history.add_message(AIMessage(content=answer_in_english))
        return final_answer

    # Load knowledge base
    knowledge_bases = load_knowledge_bases()
    file_responses = retrieve_per_file_responses(translated_query, knowledge_bases)

    if file_responses:
        combined_response = "\n\n".join(file_responses)
        message_history.add_message(HumanMessage(content=query))
        message_history.add_message(AIMessage(content=combined_response))
        return combined_response

    # Fallback to model
    persona = "You are a helpful assistant answering real estate-related queries clearly."
    prompt = f"{persona}\nConversation History:\n{chat_context}\nUser Question:\n{translated_query}\nAnswer:"
    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        answer_in_english = f"An error occurred: {str(e)}"

    final_answer = translator.translate(answer_in_english, src='en', dest=original_lang).text if original_lang != 'en' else answer_in_english
    message_history.add_message(HumanMessage(content=query))
    message_history.add_message(AIMessage(content=answer_in_english))
    return final_answer

def main():
    print("\U0001F916 Property Finder AI Agent Chatbot (type 'exit' to quit)\n")
    chat_history = ChatMessageHistory()

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot...")
            break

        response = rag_response(user_input, message_history=chat_history)
        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
