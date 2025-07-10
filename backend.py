
import os
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Configure Gemini API
my_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=my_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)

# Translator instance
translator = Translator()

# Load knowledge base files into dict
def load_knowledge_bases():
    file_list = ["Property_Finder.txt", "Bayut_Property.txt", "Find_Properties.txt"]
    knowledge_bases = {}
    for file in file_list:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                knowledge_bases[file] = content
        except FileNotFoundError:
            knowledge_bases[file] = ""  # Empty if file not found
    return knowledge_bases

# Retrieve relevant paragraphs per file with source info
def retrieve_per_file_responses(query, knowledge_bases, top_k=2, similarity_threshold=0.1):
    responses = []
    for file, content in knowledge_bases.items():
        if not content.strip():
            responses.append(f"[Source: {file.replace('.txt','')}] I am working on it.")
            continue

        paragraphs = content.split('\n\n')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        top_scores = similarities[top_indices]

        # Check if top similarity is below threshold
        if top_scores[0] < similarity_threshold:
            responses.append(f"[Source: {file.replace('.txt','')}] I am working on it.")
            continue

        best_paragraphs = [paragraphs[i].strip() for i in top_indices if similarities[i] >= similarity_threshold]
        joined_response = "\n".join(best_paragraphs) if best_paragraphs else "I am working on it."
        responses.append(f"[Source: {file.replace('.txt','')}]\n{joined_response}")
    return responses

# Simple check if user query is general/greeting type
def is_general_query(text):
    general_patterns = [
        r"how are you", r"hi", r"hello", r"hey", r"good morning", r"good evening",
        r"what is your name", r"who are you", r"thank you", r"thanks"
    ]
    text = text.lower()
    return any(re.search(pattern, text) for pattern in general_patterns)

# Main RAG response function with multi-file handling and language translation
def rag_response(query, message_history=None, target_lang='en'):
    if message_history is None:
        message_history = ChatMessageHistory()

    # Detect language
    try:
        original_lang = detect(query)
    except:
        original_lang = 'en'

    # Translate query to English
    if original_lang != 'en':
        try:
            translated_query = translator.translate(query, src=original_lang, dest='en').text
        except:
            translated_query = query
    else:
        translated_query = query

    # Build context from message_history
    chat_context = ""
    for msg in message_history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        chat_context += f"{role}: {msg.content}\n"

    # If general greeting
    if is_general_query(translated_query):
        persona = "You are a helpful assistant answering general questions briefly and kindly."
        prompt = f"{persona}\nConversation History:\n{chat_context}\nUser Question:\n{translated_query}\nAnswer:"
        try:
            response = model.generate_content(prompt)
            answer_in_english = response.text
        except Exception as e:
            answer_in_english = f"An error occurred: {str(e)}"

        if original_lang != 'en':
            try:
                final_answer = translator.translate(answer_in_english, src='en', dest=original_lang).text
            except:
                final_answer = answer_in_english
        else:
            final_answer = answer_in_english

        message_history.add_message(HumanMessage(content=query))
        message_history.add_message(AIMessage(content=answer_in_english))
        return final_answer

    # For property queries: get relevant info + include context
    knowledge_bases = load_knowledge_bases()
    file_responses = retrieve_per_file_responses(translated_query, knowledge_bases)

    combined_response = "\n\n".join(file_responses)

    # Include chat history + knowledge base response in prompt
    prompt = f"Conversation History:\n{chat_context}\nUser Question:\n{translated_query}\nRelevant Info:\n{combined_response}\nAnswer:"

    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        answer_in_english = f"An error occurred: {str(e)}"

    if original_lang != 'en':
        try:
            final_answer = translator.translate(answer_in_english, src='en', dest=original_lang).text
        except:
            final_answer = answer_in_english
    else:
        final_answer = answer_in_english

    message_history.add_message(HumanMessage(content=query))
    message_history.add_message(AIMessage(content=answer_in_english))

    return final_answer
# def main():
#     print("🤖 Property Finder AI Agent Chatbot (type 'exit' to quit)\n")
#     chat_history = ChatMessageHistory()

#     while True:
#         user_input = input("🧑 You: ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Exiting chatbot...")
#             break

#         response = rag_response(user_input, message_history=chat_history)
#         print(f"🤖 Bot: {response}\n")

# if __name__ == "__main__":
#     main()
