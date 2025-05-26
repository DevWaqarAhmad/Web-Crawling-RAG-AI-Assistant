import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
my_key= "AIzaSyBagwVt7YqZkpQQa_wzoEkVKxzilZTHPY8"
#load_dotenv()
#my_key = os.getenv("GEMINI_API_KEY") 
#my_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=my_key)
model = genai.GenerativeModel("gemini-1.5-flash")

file_path = 'propertyfinder.txt'

genai.configure(api_key=my_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data.split('\n\n')
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

knowledge_base = load_knowledge_base(file_path)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(knowledge_base)

def retrieve_relevant_chunks(query, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

# Language Helper Functions
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  

def translate_text(text, src_lang='auto', target_lang='en'):
    try:
        result = translator.translate(text, src=src_lang, dest=target_lang)
        return result.text
    except:
        return text  
chat_history = []

def rag_response(query, chat_history=[], target_lang='en'):
    # Step 1: Detect input language
    try:
        original_lang = detect(query)
    except:
        original_lang = 'en'

    # Step 2: Translate to English if needed
    if original_lang != 'en':
        try:
            translated_query = translator.translate(query, src=original_lang, dest='en').text
        except:
            translated_query = query
    else:
        translated_query = query

    # Step 3: Retrieve relevant info from knowledge base
    relevant_chunks = retrieve_relevant_chunks(translated_query, top_k=3)

    if not relevant_chunks:
        fallback = "I couldn't find relevant info. Please try rephrasing or contact us at 0900 786 01 or info@demo.ae"
        if original_lang != 'en':
            return translator.translate(fallback, src='en', dest=original_lang).text
        return fallback

    # Step 4: Build context and prompt
    context = "\n".join(relevant_chunks)
    history_text = "\n".join(chat_history)
    persona = (
        "You are a helpful AI assistant for Property Finder specializing in real estate. "
        "Your goal is to provide accurate, concise, and friendly responses to user queries. "
        "If you don't know the answer, politely inform the user."
    )

    full_context = f"{persona}\n\n{history_text}\n\n{context}"
    prompt = f"Context:\n{full_context}\n\nQuestion:\n{translated_query}\n\nAnswer:"

    # Step 5: Get Gemini response
    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

    # Step 6: Translate back to original language
    if original_lang != 'en':
        try:
            answer = translator.translate(answer_in_english, src='en', dest=original_lang).text
        except:
            answer = answer_in_english
    else:
        answer = answer_in_english

    # Step 7: Update chat history
    chat_history.append(f"User: {translated_query}")
    chat_history.append(f"Bot: {answer_in_english}")

    return answer

if __name__ == "__main__":
    print("Welcome to Property Finder AI Assistant! (Type 'exit' to quit)\n")

    chat_history = []

    while True:
        user_query = input("Ask your about UAE real estate: ")

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Thanks for using the AI assistant.")
            break

        answer = rag_response(user_query, chat_history)
        print("\nBOT Response:", answer)
        print("\n" + "-"*60 + "\n")
