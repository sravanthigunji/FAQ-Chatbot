import streamlit as st
import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

faq_data = {
    "What payment methods are accepted?":
    "We accept UPI, credit cards, debit cards and net banking.",

    "How can I track my order?":
    "You can track your order in the My Orders section.",

    "What is the return policy?":
    "Products can be returned within 7 days of delivery.",

    "How long does delivery take?":
    "Delivery usually takes 3 to 5 business days.",

    "Do you offer cash on delivery?":
    "Yes, cash on delivery is available for selected locations."
}


def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(text)

    return " ".join(words)


questions = list(faq_data.keys())

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_questions)


def chatbot(user_input):

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, X)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    if best_score > 0.3:

        best_question = questions[best_match_index]

        return faq_data[best_question]

    else:
        return "Sorry, I could not understand your question."


st.title("Shopping FAQ Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:

    response = chatbot(user_input)

    st.write("Bot:", response)