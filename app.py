import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words("english") and word not in string.punctuation]
    return " ".join(words)

st.set_page_config(page_title="AI Spam Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>🚀 SMS Spam Classifier</h1>", unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message:", height=150)

if st.button("🚀 Predict", use_container_width=True):
    if input_sms.strip():
        with st.spinner("Analyzing message..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            st.success("✅ Analysis Complete!")
            st.markdown(f"<h2 style='text-align: center; color: {'red' if result == 1 else 'green'};'>📢 {'Spam' if result == 1 else '✅ Not Spam'}</h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message before predicting.")
