import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_Vectorizer.pkl")

def remove_punctuation(text):
    translator = str.maketrans('','', string.punctuation)
    return text.translate(translator)

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered_tokens)

st.title("Spam Message Detector")

with st.expander("📊 Model Information"):
     "Model: Logistic Regression"
     "Vectorizer: TF-IDF"
     "Accuracy", "94.9%"
     "Precision", "96%"
     "Recall", "65%"
     "F1 Score", "78%"

st.write("Enter Your Message Below")
message = st.text_area("Enter Message")
if st.button("Predict"):
    message = message.lower()
    message = remove_punctuation(message)
    message = remove_stopwords(message)
    message = message.strip()
    vector = tfidf.transform([message])
    prediction = model.predict(vector)
    if prediction[0] == 1:
        st.error('🚨 Spam Detected')
    else:
        st.success('✅ Regular Message')


