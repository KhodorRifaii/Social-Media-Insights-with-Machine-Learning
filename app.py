import streamlit as st
import joblib


model = joblib.load("naive_bayes_count.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

st.set_page_config(page_title="Tweet Virality Predictor")

st.title("Tweet Virality Predictor 🚀")
st.write("Enter a tweet and the model (CountVectorizer + Naive Bayes) will predict if it goes viral:")

tweet = st.text_area("Tweet text here")

if st.button("Predict"):
    if tweet.strip():
        vec = vectorizer.transform([tweet])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()

        if pred == 1:
            st.success(f"🔥 Likely to go VIRAL (Confidence: {prob:.2f})")
        else:
            st.info(f"❄️ Not likely to go viral (Confidence: {prob:.2f})")
    else:
        st.error("Please enter a tweet first.")
