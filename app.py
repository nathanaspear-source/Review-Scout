import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


# Caching ML model so it doesn't reload every time
# a command is run in streamlit
@st.cache_resource
def load_models():
    return joblib.load("ML_NLP/ml_model/models.joblib")

# Gathering user inputted information on Amazon review
st.title("Review Scout")
st.subheader("NLP + ML Powered Fake Amazon Review Detector")

st.write("What did the reviewer rate the product")
star_rating = st.feedback(options="stars")

verified_purchaser = st.checkbox(label="Was the review written by a verified purchaser?")

helpful_votes = st.number_input(label="How many people found this review helpful?", step=1)

title = st.text_input("Enter the title of the Amazon review:")
text = st.text_area("Enter the text of the Amazon review:")

entered = st.button("Detect Fake Review")

if entered == True:
    models = load_models()

    title_vectorizer = models["title_vectorizer"]
    text_vectorizer = models["text_vectorizer"]
    loaded_model = models["ml_model"]

    new_review_data = {"rating": [star_rating + 1],
            "title": [title],
            "text": [text],
            "helpful_votes": [helpful_votes],
            "verified_purchase": [verified_purchaser]}

    new_review = pd.DataFrame(new_review_data)

    title = title_vectorizer.transform(new_review["title"])
    text = text_vectorizer.transform(new_review["text"])

    numeric_features = csr_matrix(
        new_review[["rating", "helpful_votes"]].values
    )

    verified_purchaser = csr_matrix(
        new_review[["verified_purchase"]].astype(int).values.reshape(-1, 1)
    )

    new_review = hstack([
        title,
        text,
        numeric_features,
        verified_purchaser
    ])

    prediction = loaded_model.predict(new_review)

    if prediction[0] == 0:
        st.subheader("✅ This review is likely real.")
    if prediction[0] == 1:
        st.subheader("❌ This review is likely fake.")

