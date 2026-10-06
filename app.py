import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# Caching ML model so it doesn't reload every time
# a command is run in streamlit
@st.cache_resource
def load_models():
    """Loads NLP and ML models from disk."""
    return joblib.load("ML_NLP/ml_model/models.joblib")

def ranked_terms(vectorizer, matrix, coefficients, source, top_n=5):
    """Returns DataFrame of the most important terms ranked by score."""
    names = vectorizer.get_feature_names_out()
    row = matrix.toarray()[0]
    terms = []

    # Calculates term influences on ML model prediction
    # and stores term, its contribution, its source, and
    # predicted direction
    for index, value in enumerate(row):
        if value == 0:
            continue

        # Calculates how term influenced ML model's prediction
        contribution = float(value * coefficients[index])
        terms.append({
            "source": source,
            "term": names[index],
            "contribution": contribution,
            "direction": "Fake" if contribution > 0 else "Real",
        })

    # Sorting terms to where the terms with largest contributions to prediction
    # appear first in DataFrame (absolute value removes influence of negative sign
    # to focus only on magnitude)
    terms.sort(key=lambda term: abs(term["contribution"]), reverse=True)
    return terms[:top_n]

# Gathering user inputted information on Amazon review
st.title("Review Scout")
st.subheader("NLP + ML Powered Fake Amazon Review Detector")

st.write("What did the reviewer rate the product")
star_rating = st.feedback(options="stars")

helpful_votes = st.number_input(label="How many people found this review helpful?", step=1, min_value=0)

title = st.text_input("Enter the title of the Amazon review:")
text = st.text_area("Enter the text of the Amazon review:")

entered = st.button("Detect Fake Review")

# Using the ML model to predict whether the review is real
# or fake once the "Detect Fake Review" button is pressed
if entered == True and (star_rating is None or title.strip() == "" or text.strip() == ""):
    st.warning("Please fill in all fields before detecting a fake review.")

elif entered ==True:
    models = load_models()

    # Loading TF-IDF vectorizers and Logistic Regression model
    title_vectorizer = models["title_vectorizer"]
    text_vectorizer = models["text_vectorizer"]
    loaded_model = models["ml_model"]
    text_char_vectorizer = models["text_char_vectorizer"]

    # Converting user input in Streamlit UI into DataFrame the ML
    # model can interpret
    new_review_data = {"rating": [star_rating + 1],
            "title": [title],
            "text": [text],
            "helpful_votes": [helpful_votes],
        }

    new_review = pd.DataFrame(new_review_data)

    # Applying fitted TF-IDF Vectorizers to new review's title and text
    title = title_vectorizer.transform(new_review["title"])
    text = text_vectorizer.transform(new_review["text"])
    text_char = text_char_vectorizer.transform(new_review["text"])

    # Converting numeric and numerically encoded features of new review to CSR
    # matrices so they can be horizontally stacked later with other features
    numeric_features = csr_matrix(
        new_review[["rating", "helpful_votes"]].values
    )

    # Horizontally stacking all new review features into numpy array
    # for model label prediction
    new_review = hstack([
        title,
        text,
        text_char,
        numeric_features,
    ])

    # Predicting whether review is fake or real and then
    # displaying the result and model confidence
    prediction = loaded_model.predict(new_review)
    prediction_confidences = loaded_model.predict_proba(new_review)

    predicted_class = int(prediction[0])

    prediction_confidence = round(float(prediction_confidences[0][predicted_class]) * 100, 1)

    if prediction[0] == 0:
        st.subheader("✅ This review is likely real.")
        st.subheader(f"Prediction Confidence: {prediction_confidence}%")
    if prediction[0] == 1:
        st.subheader("❌ This review is likely fake.")
        st.subheader(f"Prediction Confidence: {prediction_confidence}%")

    # Finding the most important words and characters that contributed to ML model's prediction
    coefficients = loaded_model.coef_[0]
    n_title = title.shape[1]
    n_text = text.shape[1]
    title_terms = ranked_terms(
        title_vectorizer, title, coefficients[:n_title], "title",
    )

    text_terms = ranked_terms(
        text_vectorizer, text, coefficients[n_title:n_title + n_text], "text",
    )

    n_text_char = text_char.shape[1]
    char_start = n_title + n_text
    char_terms = ranked_terms(
        text_char_vectorizer,
        text_char,
        coefficients[char_start:char_start + n_text_char],
        "text characters",
    )

    important_words = sorted(
        title_terms + text_terms + char_terms,
        key=lambda term: abs(term["contribution"]),
        reverse=True,
    )[:5]

    # Displaying terms with the greatest contribution to ML model's prediction
    st.subheader("Top 5 terms that most affected this prediction")
    st.dataframe(pd.DataFrame(important_words[:5]))
