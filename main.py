import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Loading fake reviews dataset as csv and dropping unneeded columns
reviews = pd.read_csv("data/final_labeled_fake_reviews.csv")
reviews = reviews.drop(["images", "asin", "parent_asin", "timestamp", "user_timestamp"], axis=1)
print(reviews.shape)

# Removing rows with missing data
reviews = reviews.dropna()
print(reviews.shape)

# TF-IDF for reviews title column
title_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
)

X_title = title_vectorizer.fit_transform(reviews["title"])

# TF-IDF for review text column
text_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
)

X_text = text_vectorizer.fit_transform(reviews["text"])

# Printing shapes of X_title and X_text matrices
print("Title TF-IDF:", X_title.shape)
print("Text TF-IDF:", X_text.shape)


