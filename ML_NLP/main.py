"""Preprocesses fake reviews dataset, applies TF-IDF vectorization to text-based features,
and trains a logistic regression model to predict whether a review is real or fake.

This module contains the entire code for preprocessing, NLP, and ML model training.
Unneeded features are dropped from the dataset, the dataset is split into train and test
sets using stratified sampling, the text-based feature columns are TF-IDF vectorized, and
a logistic regression model is trained. After the ML model is trained, it is saved with the
three TF-IDF vectorizers used for the title and text feature columns using joblib. Also,
model evaluation metrics are printed, including precision, recall, f1-score, and a confusion
matrix. Print statements are used as the module is run to display relevant information
about the model pipeline.

Attributes:
    DATA_PATH (Path): Path to the fake reviews dataset.
    MODEL_PATH (Path): Path to save the trained ML model and TF-IDF vectorizers.

Available Functions:
    get_top_words(words, coefs, n=5, direction=1): Returns the top n words or phrases
    in the dataset in the specified class direction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

DATA_PATH = Path(__file__).resolve().parent/"data"/"final_labeled_fake_reviews.csv"
MODEL_PATH = Path(__file__).resolve().parent/"ml_model"/"models.joblib"

def get_top_words(words, coefs, n=5, direction=1):
    """Returns the top n TF-IDF terms in the dataset that contributed
    to the specified class direction."""
    results = pd.DataFrame({
        "word_or_phrase": words,
        "coefficient": coefs,
    })

    results["direction"] = np.where(
        results["coefficient"] > 0,
        "Fake (class 1)",
        "Real (class 0)"
    )

    if direction == 1:
        return results.sort_values(
            "coefficient", ascending=False
        ).head(n)

    elif direction == 0:
        return results.sort_values(
            "coefficient", ascending=True
        ).head(n)
    else:
        print("Invalid direction")

# Loading fake reviews dataset as csv and dropping unneeded columns
reviews = pd.read_csv(DATA_PATH)
reviews = reviews.drop(
    [
    "images",
    "verified_purchase",
    "asin", "parent_asin",
    "timestamp",
    "user_timestamp"],
    axis=1
)

# Removing rows with missing data
reviews = reviews.dropna()

# Sorting reviews DataFrame into features and label
X = reviews[["rating", "title", "text", "helpful_vote"]]
y = reviews["label"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# TF-IDF for reviews title column
title_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
)

text_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
)

# Applying TF-IDF to title and text corpus
X_train_title = title_vectorizer.fit_transform(X_train["title"])
X_train_text = text_vectorizer.fit_transform(X_train["text"])

X_test_title = title_vectorizer.transform(X_test["title"])
X_test_text = text_vectorizer.transform(X_test["text"])

# Converting numeric features to csr matrix for hstack with corpus features
X_train_numeric = csr_matrix(
    X_train[["rating", "helpful_vote"]].values
)

X_test_numeric = csr_matrix(
    X_test[["rating", "helpful_vote"]].values
)

text_char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
)

X_train_text_char = text_char_vectorizer.fit_transform(X_train["text"])
X_test_text_char = text_char_vectorizer.transform(X_test["text"])

# Horizontally stacking train and test text-based data with other feature columns
X_train = hstack([
    X_train_title,
    X_train_text,
    X_train_text_char,
    X_train_numeric,
])

X_test = hstack([
    X_test_title,
    X_test_text,
    X_test_text_char,
    X_test_numeric,
])

# Printing shapes of stacked training and test matrices
print("X_train TF-IDF Shape:", X_train.shape)
print("X_test TF-IDF Shape:", X_test.shape, "\n")
print("=" * 60, "\n")

# Training Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Saving ML model to disk
joblib.dump(
    {
        "ml_model": model,
        "title_vectorizer": title_vectorizer,
        "text_vectorizer": text_vectorizer,
        "text_char_vectorizer": text_char_vectorizer,
    },
    MODEL_PATH
)
print("\nModel saved successfully\n")
print("=" * 60, "\n")

# Printing model evaluation scores
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\n", "=" * 60, "\n")

# Getting coefficients learned by Logistic Regression Model
coefficients = model.coef_[0]

# Getting number of features in each TF-IDF matrix
n_title = X_train_title.shape[1]
n_text = X_train_text.shape[1]

# Getting TF-IDF feature names
title_words = title_vectorizer.get_feature_names_out()
text_words = text_vectorizer.get_feature_names_out()
char_features = text_char_vectorizer.get_feature_names_out()

# Coefficients for each feature group
title_coefficients = coefficients[:n_title]
text_coefficients = coefficients[n_title:n_title + n_text]

# Checking to see if length of word and coefficient arrays
# are the same length for title and text features
print("Check to see if length of word and coefficient arrays are equal")
print("\n Title:", len(title_words), len(title_coefficients))
print("Text:", len(text_words), len(text_coefficients), "\n")

print("Top title terms associated with fake reviews:")
print(get_top_words(title_words, title_coefficients, direction=1))

print("\nTop text terms associated with fake reviews:")
print(get_top_words(text_words, text_coefficients, direction=1), "\n")

print("=" * 60, "\n")

print("Top title terms associated with real reviews:")
print(get_top_words(title_words, title_coefficients, direction=0))

print("\nTop text terms associated with real reviews:")
print(get_top_words(text_words, text_coefficients, direction=0))

print("\n", "=" * 60, "\n")

# Model feature importance
feature_names = np.concatenate([
    "title: " + title_words,
    "text: " + text_words,
    "text_char: " + char_features,
    np.array(["rating", "helpful_vote"]),
])

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": np.abs(coefficients),
    "Coefficient": coefficients,
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print("Most important terms found by model:")
print(feature_importance.head(20))
