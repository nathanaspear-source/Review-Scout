import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent/"data"/"final_labeled_fake_reviews.csv"
MODEL_PATH = Path(__file__).resolve().parent/"ml_model"/"models.joblib"

# Loading fake reviews dataset as csv and dropping unneeded columns
reviews = pd.read_csv(DATA_PATH)
reviews = reviews.drop(["images", "verified_purchase", "asin", "parent_asin", "timestamp", "user_timestamp"], axis=1)

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

# Printing shapes of X_title and X_text matrices
print("X_train TF-IDF:", X_train.shape)
print("X_test TF-IDF:", X_test.shape)

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
print("Model saved successfully")

# Printing model evaluation scores
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Getting coefficients learned by Logistic Regression Model
coefficients = model.coef_[0]

# Getting number of features in each TF-IDF matrix
n_title = X_train_title.shape[1]
n_text = X_train_text.shape[1]
n_text_char = X_train_text_char.shape[1]

# Getting TF-IDF feature names
title_words = title_vectorizer.get_feature_names_out()
text_words = text_vectorizer.get_feature_names_out()
char_features = text_char_vectorizer.get_feature_names_out()

# Coefficients for each feature group
title_coefficients = coefficients[:n_title]
text_coefficients = coefficients[n_title:n_title + n_text]
char_coefficients = coefficients[n_title + n_text:n_title + n_text+ n_text_char]

numeric_coefficients = coefficients[
    n_title + n_text + n_text_char:
]

# Checking to see if length of word and coefficient arrays are same length for title and text features
print("\n Title:", len(title_words), len(title_coefficients))
print("Text:", len(text_words), len(text_coefficients), "\n")

def get_top_words(words, coefs, n=5, direction=1):
    results = pd.DataFrame({
        "word_or_phrase": words,
        "coefficient": coefs,
    })

    results["direction"] = np.where(
        results["coefficient"] > 0,
        "Fake (class 1)",
        "Genuine (class 0)"
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

print("Top title terms associated with fake reviews:")
print(get_top_words(title_words, title_coefficients, direction=1))

print("\nTop text terms associated with fake reviews:")
print(get_top_words(text_words, text_coefficients, direction=1), "\n")

print("=" * 60)

print("Top title terms associated with real reviews:")
print(get_top_words(title_words, title_coefficients, direction=0))

print("\nTop text terms associated with real reviews:")
print(get_top_words(text_words, text_coefficients, direction=0), "\n")

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

print(feature_importance.head(20))