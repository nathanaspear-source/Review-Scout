import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.sparse import hstack, csr_matrix

# Loading fake reviews dataset as csv and dropping unneeded columns
reviews = pd.read_csv("data/final_labeled_fake_reviews.csv")
reviews = reviews.drop(["images", "asin", "parent_asin", "timestamp", "user_timestamp"], axis=1)

# Removing rows with missing data
reviews = reviews.dropna()

# Sorting reviews DataFrame into features and label
X = reviews[["rating", "title", "text", "helpful_vote", "verified_purchase"]]
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
)

text_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
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

# Converting boolean verified_purchase column to numeric encodings
X_train_vp = csr_matrix(
    X_train["verified_purchase"].astype(int).values.reshape(-1, 1)
)

X_test_vp = csr_matrix(
    X_test["verified_purchase"].astype(int).values.reshape(-1, 1)
)

# Horizontally stacking train and test text-based data with other feature columns
X_train = hstack([
    X_train_title,
    X_train_text,
    X_train_numeric,
    X_train_vp
])

X_test = hstack([
    X_test_title,
    X_test_text,
    X_test_numeric,
    X_test_vp
])

# Printing shapes of X_title and X_text matrices
print("X_train TF-IDF:", X_train.shape)
print("X_test TF-IDF:", X_test.shape)

# LinearSVC Gridsearch
params = {"C": [0.1, 1, 10, 100, 1000],
          "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
}

grid = GridSearchCV(SVC(), params, refit = True, verbose = 3, n_jobs = -1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Training Logistic Regression Model
# model = LinearSVC()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))


