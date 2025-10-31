import pandas as pd
import re
import nltk
from nltk.corpus import words, stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_raw_spam_csv(path="data/spam.csv"):
    df = pd.read_csv(
        path,
        encoding="latin1",
        engine="python",
        usecols=[0, 1],
        names=["class", "message"],
        header=0,
    )
    return df

def remove_special_characters(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_messages(df):
    df = df.copy()
    df["clean_message"] = df["message"].astype(str).apply(remove_special_characters)
    return df


def encode_labels(df):
    df = df.copy()
    df["label"] = LabelEncoder().fit_transform(df["class"])
    return df


def split_data(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label"], test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def create_vectorizer(max_features=3000):
    return TfidfVectorizer(max_features=max_features)


def vectorize_train_test(vectorizer, X_train, X_test):
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf