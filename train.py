# train.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils import (
    load_raw_spam_csv,
    clean_messages,
    encode_labels,
    split_data,
    create_vectorizer,
    vectorize_train_test,
)


df = load_raw_spam_csv("data/spam.csv")
df = clean_messages(df)
df = encode_labels(df)


X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)


vectorizer = create_vectorizer(max_features=3000)
X_train_tfidf, X_test_tfidf = vectorize_train_test(vectorizer, X_train, X_test)


model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, "models/spam_classifier.joblib")
joblib.dump(vectorizer, "tfidf.joblib")