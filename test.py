import joblib
from utils import remove_special_characters

LABELS = {0: "ham", 1: "spam"}


model = joblib.load("models/spam_classifier.joblib")
vectorizer = joblib.load("models/tfidf.joblib")


text = input("Enter a message to classify: ")


cleaned = remove_special_characters(text)
X = vectorizer.transform([cleaned])
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0][pred]


print("\n--- Prediction Result ---")
print(f"Original : {text}")
print(f"Cleaned  : {cleaned}")
print(f"Class    : {LABELS[int(pred)]}")
print(f"Confidence: {proba:.3f}")
