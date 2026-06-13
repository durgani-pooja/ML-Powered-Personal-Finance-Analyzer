from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data for categorization
TRAINING_DATA = [
    ("grocery store supermarket food mart", "Food"),
    ("restaurant dinner lunch breakfast cafe coffee", "Food"),
    ("netflix spotify amazon prime hulu entertainment movie", "Entertainment"),
    ("electric bill water bill internet phone utility", "Utilities"),
    ("uber taxi bus train transport petrol gas fuel", "Transport"),
    ("salary bonus freelance income payment received", "Income"),
    ("amazon flipkart shopping clothes shoes purchase", "Shopping"),
    ("gym doctor pharmacy hospital medicine health", "Health"),
    ("course book school college tuition education", "Education"),
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([t[0] for t in TRAINING_DATA])
y = [t[1] for t in TRAINING_DATA]
model = LogisticRegression()
model.fit(X, y)

def load_classifier():
    return (vectorizer, model)

def categorize_transactions(df, classifier):
    vectorizer, model = classifier
    descriptions = df['Description'].str.lower().tolist()
    X = vectorizer.transform(descriptions)
    df['Category'] = model.predict(X)
    return df