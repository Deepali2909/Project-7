# app.py

from flask import Flask, request, render_template
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove punctuation
    text = text.lower().strip()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.join("model", "disaster_model.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        tweet = request.form["tweet"]
        cleaned_tweet = clean_text(tweet)
        features = vectorizer.transform([cleaned_tweet])
        result = model.predict(features)[0]
        prediction = "🔥 Disaster Tweet" if result == 1 else "✅ Not a Disaster Tweet"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
