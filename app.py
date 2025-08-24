from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
import nltk
import sys

# --- Preprocessing Function (copy from your original code) ---
def preprocess_text(text):
    """Cleans and preprocesses a single text review."""
    text = text.lower()
    clean_text = re.sub(r'<br\s*/>', ' ', text)
    clean_text = re.sub(r'[^a-z\s]', '', clean_text)
    tokens = clean_text.split()
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

# --- Load the trained model ---
try:
    model = joblib.load('sentiment_analysis_model.pkl')
except FileNotFoundError:
    print("Model file not found. Ensure 'sentiment_analysis_model.pkl' is in the same directory.")
    sys.exit(1)

app = Flask(__name__)

# --- Web Page Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    processed_review = preprocess_text(review)
    prediction = model.predict([processed_review])[0]
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)