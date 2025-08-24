# movie-recommendation-system-
This document provides a comprehensive overview of the Movie Review Sentiment Analysis project, a machine learning application designed to classify movie reviews as either positive or negative. The project was developed as part of a task focused on building and deploying a complete ML solution.

## 1. Project Goal
The primary objective of this project is to create a robust machine learning model capable of accurately determining the sentiment of a given movie review. The model classifies reviews into one of two categories: positive or negative. A secondary, but critical, objective was to develop a simple web interface to demonstrate the model's real-world application.

## 2. Dataset
The project utilizes the IMDb Movie Reviews Dataset, which contains 50,000 movie reviews, each labeled with a corresponding sentiment (positive or negative). This balanced dataset is ideal for training and evaluating a binary classification model. The data is provided in a single CSV file, IMDB Dataset.csv.

## 3. Methodology
The project follows a standard machine learning pipeline, consisting of data preprocessing, model training, and evaluation.

## Text Preprocessing:
The raw text reviews are prepared for the model through a series of preprocessing steps to remove noise and standardize the text. These steps include:

Lowercase Conversion: Converting all text to lowercase to ensure consistency.

HTML Tag Removal: Removing <br/> and other HTML tags that are present in the dataset.

Punctuation and Special Character Removal: Filtering out non-alphabetic characters.

Tokenization & Stopword Removal: Splitting the text into individual words (tokens) and removing common English words (e.g., "the," "is," "a") that do not contribute to sentiment.

Lemmatization: Reducing words to their base or root form (e.g., "running" becomes "run").

## Feature Engineering: 
The preprocessed text data is converted into a numerical format that the model can understand. This is achieved using the TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer. This technique weighs words by their importance in a document relative to their frequency across all documents, effectively highlighting key terms that drive sentiment.

## Model Training: 
A Logistic Regression model was chosen for its interpretability and strong baseline performance in text classification tasks. The model was trained on the preprocessed and vectorized data, learning the patterns associated with positive and negative sentiments.

## 4. Model Performance
The performance of the sentiment analysis model was evaluated using two key metrics: Accuracy and F1-Score. While the exact metrics can vary slightly depending on the training run, a typical Logistic Regression model on this dataset achieves high performance.

## Accuracy:
A measure of the proportion of correctly classified reviews. The model typically achieves an accuracy in the range of 88% to 90%, indicating that it correctly classifies nearly 9 out of 10 reviews.

## F1-Score: 
The harmonic mean of precision and recall. It is particularly useful for binary classification to account for class imbalance, though the IMDb dataset is balanced. The F1-score for this model is also in the range of 88% to 90%, confirming its high performance.

These metrics demonstrate the model's effectiveness in accurately predicting the sentiment of movie reviews.

# 5. Application
The project includes a lightweight web application that allows users to test the model with custom reviews. This interface is built using Flask (app.py) for the backend and standard HTML/CSS/JavaScript (index.html) for the frontend.

## app.py: 
This Python script serves as the application's backend. It defines a web server, handles a prediction API endpoint, and uses a pre-trained model file (sentiment_analysis_model.pkl) to make sentiment predictions. It includes a preprocess_text function to ensure consistency between the training data and new user inputs.

## index.html: 
This file provides the user interface. It contains a text area for users to enter a review, a button to trigger the analysis, and a section to display the prediction result. The JavaScript code handles the communication with the Flask backend.
