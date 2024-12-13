from flask import Flask, request, jsonify, render_template
from pandas import DataFrame, concat
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import numpy as np

import streamlit as st
import pickle

# Load the saved SVM model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

import streamlit as st
import joblib

# Load the trained SVM model
svm_model = joblib.load('sentiment_model_1.h5')

# Get user input for text
user_input = st.text_input("Enter text for sentiment analysis:")

# Perform sentiment analysis using the SVM model
if user_input:
    # Then use the trained SVM model to make predictions
    predicted_label = svm_model.predict([user_input])[0]

    # Display the sentiment prediction
    st.write("Sentiment:", predicted_label)

#Defining a function to collect the user input and make predictions

def main():
    st.title('Sentiment Analysis with NLP Model')
    text_input = st.text_area('Enter text:', 'Type your text here...')
    if st.button('Analyze'):
        if text_input:
            result = model(text_input)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            st.write(f'Sentiment: {sentiment} with confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()

# # Initialize the Flask app
# app = Flask(__name__)

# # Load your trained model (update the filename and loading method based on your model type)
# model_filename = 'sentiment_model_1.h5'
# model = load_model(model_filename)
# print('Model Loaded.')

# # Define the label mapping (update based on your model's output classes)
# EMOTION_LABELS = {
#     0: 'Negative',
#     1: 'Neutral',
#     2: 'Positive'
# }

# @app.route('/')
# def home():
#     return render_template('index.html')  # You need an index.html file in a 'templates' folder

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle prediction requests."""
#     try:
#         # Extract input data from the request (adjust for your data format)
#         data = request.json
#         if not data or 'comment' not in data:
#             return jsonify({'error': 'Invalid input, expected a JSON object with a "comment" field.'}), 400

#         comment = data['comment']
#         # You may need to preprocess the comment based on your model's requirements
#         # Example: Convert text to feature vector (update this as needed)
#         processed_input = preprocess_text(comment)  # Define your preprocessing function
#         print('Pre-Processed Input:', processed_input)

#         try:
#             # Perform the prediction
#             prediction = model.predict(processed_input)
#             print('Prediction:', prediction)
#         except Exception as ex:
#             print('Exception!', ex)

#         emotion = EMOTION_LABELS.get(prediction[0], 'Unknown')
#         print('Emotion:', emotion)

#         return jsonify({'emotion': emotion})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
# def preprocessing_text(model, comment):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_text(comment)
#     label_encoder = LabelEncoder()
    

#     return predict_sentiment(model, tokenizer, comment, label_encoder)
    
# def predict_sentiment(model, tokenizer, text, label_encoder, max_length = 100):
#     """
#     Predicts the sentiment of the given text using the specified model.
    
#     Parameters:
#         model: Trained model (e.g., BiLSTM or LSTM+GRU)
#         tokenizer: Tokenizer used during training
#         text: Input text (string)
#         label_encoder: Label encoder used for decoding class labels
#         max_length: Maximum sequence length used during training
    
#     Returns:
#         Predicted sentiment label (string)
#     """
#     # Preprocess the text: Tokenize and pad
#     text_sequence = tokenizer.texts_to_sequences([text])
#     text_padded = pad_sequences(text_sequence, maxlen=max_length, padding='post', truncating='post')
    
#     # Reshape the input to match the expected shape for the model
#     text_padded = np.reshape(text_padded, (text_padded.shape[0], 1, text_padded.shape[1]))  # Shape: (1, 1, 100)
    
#     # Predict using the specified model
#     prediction = model.predict(text_padded)
    
#     # Decode the predicted label
#     predicted_label_index = np.argmax(prediction, axis=1)[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
#     return predicted_label

# if __name__ == '__main__':
#     app.run(debug=True)