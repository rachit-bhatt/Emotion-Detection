import streamlit as st
import pickle
import joblib
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# model_file_name = 'model.pkl'
model_filename = 'lstm_model_with_regularization.h5'
model = load_model(model_filename)

# # Load the saved SVM model
# with open(model_file_name, 'rb') as f:
#     svm_model = pickle.load(f)

# Load the trained SVM model
# svm_model = joblib.load(model_file_name)

# def predict_sentiment(model, tokenizer, text, label_encoder, max_length):
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
    
#     # Reshape the padded input to add the third dimension (features = 1)
#     text_reshaped = np.expand_dims(text_padded, axis=-1)  # Add 3rd dimension
    
#     # Predict using the specified model
#     prediction = model.predict(text_reshaped)
    
#     # Decode the predicted label
#     predicted_label_index = np.argmax(prediction, axis=1)[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
#     return predicted_label

# Define a function to preprocess and predict sentiment for a given model
def predict_sentiment(model, tokenizer, text, label_encoder, max_length):
    """
    Predicts the sentiment of the given text using the specified model.
    
    Parameters:
        model: Trained model (e.g., BiLSTM or LSTM+GRU)
        tokenizer: Tokenizer used during training
        text: Input text (string)
        label_encoder: Label encoder used for decoding class labels
        max_length: Maximum sequence length used during training
    
    Returns:
        Predicted sentiment label (string)
    """
    # Preprocess the text: Tokenize and pad
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Reshape the input to match the expected shape for the model
    text_padded = np.reshape(text_padded, (text_padded.shape[0], 1, text_padded.shape[1]))  # Shape: (1, 1, 100)
    
    # Predict using the specified model
    prediction = model.predict(text_padded)
    
    # Decode the predicted label
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
    return predicted_label

# Get user input for text
# user_input = st.text_input("Enter text for sentiment analysis:")

def x(user_input):

    # Perform sentiment analysis using the SVM model
    if user_input:
        # Then use the trained SVM model to make predictions
        # predicted_label = svm_model.predict([user_input])[0]

        tokenizer = Tokenizer()
        # Load the dataframe
        data = pd.read_csv('Preprocessed_data_1.csv')

        # Create a copy of the dataframe
        pre_df = data.copy()

        # Handling missing values in the 'text' column
        pre_df['text'] = pre_df['text'].fillna('')

        # Drop rows with empty strings in the 'text' column
        pre_df = pre_df[pre_df['text'] != '']

        # Reset the index of the DataFrame
        pre_df.reset_index(drop=True, inplace=True)
        texts = pre_df['text']
        tokenizer.fit_on_texts(texts)

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Vectorization
        vectorizer = TfidfVectorizer()
        text_vectorized = vectorizer.fit_transform(pre_df['text'].values.astype('U'))
        svd = TruncatedSVD(n_components=100)
        text_vectorized_svd = svd.fit_transform(text_vectorized)
        vectorized_df = pd.DataFrame(text_vectorized_svd)
        pre_df_vectorized = pd.concat([pre_df, vectorized_df], axis=1)

        X = text_vectorized_svd
        y = pre_df['encoded_labels']
        _, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

        label_encoder.fit(y_train)

        lstm_gru_sentiment = predict_sentiment(model, tokenizer, user_input, label_encoder, 100)
        print(f"LSTM+GRU Model Predicted Sentiment: {lstm_gru_sentiment}")

        return lstm_gru_sentiment

        # Display the sentiment prediction
        # st.write("Sentiment:", predicted_label)

#Defining a function to collect the user input and make predictions

def main():
    # print('Finding Result...')
    # result = x('This is a nice restaurant.')
    # print('Result:',result)

    st.title('Sentiment Analysis with NLP Model')
    text_input = st.text_area('Enter text:', 'Type your text here...')
    if st.button('Analyze'):
        if text_input:
            result = x(text_input)
            # sentiment = result[0]['label']
            # confidence = result[0]['score']
            # st.write(f'Sentiment: {sentiment} with confidence: {confidence:.2f}')
            print('Result:', result)
            if result == 0: # Negative
                st.write('Result: Negative')
            elif result == 1: # Negative
                st.write('Result: Neutral')
            elif result == 2: # Negative
                st.write('Result: Positive')

if __name__ == '__main__':
    main()