import streamlit as st
import pickle
import joblib

model_file_name = 'model.pkl'

# Load the saved SVM model
with open(model_file_name, 'rb') as f:
    SVM_model = pickle.load(f)

# Load the trained SVM model
svm_model = joblib.load(model_file_name)

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
            result = svm_model(text_input)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            st.write(f'Sentiment: {sentiment} with confidence: {confidence:.2f}')


if __name__ == '__main__':
    main()