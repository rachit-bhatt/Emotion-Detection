from flask import Flask, request, jsonify, render_template
from pandas import DataFrame, concat
import pickle
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (update the filename and loading method based on your model type)
model_filename = 'lstm_model_with_regularization.h5'
model = load_model(model_filename)
print('Model Loaded.')

# Define the label mapping (update based on your model's output classes)
EMOTION_LABELS = {
    0: 'Happy',
    1: 'Sad',
    2: 'Angry',
    3: 'Fearful',
    4: 'Neutral',
}

@app.route('/')
def home():
    return render_template('index.html')  # You need an index.html file in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Extract input data from the request (adjust for your data format)
        data = request.json
        if not data or 'comment' not in data:
            return jsonify({'error': 'Invalid input, expected a JSON object with a "comment" field.'}), 400

        comment = data['comment']
        # You may need to preprocess the comment based on your model's requirements
        # Example: Convert text to feature vector (update this as needed)
        processed_input = preprocess_comment([comment])  # Define your preprocessing function
        print('Pre-Processed Input:', processed_input)

        try:
            # Perform the prediction
            prediction = model.predict(processed_input)
            print('Prediction:', prediction)
        except Exception as ex:
            print('Exception!', ex)

        emotion = EMOTION_LABELS.get(prediction[0], 'Unknown')
        print('Emotion:', emotion)

        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_comment(comments):
    """Placeholder for preprocessing the input comment. Update this based on your model."""
    # Example: Tokenization, vectorization, etc.
    # Here we simply return the raw comment; replace with actual preprocessing logic
    vector = TfidfVectorizer()
    text_vector = vector.fit_transform(comments)
    svd = TruncatedSVD()
    vector_df = svd.fit_transform(text_vector)
    # vector_df = DataFrame(vector_df)
    # return concat([DataFrame(comments), vector_df], axis = 1)
    return vector_df

if __name__ == '__main__':
    app.run(debug=True)