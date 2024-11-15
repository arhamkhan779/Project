from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from tensorflow import keras

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
ls=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load preprocessor and model

model_path = "artifacts/trained_model.h5"
model = keras.models.load_model(model_path)

def preprocess(text):
    preprocess_text=[]
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [ls.lemmatize(word) for word in words if word not in stop_words]
    preprocess_text.append(' '.join(words))
    one_hot_rep = [keras.preprocessing.text.one_hot(words, 10000) for words in preprocess_text] 
    padded_docs = keras.preprocessing.sequence.pad_sequences(one_hot_rep, padding='pre', maxlen=600)
    return padded_docs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocess text and make prediction
        text_transformed = preprocess(text)
        prediction = model.predict(text_transformed)
        
        # Determine if it's human or AI-generated text
        prediction_label = 'Human Text' if prediction[0] < 0.5 else 'AI Generated Text'
        
        return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
