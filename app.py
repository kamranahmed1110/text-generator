from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model
model = load_model('text_generation_model.h5')

# Load and fit the tokenizer (ensure the same tokenizer is used as in training)
tokenizer = Tokenizer()
with open('next_word_predictor.txt', 'r', encoding='utf-8') as file:
    text = file.read()
tokenizer.fit_on_texts([text])

# Function to generate text
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Initialize Flask app
app = Flask(__name__)

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Use "Once upon a time" as the seed text for prediction
    seed_text = data.get('seed_text', 'Once upon a time')  # Default seed text is "Once upon a time"
    next_words = data.get('next_words', 20)  # Default is to generate 20 words
    max_sequence_len = data.get('max_sequence_len', 100)  # Default max sequence length

    # Generate the predicted text
    generated_text = generate_text(seed_text, next_words, max_sequence_len)
    
    return jsonify({"generated_text": generated_text})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
