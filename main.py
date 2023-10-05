from flask import Flask, request, jsonify
import easyocr
import cv2
import re
import numpy as np


app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['ar'], gpu=True)

# Create a custom character set that includes Arabic punctuation marks
custom_character_set = set(['.', '؟', '!'])

# Define a function to split the string into words, including ".", "?", and "!"
def split_words(text):
    """Splits the given text into words, including ".", "?", and "!".

    Args:
      text: The text to split.

    Returns:
      A list of words.
    """
    # Use regular expression to split text into words
    words = re.findall(r'\S+|[.؟!?]', text)
    return words

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        # Get the uploaded image from the request
        image = request.files['image']

        # Load the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarization
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform text recognition on the preprocessed image
        result = reader.readtext(thresholded_image)

        # Reverse the order of recognition results (for right-to-left reading)
        result.reverse()

        # Initialize variables to store words and their confidence scores
        words = []
        confidence_sum = 0.0

        # Process and append each recognized word and its corresponding confidence score
        for res in result:
            sentence = res[1]
            # Split the sentence into words, including ".", "?", and "!"
            sentence_words = split_words(sentence)

            for word in sentence_words:
                confidence = res[2]
                words.append((word, confidence))
                confidence_sum += confidence

        # Calculate the average confidence score
        num_words = len(words)
        average_confidence = confidence_sum / num_words

        # Join the words with spaces to form the sentence
        sentence = ' '.join([word for word, _ in words])

        # Create a response JSON
        response_data = {
            'sentence': sentence,
            'average_confidence': average_confidence
        }

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


