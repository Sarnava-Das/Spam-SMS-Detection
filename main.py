# Import necessary libraries
import pickle
from flask import Flask, request, render_template

import re
import string
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

# Download the NLTK data needed for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder='templates')

# Load the pickled model
with open('D:/Projects/Spam-SMS-Detection/models/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
tfidf_vectorizer_path = 'D:/Projects/Spam-SMS-Detection/models/tfidf_vectorizer.pkl'
with open(tfidf_vectorizer_path, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Open the file in binary read mode and load the label encoder
with open('D:/Projects/Spam-SMS-Detection/models/label_encoding.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Define a function to remove emojis using a regular expression
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                           u"\U0001F700-\U0001F77F"  # Alphabetic presentation forms
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U0001F004-\U0001F0CF"  # Additional emoticons
                           u"\U0001F110-\U0001F251"  # Geometric Shapes Extended
                           u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
                           u"\U0001F910-\U0001F91E"  # Emoticons
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def process_sms(sentence):   
    # Text Lowercasing
    sentence = sentence.lower()
    # Removing Special Characters and Punctuation
    sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
    # Remove html tags 
    sentence = remove_html_tags(sentence)
    # Remove emoji 
    sentence = remove_emojis(sentence)
    # Tokenization
    tokens = word_tokenize(sentence)
    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming (using NLTK)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)

# Define a function to preprocess the input and make predictions
def predict_sms(sms):
  # Clean and preprocess the input plot
    cleaned_sms = process_sms(sms)
    
    # Use the pre-trained TF-IDF vectorizer to transform the input text
    input_tfidf = tfidf_vectorizer.transform([cleaned_sms])
    
    prediction = model.predict(input_tfidf)
  
    #  use the loaded label encoder to decode labels
    decoded_prediction = loaded_label_encoder.inverse_transform(prediction)
    # You can convert it to a string and then strip the brackets.
    if str(decoded_prediction[0]) == 'ham':
        return 'Not Spam'
    elif str(decoded_prediction[0]) == 'spam':
        return 'Spam'

# Define a route to handle the HTML form
@app.route('/', methods=['GET', 'POST'])
def predict_spam_sms():
    if request.method == 'POST':
        sms = request.form['plot']
       
        prediction = predict_sms(sms)
      
        return render_template('index.html', spam=prediction)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    
    app.run()
