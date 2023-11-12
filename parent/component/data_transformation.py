import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import chardet

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
# from textblob import TextBlob

import os
import pandas as pd
import pathlib 

import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))
# sys.path.append(source_folder_path)

# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)



# Download the NLTK data needed for tokenization and stopwords
nltk.download(source_file.NLTK_DOWNLOAD)
nltk.download(source_file.NLTK_STOPWORDS)

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

def process_genre(genre,column):
    label_encoded_column=source_file.LABEL_ENCODED_COLUMN
     #  Text Lowercasing
    genre[column] = genre[column].str.lower()
    # Removing Special Characters and Punctuation
    genre[column] = genre[column].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    #  Remove html tags
    genre[column] = genre[column].apply(remove_html_tags)
    # Remove emoji
    genre[column] = genre[column].apply(remove_emojis)
    # label encoding the genre column
    label_encoder = LabelEncoder()
    genre[label_encoded_column] = label_encoder.fit_transform(genre[column])

def process_plot(plot_description,column):   
    
   
    #  Text Lowercasing
    plot_description[column] = plot_description[column].str.lower()
    # Removing Special Characters and Punctuation
    plot_description[column] = plot_description[column].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    #  Remove html tags
    plot_description[column] = plot_description[column].apply(remove_html_tags)
    # Remove emoji
    plot_description[column] = plot_description[column].apply(remove_emojis)
    # Tokenization
    plot_description[column] = plot_description[column].apply(word_tokenize)
    #  Stop Word Removal
    stop_words = set(stopwords.words('english'))
    plot_description[column] = plot_description[column].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # # Spelling Correction (using TextBlob)
    # plot_description[column] = plot_description[column].apply(lambda tokens: " ".join([str(TextBlob(token).correct()) for token in tokens]))

    #  Stemming (using NLTK)
    stemmer = PorterStemmer()
    plot_description[column] = plot_description[column].apply(lambda tokens: " ".join([stemmer.stem(token) for token in tokens]))

    # # Impute missing values with a specific content
    # plot_description[column].fillna("No content", inplace=True)
    
    return plot_description


def getfile():
    path=[]
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): 
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.'+source_file.TRAIN_SET.split('.')[1]):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
  
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET):
            train_set_filename=filename
        
    return train_set_filename

def batch_processing(data):
  
    batch_size = 1000  
    column_to_clean = source_file.COLUMN_TO_CLEAN
    column_to_encode=source_file.COLUMN_TO_ENCODE
    processed_data=pd.DataFrame()
    
    
    for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
        
            # Get the current batch of data
            batch_data = data.iloc[batch_start:batch_end]
            processed_data = pd.concat([processed_data, process_plot(batch_data,column_to_clean),process_genre(batch_data,column_to_encode)])

    return processed_data
   

def main():
  
    train_file=source_file.TRAIN_SET_PROCESSED_PATH
    train_set_file=getfile()

    #determine the encoding of the csv file
    with open(train_set_file, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    
    
    processed_train=batch_processing(pd.read_csv(train_set_file,encoding=result['encoding']))
    processed_train.to_csv(train_file, index=False)
        
if __name__ == "__main__":
    main()

