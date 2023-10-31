import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pickle
from model_trainer import tune_model
import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

# def load_model():
   

def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects/Spam-SMS-Detection'): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)=='train_processed.csv'): #filename with extension
            train_set_filename=filename
    return train_set_filename

def main():
    train_set_file=getfile()
      
    # Split the data into train and test sets 
    train_data, test_data = train_test_split(pd.read_csv(train_set_file), test_size=0.2, random_state=42)
    
    # Fill missing values with "No content"
    train_data['v2'].fillna("No content", inplace=True)
    test_data['v2'].fillna("No content", inplace=True)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['v2'])
    X_test_tfidf = tfidf_vectorizer.transform(test_data['v2'])

    with open('D:/Projects/Spam-SMS-Detection/models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

    
    # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Fit the encoder and transform the labels
    train_labels = label_encoder.fit_transform(train_data['v1'])
    test_labels = label_encoder.fit_transform(test_data['v1'])
    
    # Open the file in binary write mode and save the label encoder
    with open('D:/Projects/Spam-SMS-Detection/models/label_encoding.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    best_model=tune_model(train_data,X_train_tfidf)
    # Save the trained model to a pickle file
    with open('D:/Projects/Spam-SMS-Detection/models/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

if __name__ == "__main__":
    main()