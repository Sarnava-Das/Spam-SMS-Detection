import os
import pandas as pd
import pathlib 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
import importlib.util

# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def nb_model(train_data,test_data,X_train,X_test):
    # Create and train the Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train, train_data[source_file.LABEL_ENCODED_COLUMN])

    # Make predictions on the test data
    predictions = naive_bayes_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(test_data[source_file.LABEL_ENCODED_COLUMN], predictions)
   
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")


def svm_model(train_data,test_data,X_train,X_test):
   
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train,train_data[source_file.LABEL_ENCODED_COLUMN] )
    predictions = svm_classifier.predict(X_test)

   
    accuracy = accuracy_score(test_data[source_file.LABEL_ENCODED_COLUMN], predictions)
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")

  
    
def lr_model(train_data,test_data_soln,X_train,X_test):
   
    
    # Create and train the Logistic Regression model
    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(X_train, train_data[source_file.LABEL_ENCODED_COLUMN])

    # Predict the genres for the test data
    y_pred = logistic_regression_model.predict(X_test)

  

    # Calculate accuracy 
    accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)
    
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
   

def getfile():
    path=[]
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): 
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.'+source_file.TRAIN_SET.split('.')[1]):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET_PROCESSED_NAME): 
            train_set_filename=filename
    return train_set_filename

def main():
    train_set_file=getfile()
      
    # Split the data into train and test sets 
    train_data, test_data = train_test_split(pd.read_csv(train_set_file), test_size=0.2, random_state=42)
    
    # Fill missing values with "No content"
    train_data[source_file.COLUMN_TO_CLEAN].fillna("No content", inplace=True)
    test_data[source_file.COLUMN_TO_CLEAN].fillna("No content", inplace=True)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data[source_file.COLUMN_TO_CLEAN])
    X_test_tfidf = tfidf_vectorizer.transform(test_data[source_file.COLUMN_TO_CLEAN])

    
    svm_model(train_data,test_data,X_train_tfidf,X_test_tfidf)
    nb_model(train_data,test_data,X_train_tfidf,X_test_tfidf)
    lr_model(train_data,test_data,X_train_tfidf,X_test_tfidf)

if __name__ == "__main__":
    main()