import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pickle

import importlib.util



# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def tune_model(train_data,X_train):
    
        # Define a range of hyperparameters to sample randomly
    param_dist = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
        'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    }
   
    # Create an SVM model
    model = SVC()

    # Perform randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
    random_search.fit(X_train, train_data['CATEGORY-ENCODED'])
    
   
  
    # best_params = random_search.best_params_

    # Get the best hyperparameters
    return random_search.best_estimator_

   
   
  

def train_model(best_model,test_data_soln,X_test):
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)
    report = classification_report(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)

    print("Classification Report of Logistic Regression:")
    print(report)
    print(f'Logistic Regression Accuracy: {accuracy}')


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
    global best_model
    train_set_file=getfile()
      
    # Split the data into train and test sets 
    train_data, test_data = train_test_split(pd.read_csv(train_set_file), test_size=0.2, random_state=42)
    
    # Fill missing values with "No content"
    train_data['v2'].fillna("No content", inplace=True)
    test_data['v2'].fillna("No content", inplace=True)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['v2'])
    X_test_tfidf = tfidf_vectorizer.transform(test_data['v2'])

    
    model=tune_model(train_data,X_train_tfidf)
    train_model(model,test_data,X_test_tfidf)
  


if __name__ == "__main__":
    main()
