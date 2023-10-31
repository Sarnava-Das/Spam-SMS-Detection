import os

ROOT_DIR = os.getcwd()  #to get current working directory

# Data ingestion related variables
DATASET_IDENTIFIER = 'hijest/genre-classification-dataset-imdb'
DATASET_DIR = "datasets"
DATASET_DESTINATION_PATH = os.path.join(ROOT_DIR,DATASET_DIR)
INPUT_ENCODING='ISO-8859-1'
OUTPUT_ENCODING='utf-8'



# Data transformation,evaluation related variables
LABEL_ENCODED_COLUMN = 'GENRE-ENCODED'
COLUMN_TO_CLEAN='DESCRIPTION'
COLUMN_TO_ENCODE ='GENRE'
TRAIN_SET='train_data.csv'
TEST_SET='test_data.csv'
TEST_SET_SOLN='test_data_solution.csv'
DATASET_PROCESSED_DIR='Genre Classification Dataset'
TRAIN_SET_PROCESSED_NAME='train_processed.csv'
TEST_SET_PROCESSED_NAME='test_processed.csv'
TEST_SET_SOLN_PROCESSED_NAME='test_data_solution_processed.csv'
TRAIN_SET_PROCESSED_PATH=os.path.join(ROOT_DIR,DATASET_DIR,DATASET_PROCESSED_DIR,TRAIN_SET_PROCESSED_NAME)
TEST_SET_PROCESSED_PATH=os.path.join(ROOT_DIR,DATASET_DIR,DATASET_PROCESSED_DIR,TEST_SET_PROCESSED_NAME)
TEST_SET_SOLN_PROCESSED_PATH=os.path.join(ROOT_DIR,DATASET_DIR,DATASET_PROCESSED_DIR,TEST_SET_SOLN_PROCESSED_NAME)
NLTK_DOWNLOAD='punkt'
NLTK_STOPWORDS='stopwords'

# Model pusher related variables
MODEL_DIR='models'
PRED_MODEL_NAME='model.pkl'
TFIDF_NAME='tfidf_vectorizer.pkl'
ENCODING_NAME='label_encoding.pkl'
ENCODING_PATH=os.path.join(ROOT_DIR,MODEL_DIR,ENCODING_NAME)
TFIDF_PATH=os.path.join(ROOT_DIR,MODEL_DIR,TFIDF_NAME)
PRED_MODEL_PATH=os.path.join(ROOT_DIR,MODEL_DIR,PRED_MODEL_NAME)

