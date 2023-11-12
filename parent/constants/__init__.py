import os

ROOT_DIR = os.getcwd()  #to get current working directory

# Data ingestion related variables
DATASET_IDENTIFIER = 'uciml/sms-spam-collection-dataset'
DATASET_DIR = "datasets"
DATASET_DESTINATION_PATH = os.path.join(ROOT_DIR,DATASET_DIR)

# Data transformation,evaluation related variables
LABEL_ENCODED_COLUMN = 'CATEGORY-ENCODED'
COLUMN_TO_CLEAN='v2'
COLUMN_TO_ENCODE ='v1'
TRAIN_SET='spam.csv'
TRAIN_SET_PROCESSED_NAME='train_processed.csv'
TRAIN_SET_PROCESSED_PATH=os.path.join(ROOT_DIR,DATASET_DIR,TRAIN_SET_PROCESSED_NAME)
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

