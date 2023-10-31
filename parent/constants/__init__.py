import os

ROOT_DIR = os.getcwd()  #to get current working directory

# Data ingestion related variables
DATASET_IDENTIFIER = 'hijest/genre-classification-dataset-imdb'
DATASET_DIR = "datasets"
DATASET_DESTINATION_PATH = os.path.join(ROOT_DIR,DATASET_DIR)
INPUT_ENCODING='ISO-8859-1'
# OUTPUT_ENCODING='utf-8'
OUTPUT_ENCODING='latin-1'


# Data transformation,evaluation related variables
LABEL_ENCODED_COLUMN = 'CATEGORY-ENCODED'
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










# # Data transformation related variable
# DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
# DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
# DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
# DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
# DATA_INGESTION_ZIP_DOWNLOAD_DIR_KEY = "zip_download_dir"
# DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
# DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
# DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

# # Data Validation related variables
# DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
# DATA_VALIDATION_ARTIFACT_DIR = "data_validation"
# DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
# DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
# DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
# DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"

# # Data Transformation related variables
# DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
# DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
# DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
# DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
# DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
# DATA_TRANSFORMATION_PREPROCESSING_DIR_NAME_KEY = "preprocessing_dir"
# DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME_KEY = "preprocessed_object_file_name"

# # These variables are to read from the schema.yaml file
# DATASET_SCHEMA_COLUMNS_KEY = "columns"
# NUMERICAL_COLUMN_KEY = "numerical_columns"
# CATEGORICAL_COLUMN_KEY = "categorical_columns"
# TARGET_COLUMN_NAME_KEY = "target_column"

# # Model Training related variables
# MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
# MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
# MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
# MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
# MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
# MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
# MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"

# # Production model evaluation related variables
# MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
# MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"
# MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"


# # Model pusher config key
# MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
# MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

# BEST_MODEL_KEY = "best_model"
# HISTORY_KEY = "history"
# MODEL_PATH_KEY = "model_path"

# EXPERIMENT_DIR_NAME="experiment"
# EXPERIMENT_FILE_NAME="experiment.csv"