import os
import zipfile
import subprocess
import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))
# sys.path.append(source_folder_path)

# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def read_data():
   
    dataset_identifier = source_file.DATASET_IDENTIFIER
    destination_folder = source_file.DATASET_DESTINATION_PATH

    # Run the Kaggle command to download the dataset
    command = f'kaggle datasets download -d {dataset_identifier} -p {destination_folder} --force'
    subprocess.call(command, shell=True)

   
    file_list = os.listdir(destination_folder)
    zip_files = [file for file in file_list if file.endswith('.zip')]
    for zip_file in zip_files:
        # Define the name of the downloaded zip file
        zip_file_name=zip_file
  
    # Unzip the downloaded file to the destination folder
    zip_file_path = os.path.join(destination_folder, zip_file_name)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    
def main():
    read_data()
 
if __name__ == "__main__":
    main()