import os
import zipfile
import subprocess
import csv

def read_data():
   
    dataset_identifier = 'uciml/sms-spam-collection-dataset'
    destination_folder = 'D:/Projects/Spam-SMS-Detection/datasets'

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