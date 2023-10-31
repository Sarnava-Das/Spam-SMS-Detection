import subprocess
import os
import importlib.util


# Specify the absolute path to __init__.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))
# sys.path.append(source_folder_path)

# Use importlib to import __init__.py
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

if __name__ == '__main__':
    

    # path to the requirements.txt file
    requirements_file = os.path.join(source_file.ROOT_DIR,'requirements.txt')

    # pip install 
    try:
        subprocess.check_call(['pip', 'install', '-r', requirements_file])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install packages.")