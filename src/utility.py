import os

def list_files(directory):
    if not os.path.isdir(directory):
        print("{} is not an existing directory")
        return None
    
    return os.listdir(directory)