import os

def file_exists(file):
    return os.path.isfile(file)

def file_absolute_path(file):
    return os.path.abspath(file)