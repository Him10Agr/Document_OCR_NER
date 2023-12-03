import os

BASE_DIR = os.getcwd()
MEDIA_DIR = 'static/media'

SAVE_DIR = os.path.join(BASE_DIR, MEDIA_DIR)

def join_path(dir, file_name):
    filepath = os.path.join(dir, file_name)
    return filepath