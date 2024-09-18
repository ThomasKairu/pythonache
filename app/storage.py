import os
import shutil
from fastapi import UploadFile

UPLOAD_DIRECTORY = "uploads"

def save_upload_file(upload_file: UploadFile) -> str:
    try:
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIRECTORY, upload_file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return ""

def delete_file(file_path: str) -> bool:
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        return False