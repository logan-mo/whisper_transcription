from google.cloud import storage
import os
import zipfile

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists(os.path.join("data", "bismillah")):
    os.mkdir(os.path.join("data", "bismillah"))
if not os.path.exists("zip_files"):
    os.mkdir("zip_files")
if not os.path.exists("saved_model"):
    os.mkdir("saved_model")

storage_client = storage.Client("[whisper-arabic-transcription]")
bucket = storage_client.get_bucket("recitation_dataset")
blobs = list(storage_client.list_blobs("recitation_dataset"))

output_path = os.path.join("data", "bismillah")

for blob in blobs:
    zip_name = os.path.join("zip_files", blob.name)
    print(zip_name)
    blob.download_to_filename(zip_name)
    print("Extracting")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(os.path.join("data", "bismillah"))
