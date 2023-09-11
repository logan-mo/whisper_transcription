from google.cloud import storage
import os
import zipfile

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
