import os
import requests
import json
import zipfile

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

kaggle_username = "USERNAME"
kaggle_key = "USERKEY"
dataset_path = "shayanfazeli/heartbeat"

data_path = root_dir.joinpath('data')
Path(data_path).mkdir(parents=True, exist_ok=True)
save_path = data_path.joinpath('data.zip')

# Set up the API URL and headers
KAGGLE_API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_path}"

print(f"Downloading {KAGGLE_API_URL}")
# Request the dataset
try:
    response = requests.get(KAGGLE_API_URL, headers={
        "Authorization": f"Basic {kaggle_username}:{kaggle_key}"
    }, stream=True)
    response.raise_for_status()  # Check if request was successful

    # Write the dataset to file
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Dataset downloaded and saved as {save_path}")

except requests.exceptions.RequestException as e:
    print("Error downloading dataset:", e)

mitbih_dir = data_path.joinpath("mitbih")
ptbdb_dir = data_path.joinpath("ptbdb")

print(f"Making {mitbih_dir} directory")
Path(mitbih_dir).mkdir(parents=True, exist_ok=True)
print(f"Making {ptbdb_dir} directory")
Path(ptbdb_dir).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(save_path, 'r') as zip_ref:
    for file_info in zip_ref.infolist():
        if "mitbih" in file_info.filename:
            print(f"Extract {file_info} into {mitbih_dir}")
            zip_ref.extract(file_info, mitbih_dir)
        elif "ptbdb" in file_info.filename:
            print(f"Extract {file_info} into {ptbdb_dir}")
            zip_ref.extract(file_info, ptbdb_dir)
        else:
            print(f"file_info {file_info} not used.")