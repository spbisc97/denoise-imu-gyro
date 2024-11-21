import os
import requests
import tarfile
from tqdm import tqdm

# Base URL for TUM VI dataset
TUM_VI_BASE_URL = "https://vision.in.tum.de/tumvi/exported/euroc/512_16/"

# List of dataset files to download
DATASETS = [
    "dataset-corridor1_512_16.tar",
    "dataset-corridor2_512_16.tar",
    "dataset-magistrale1_512_16.tar",
    "dataset-magistrale2_512_16.tar",
    "dataset-room1_512_16.tar",
    "dataset-room2_512_16.tar",
]

# Folders for downloaded files and extracted data
DOWNLOAD_FOLDER = "data/TUMVI/downloads"
EXTRACT_FOLDER = "data/TUMVI/extractions"

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download {url}. HTTP Status Code: {response.status_code}")
        return
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tar_path, extract_to):
    """Extract a .tar file to the specified folder."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)

def download_and_extract_datasets():
    """Download and extract the TUM VI dataset files."""
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    if not os.path.exists(EXTRACT_FOLDER):
        os.makedirs(EXTRACT_FOLDER)

    for dataset_path in DATASETS:
        file_url = TUM_VI_BASE_URL + dataset_path
        file_name = os.path.basename(dataset_path)
        download_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        extract_path = os.path.join(EXTRACT_FOLDER, os.path.splitext(file_name)[0])

        # Download the dataset
        print(f"Downloading {file_name} from {file_url}...")
        download_file(file_url, download_path)

        # Extract the dataset
        print(f"Extracting {file_name} to {extract_path}...")
        extract_tar(download_path, extract_path)

if __name__ == "__main__":
    download_and_extract_datasets()
