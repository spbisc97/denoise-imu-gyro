import os
import requests
import zipfile
from tqdm import tqdm

# Base URL for KITTI dataset
KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"

# List of dataset files to download
DATASETS = [
    "2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip",
    "2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",
    "2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",
    "2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip",
    "2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip"
]

# Folders for downloaded files and extracted data
DOWNLOAD_FOLDER = "data/KITTI/downloads"
EXTRACT_FOLDER = "data/KITTI/extractions"

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

def extract_zip(zip_path, extract_to):
    """Extract a .zip file to the specified folder."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_extract_datasets():
    """Download and extract the KITTI dataset files."""
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    if not os.path.exists(EXTRACT_FOLDER):
        os.makedirs(EXTRACT_FOLDER)

    for dataset_file in DATASETS:
        file_url = KITTI_BASE_URL + dataset_file
        download_path = os.path.join(DOWNLOAD_FOLDER, os.path.basename(dataset_file))
        extract_path = os.path.join(EXTRACT_FOLDER, os.path.splitext(os.path.basename(dataset_file))[0])

        # Download the dataset
        print(f"Downloading {dataset_file} from {file_url}...")
        download_file(file_url, download_path)

        # Extract the dataset
        print(f"Extracting {dataset_file} to {extract_path}...")
        extract_zip(download_path, extract_path)

if __name__ == "__main__":
    download_and_extract_datasets()
