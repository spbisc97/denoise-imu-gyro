import os
import requests
import zipfile
from tqdm import tqdm

# Base URL for EuRoC dataset
EUROC_BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"

# List of dataset paths to download
DATASETS = [
    "machine_hall/MH_01_easy/MH_01_easy.zip",
    "machine_hall/MH_02_easy/MH_02_easy.zip",
    "machine_hall/MH_03_medium/MH_03_medium.zip",
    "machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "vicon_room1/V1_01_easy/V1_01_easy.zip",
    "vicon_room1/V1_02_medium/V1_02_medium.zip",
    "vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "vicon_room2/V2_01_easy/V2_01_easy.zip",
    "vicon_room2/V2_02_medium/V2_02_medium.zip",
    "vicon_room2/V2_03_difficult/V2_03_difficult.zip",
]

# Folders for downloaded files and extracted data
DOWNLOAD_FOLDER = "data/EUROC/downloads"
EXTRACT_FOLDER = "data/EUROC/extractions"

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
    """Download and extract the EuRoC dataset files."""
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
    if not os.path.exists(EXTRACT_FOLDER):
        os.makedirs(EXTRACT_FOLDER)

    for dataset_path in DATASETS:
        file_url = EUROC_BASE_URL + dataset_path
        file_name = os.path.basename(dataset_path)
        download_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        extract_path = os.path.join(EXTRACT_FOLDER, os.path.splitext(file_name)[0])

        # Download the dataset
        print(f"Downloading {file_name} from {file_url}...")
        download_file(file_url, download_path)

        # Extract the dataset
        print(f"Extracting {file_name} to {extract_path}...")
        extract_zip(download_path, extract_path)

if __name__ == "__main__":
    download_and_extract_datasets()
