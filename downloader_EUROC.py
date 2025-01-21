import os
import requests
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URL for EuRoC dataset
EUROC_BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"

# List of dataset paths to download
DATASETS = [
    "machine_hall/MH_01_easy/MH_01_easy.zip",
    "machine_hall/MH_02_easy/MH_02_easy.zip",
    "machine_hall/MH_03_medium/MH_03_medium.zip",
    "machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "machine_hall/MH_05_difficult/MH_05_difficult.zip",
    "vicon_room1/V1_01_easy/V1_01_easy.zip",
    "vicon_room1/V1_02_medium/V1_02_medium.zip",
    "vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "vicon_room2/V2_01_easy/V2_01_easy.zip",
    "vicon_room2/V2_02_medium/V2_02_medium.zip",
    "vicon_room2/V2_03_difficult/V2_03_difficult.zip",
]

# Folders for downloaded files and extracted data
DOWNLOAD_FOLDER = "data/EUROC/downloads"
EXTRACT_FOLDER = "data/EUROC/datasets"

def ensure_directory_exists(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}. Skipping download.")
        return

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an error for HTTP issues
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
        print(f"Downloaded: {output_path}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)  # Clean up partial files

def extract_zip(zip_path, extract_to):
    """Extract a .zip file to the specified folder."""
    if os.path.exists(extract_to):
        print(f"Dataset already extracted: {extract_to}. Skipping extraction.")
        return

    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to: {extract_to}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting {zip_path}: {e}")

def process_dataset(dataset_path):
    """Process a single dataset: download and extract."""
    file_url = EUROC_BASE_URL + dataset_path
    file_name = os.path.basename(dataset_path)
    download_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    extract_path = os.path.join(EXTRACT_FOLDER, os.path.splitext(file_name)[0])

    # Download and extract dataset
    download_file(file_url, download_path)
    extract_zip(download_path, extract_path)

def download_and_extract_datasets(concurrent_downloads=4):
    """Download and extract the EuRoC dataset files."""
    ensure_directory_exists(DOWNLOAD_FOLDER)
    ensure_directory_exists(EXTRACT_FOLDER)

    print(f"Starting dataset download and extraction...")
    with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
        futures = [executor.submit(process_dataset, dataset) for dataset in DATASETS]
        for future in as_completed(futures):
            try:
                future.result()  # Raises any exceptions caught during execution
            except Exception as e:
                print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    download_and_extract_datasets()
