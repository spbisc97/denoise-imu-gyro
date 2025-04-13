import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URLs for datasets
URLS = {
    "EUROC": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/",
    "KITTI": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/",
    "TUMVI": "https://vision.in.tum.de/tumvi/exported/euroc/512_16/",
}

# Datasets to download
DATASETS = {
    "EUROC": [
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
    ],
    "KITTI": [
        "2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip",
        "2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",
        "2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",
        "2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip",
    ],
    "TUMVI": [
        "dataset-corridor1_512_16.tar",
        "dataset-corridor2_512_16.tar",
        "dataset-magistrale1_512_16.tar",
        "dataset-magistrale2_512_16.tar",
        "dataset-room1_512_16.tar",
        "dataset-room2_512_16.tar",
        "dataset-room3_512_16.tar",
        "dataset-room4_512_16.tar",
        "dataset-room5_512_16.tar",
        "dataset-room6_512_16.tar",
    ],
}

def ensure_directory_exists(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    if os.path.exists(output_path):
        print(f"[INFO] File already exists: {output_path}. Skipping download.")
        return

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                size = file.write(chunk)
                bar.update(size)
    except requests.RequestException as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)  # Remove incomplete files

def extract_file(path, extract_to):
    """Extract a .zip or .tar file to the specified directory."""
    if os.path.exists(extract_to):
        print(f"[INFO] Dataset already extracted: {extract_to}. Skipping extraction.")
        return

    try:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif path.endswith(".tar"):
            with tarfile.open(path, 'r') as tar_ref:
                tar_ref.extractall(path=extract_to)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[ERROR] Failed to extract {path}: {e}")

def process_dataset(source_url, dataset_path, download_folder, extract_folder):
    """Download and extract a dataset."""
    file_url = source_url + dataset_path
    file_name = os.path.basename(dataset_path)
    download_path = os.path.join(download_folder, file_name)
    extract_path = os.path.join(extract_folder, os.path.splitext(file_name)[0])

    # Check if the dataset is already extracted
    if os.path.exists(extract_path):
        print(f"[INFO] Dataset already exists: {extract_path}. Skipping download and extraction.")
        return

    # Download and extract dataset
    download_file(file_url, download_path)
    extract_file(download_path, extract_path)

def download_and_extract_datasets(concurrent_downloads=6):
    """Download and extract all datasets."""
    for source in URLS.keys():
        source_url = URLS[source]
        download_folder = os.path.join("data", source, "downloads")
        extract_folder = os.path.join("data", source, "dataset")

        ensure_directory_exists(download_folder)
        ensure_directory_exists(extract_folder)

        print(f"[INFO] Processing datasets for {source}...")
        with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
            futures = [
                executor.submit(process_dataset, source_url, dataset, download_folder, extract_folder)
                for dataset in DATASETS[source]
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Exception occurred: {e}")

if __name__ == "__main__":
    download_and_extract_datasets()
