import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tarfile
import zipfile
from tqdm import tqdm

"""
Dataset downloader / extractor.

Notes:
- Supports selecting sources via `--sources` (e.g. `TUMVI` only).
- Includes retries and longer timeouts vs the original script.
- Some hosts (notably TUM-VI CDN) may present TLS/DNS issues depending on the network.
  Use `--insecure` only if you understand the risk (disables TLS cert verification).
"""


URLS = {
    # EuRoC MAV dataset (zip per sequence)
    "EUROC": [
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/",
    ],
    # KITTI raw data (zip per drive) - optional
    "KITTI": [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/",
    ],
    # TUM-VI exported in EuRoC format (tar per sequence)
    "TUMVI": [
        "https://vision.in.tum.de/tumvi/exported/euroc/512_16/",
        # Alternate CDNs sometimes referenced by the dataset page.
        "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/",
        "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/",
        "https://cdn1.vision.in.tum.de/tumvi/exported/euroc/512_16/",
    ],
    # Blackbird UAV dataset (download only the minimal CSV/GT files needed by this repo).
    "BLACKBIRD": [
        "https://ijrr20-blackbird-dataset.s3-accelerate.amazonaws.com/BlackbirdDatasetData/",
        "http://blackbird-dataset.mit.edu/BlackbirdDatasetData/",
    ],
}


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
    "BLACKBIRD": [
        "clover/yawConstant/maxSpeed3p0",
        "clover/yawForward/maxSpeed3p0",
        "halfMoon/yawConstant/maxSpeed4p0",
        "mouse/yawForward/maxSpeed5p0",
        "picasso/yawConstant/maxSpeed4p0",
        "sphinx/yawForward/maxSpeed4p0",
        "thrice/yawForward/maxSpeed5p0",
        "winter/yawConstant/maxSpeed4p0",
        "mouse/yawConstant/maxSpeed5p0",
        "sid/yawForward/maxSpeed4p0",
        "star/yawConstant/maxSpeed5p0",
        "clover/yawForward/maxSpeed5p0",
        "mouse/yawForward/maxSpeed6p0",
        "thrice/yawConstant/maxSpeed6p0",
        "tiltedThrice/yawForward/maxSpeed6p0",
    ],
}


BLACKBIRD_GLOBAL_FILES = [
    "trajectoryOffsets.yaml",
]

BLACKBIRD_SEQUENCE_FILES = [
    "flightNormalizationOffset.csv",
    "groundTruthPoses.csv",
    "csv/blackbird_slash_imu.csv",
    "csv/blackbird_slash_state.csv",
]


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_nonempty_dir(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def download_file(
    url: str,
    output_path: str,
    *,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    session: requests.Session,
    verify_tls: bool,
) -> bool:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[INFO] File already exists: {output_path}. Skipping download.")
        return True

    ensure_directory_exists(os.path.dirname(output_path))

    headers = {"User-Agent": "denoise-imu-gyro-downloader/1.0"}
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, stream=True, timeout=timeout_s, headers=headers, verify=verify_tls)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(output_path, "wb") as f, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written = f.write(chunk)
                    bar.update(written)
            return True
        except requests.RequestException as e:
            print(f"[ERROR] Failed to download {url} (attempt {attempt}/{retries}): {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    return False


def extract_file(path: str, extract_to: str, *, keep_images: bool) -> bool:
    if _is_nonempty_dir(extract_to):
        print(f"[INFO] Dataset already extracted: {extract_to}. Skipping extraction.")
        return True

    if not os.path.exists(path):
        print(f"[WARN] Archive not found, skipping extraction: {path}")
        return False

    def should_extract(name: str) -> bool:
        if keep_images:
            return True
        skip_patterns = [
            "/cam0/", "/cam1/", "/cam2/", "/cam3/",
            "/image_00/", "/image_01/", "/image_02/", "/image_03/",
            "/velodyne_points/", "/pointcloud0/"
        ]
        return not any(p in name for p in skip_patterns)

    ensure_directory_exists(extract_to)
    try:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                members = [m for m in zf.namelist() if should_extract(m)]
                zf.extractall(extract_to, members=members)
        elif path.endswith(".tar"):
            with tarfile.open(path, "r") as tf:
                members = [m for m in tf.getmembers() if should_extract(m.name)]
                tf.extractall(path=extract_to, members=members)
        else:
            print(f"[WARN] Unknown archive type: {path}")
            return False
        return True
    except (zipfile.BadZipFile, tarfile.TarError, FileNotFoundError) as e:
        print(f"[ERROR] Failed to extract {path}: {e}")
        return False


def process_dataset(
    source: str,
    dataset_path: str,
    download_folder: str,
    extract_folder: str,
    *,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    session: requests.Session,
    verify_tls: bool,
    keep_images: bool,
) -> None:
    file_name = os.path.basename(dataset_path)
    download_path = os.path.join(download_folder, file_name)
    extract_path = os.path.join(extract_folder, os.path.splitext(file_name)[0])

    if _is_nonempty_dir(extract_path):
        print(f"[INFO] Dataset already exists: {extract_path}. Skipping.")
        return

    for base_url in URLS[source]:
        file_url = base_url + dataset_path
        ok = download_file(
            file_url,
            download_path,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            session=session,
            verify_tls=verify_tls,
        )
        if ok:
            extracted = extract_file(download_path, extract_path, keep_images=keep_images)
            if extracted:
                return
    print(f"[ERROR] Failed to download/extract {dataset_path} from all configured URLs for {source}.")


def _download_blackbird_file(
    base_url: str,
    rel_path: str,
    extract_folder: str,
    *,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    session: requests.Session,
    verify_tls: bool,
) -> bool:
    output_path = os.path.join(extract_folder, rel_path)
    return download_file(
        base_url + rel_path,
        output_path,
        timeout_s=timeout_s,
        retries=retries,
        backoff_s=backoff_s,
        session=session,
        verify_tls=verify_tls,
    )


def process_blackbird_flight(
    flight: str,
    extract_folder: str,
    *,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    session: requests.Session,
    verify_tls: bool,
) -> None:
    required = [os.path.join(flight, rel) for rel in BLACKBIRD_SEQUENCE_FILES]
    if all(os.path.isfile(os.path.join(extract_folder, rel)) for rel in required):
        print(f"[INFO] Blackbird flight already present: {flight}. Skipping.")
        return

    for base_url in URLS["BLACKBIRD"]:
        ok = True
        for rel in required:
            if not _download_blackbird_file(
                base_url,
                rel,
                extract_folder,
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                session=session,
                verify_tls=verify_tls,
            ):
                ok = False
                break
        if ok:
            return
    print(f"[ERROR] Failed to download Blackbird flight {flight!r} from all configured URLs.")


def download_blackbird_globals(
    extract_folder: str,
    *,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    session: requests.Session,
    verify_tls: bool,
) -> None:
    for rel in BLACKBIRD_GLOBAL_FILES:
        for base_url in URLS["BLACKBIRD"]:
            if _download_blackbird_file(
                base_url,
                rel,
                extract_folder,
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                session=session,
                verify_tls=verify_tls,
            ):
                break
        else:
            print(f"[WARN] Failed to download optional Blackbird global file: {rel}")


def download_and_extract_datasets(
    sources: list[str],
    *,
    concurrent_downloads: int,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    verify_tls: bool,
    keep_images: bool,
) -> None:
    session = requests.Session()
    for source in sources:
        download_folder = os.path.join("data", source, "downloads")
        extract_folder = os.path.join("data", source, "dataset")
        ensure_directory_exists(download_folder)
        ensure_directory_exists(extract_folder)

        print(f"[INFO] Processing datasets for {source}...")
        if source == "BLACKBIRD":
            if keep_images:
                print("[WARN] --images is not supported for BLACKBIRD in this downloader; downloading minimal CSV/GT files only.")
            download_blackbird_globals(
                extract_folder,
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                session=session,
                verify_tls=verify_tls,
            )
            with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
                futures = [
                    executor.submit(
                        process_blackbird_flight,
                        flight,
                        extract_folder,
                        timeout_s=timeout_s,
                        retries=retries,
                        backoff_s=backoff_s,
                        session=session,
                        verify_tls=verify_tls,
                    )
                    for flight in DATASETS[source]
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Exception occurred: {e}")
            continue
        with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
            futures = [
                executor.submit(
                    process_dataset,
                    source,
                    dataset,
                    download_folder,
                    extract_folder,
                    timeout_s=timeout_s,
                    retries=retries,
                    backoff_s=backoff_s,
                    session=session,
                    verify_tls=verify_tls,
                    keep_images=keep_images,
                )
                for dataset in DATASETS[source]
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Exception occurred: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="EUROC,TUMVI",
        help="Comma-separated list: EUROC,TUMVI,KITTI,BLACKBIRD (default: EUROC,TUMVI)",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--backoff-s", type=float, default=2.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for downloads (use only if required).",
    )
    parser.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract image and pointcloud data (skipped by default to save space).",
    )
    args = parser.parse_args()

    sources = [s.strip().upper() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in URLS]
    if unknown:
        print(f"Unknown sources: {unknown}. Valid: {sorted(URLS)}")
        return 2

    download_and_extract_datasets(
        sources=sources,
        concurrent_downloads=args.max_workers,
        timeout_s=args.timeout_s,
        retries=args.retries,
        backoff_s=args.backoff_s,
        verify_tls=not args.insecure,
        keep_images=args.images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
