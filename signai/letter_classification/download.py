import requests
from tqdm import tqdm
import logging
import time
import zipfile
import os

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------
# Setup folder for dataset (must be BEFORE the banner)
# ----------------------------
dataset_folder = "signai/letter_classification/data/dataset"
os.makedirs(dataset_folder, exist_ok=True)

# URL to download the dataset
url = "https://data.mendeley.com/public-api/zip/8fmvr9m98w/download/2"

# Output ZIP path
out = os.path.join(dataset_folder, "SignAlphaSet_v2.zip")

# ----------------------------
# Dataset Info Banner (now safe)
# ----------------------------
print("\n" + "="*70)
print("  SignAlphaSet Dataset Downloader  ")
print("="*70)
print("Dataset Name      : SignAlphaSet")
print("Description       : A collection of sign language pictures of the ASL alphabet, ")
print("                    designed for use in training and research in machine learning")
print(f"Dataset URL       : {url}")
print(f"Target Folder     : {dataset_folder}")
print("Notes             :")
print("  - The dataset is provided as a ZIP archive")
print("  - This script will download and extract it automatically")
print("  - Ensure you have a stable internet connection for the download")
print("="*70 + "\n")

def download_file(url, out):
    if not os.path.exists("signai/letter_classification/data/SignAlphaSet/SignAlphaSet/A") and not os.path.exists("signai/letter_classification/data/ASL_dynamic/ASL_dynamic/A"):
        start_time = time.time()

        logging.info("Download started ...")
        logging.info(f"URL: {url}")
        logging.info(f"Location: {out}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            size_mb = total / (1024 * 1024)

            logging.info(f"File size: {size_mb:.2f} MB")
            logging.info("Connection successful\nDownload in progress ...")

            with open(out, "wb") as f, tqdm(
                desc="Downloading",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            duration = time.time() - start_time
            logging.info(f"Download completed in {duration:.1f} seconds")

        except requests.exceptions.Timeout:
            logging.error("Timeout – Server not responding")

        except requests.exceptions.ConnectionError:
            logging.error("No internet connection")

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error: {e}")

        except KeyboardInterrupt:
            logging.warning("Download manually interrupted (Ctrl+C)")

        except Exception as e:
            logging.error(f"Unknown error: {e}")
        
    else:
        # skip download if file already exists
        logging.info("File already exists, skipping download.")

def extract_zip(file_path):
    """
    Extracts the main dataset zip.
    Then extracts the two inner archives:
        - ASL_dynamic.zip
        - SignAlphaSet.zip
    Finally removes all zip files.
    """

    start_time = time.time()

    base_folder = "signai/letter_classification/data"
    os.makedirs(base_folder, exist_ok=True)

    dynamic_folder = os.path.join(base_folder, "ASL_dynamic")
    signalpha_folder = os.path.join(base_folder, "SignAlphaSet")

    os.makedirs(dynamic_folder, exist_ok=True)
    os.makedirs(signalpha_folder, exist_ok=True)

    if os.path.exists("signai/letter_classification/data/SignAlphaSet/ASL_dynamic.zip") and os.path.exists("signai/letter_classification/data/SignAlphaSet/SignAlphaSet.zip"):

        # ----------------------------
        # 1) Extract main archive
        # ----------------------------
        logging.info("Extracting main archive ...")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(base_folder)

        # paths of known inner zips
        dynamic_zip = "signai/letter_classification/data/SignAlphaSet/ASL_dynamic.zip"
        signalpha_zip = "signai/letter_classification/data/SignAlphaSet/SignAlphaSet.zip"


        # ----------------------------
        # 2) Extract ASL_dynamic
        # ----------------------------
        logging.info("Extracting ASL_dynamic.zip ...")

        with zipfile.ZipFile(dynamic_zip, 'r') as zip_ref:
            zip_ref.extractall(dynamic_folder)

        # ----------------------------
        # 3) Extract SignAlphaSet
        # ----------------------------
        logging.info("Extracting SignAlphaSet.zip ...")

        with zipfile.ZipFile(signalpha_zip, 'r') as zip_ref:
            zip_ref.extractall(signalpha_folder)

        # ----------------------------
        # 4) Cleanup zip files
        # ----------------------------
        logging.info("Removing zip files ...")

        for z in [file_path, dynamic_zip, signalpha_zip]:
            if os.path.exists(z):
                os.remove(z)

        duration = time.time() - start_time
        logging.info(f"Extraction finished in {duration:.1f} seconds")

    elif os.path.exists("signai/letter_classification/data/SignAlphaSet/SignAlphaSet/A") and os.path.exists("signai/letter_classification/data/ASL_dynamic/ASL_dynamic/A"):
        logging.warning("Inner zip files have already been extracted. Skipping extraction step.")

    else:
        logging.error("Expected inner zip files not found. Extraction aborted.")



if __name__ == "__main__":
    # start 
    start_download = time.time()
    download_file(url, out)

    start_extract = time.time()
    extract_zip(out)
    
    # calc duration
    download_duration = start_extract - start_download
    extract_duration = time.time() - start_extract

    def get_folder_size_mb(folder):
        total = 0
        for root, _, files in os.walk(folder):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)


    data_root = "signai/letter_classification/data"
    dataset_size = get_folder_size_mb(data_root)
    num_files = sum(len(files) for _, _, files in os.walk(data_root))

    # print summary and stats
    print("\n" + "="*70)
    print("  Download & Extraction Completed!  ")
    print("="*70)
    print(f"Dataset Root Folder : {data_root}")
    print(f"ASL_dynamic Folder  : {os.path.join(data_root, 'ASL_dynamic')}")
    print(f"SignAlphaSet Folder : {os.path.join(data_root, 'SignAlphaSet')}")
    print(f"Total Size          : {dataset_size:.2f} MB")
    print(f"Download Duration   : {download_duration:.1f} seconds")
    print(f"Extraction Duration : {extract_duration:.1f} seconds")
    print(f"Number of files     : {num_files}")
    print("="*70)
