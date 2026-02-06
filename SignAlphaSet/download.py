import requests
from tqdm import tqdm
import logging
import time
from datetime import datetime
import zipfile

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

url = "https://data.mendeley.com/public-api/zip/8fmvr9m98w/download/2"
out = "SignAlphaSet_v2.zip"


def download_file(url, out):
    start_time = time.time()

    logging.info("This download contatins the SignAlphaSet dataset.")
    logging.info("Download started ...")
    logging.info(f"URL: {url}")
    logging.info(f"Location: {out}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        size_mb = total / (1024 * 1024)

        logging.info(f"File size: {size_mb:.2f} MB")
        logging.info("Connection successful – Download in progress ...")

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


if __name__ == "__main__":
    download_file(url, out)
