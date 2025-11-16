import os
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# basis dir from this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# dir to download WLASL data into (project root /data folder)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "data"))

os.makedirs(DATA_DIR, exist_ok=True)

# Load the dataset in FiftyOne-Dataset-Name file
print(f"Load dataset 'WLASL' in directory: {DATA_DIR}")
dataset = fouh.load_from_hub(
    "Voxel51/WLASL",
    name="WLASL",
    dataset_dir=DATA_DIR,
)

print("Finished downloading dataset.")

# start app
if __name__ == "__main__":
    print("Starts FiftyOne app ...")
    session = fo.launch_app(dataset)
    session.wait()  # hold window open until closed by user
