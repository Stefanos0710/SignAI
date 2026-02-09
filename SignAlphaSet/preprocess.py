import logging
import os
import time

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# folder of the alphabet-dataset
dataset_folder = "SignAlphaSet/data/SignAlphaSet/SignAlphaSet"

# extract keypoints from image
def extract_keypoints(image_path):
    pass

# center the keypoints
def center_keypoints(x, y):
    pass

# normalize the keypoints
def normalize_keypoints(x, y):
    pass

# save the keypoints to a file
def save_keypoints(keypoints, save_path):
    pass

# split the dataset into train, val and test
def split_dataset(dataset_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    pass

if __name__ == "__main__":
    alphabet = sorted(os.listdir(dataset_folder))  #  sort to["A", "B", ..., "Z"]

    for class_name in alphabet:
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            logging.info(f"Class '{class_name}' has {len(files)} files.")

            # print all files from class_name
            for file_name in files:
                print(file_name)
                
                # extract keypoints from current image
                tmp_keypoints = extract_keypoints(os.path.join(class_path, file_name))

                # center the keypoints
                centered_keypoints = center_keypoints(tmp_keypoints)

                # normalize the keypoints
                normalized_keypoints = normalize_keypoints(centered_keypoints)

                # save the keypoints to a file
                save_path = os.path.join("SignAlphaSet/data/tmp_keypoints", class_name)
                save_keypoints(normalized_keypoints, save_path)
                
    # split the dataset into train, val and test
    split_dataset(dataset_folder)
