import gzip
import pickle

def read_gzip_pickle(file_path):
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
        print("Load pickle gzip file successfully!")
    return data

obj = read_gzip_pickle(
    r"C:\Users\stefa\Documents\GitHub\SignAI\data\PHOENIX-Weather-2014T-compressed\phoenix14t.pami0.dev.annotations_only.gzip"
)

for i in range(0, len(obj)):
    # var
    gloss = obj[i]['gloss']
    video_id = r"C:\Users\stefa\Documents\GitHub\SignAI\data\PHOENIX-Weather-2014T-compressed\videos_phoenix\videos/" + obj[i]["name"]

    print(gloss)
    print(video_id)
