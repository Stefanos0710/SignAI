"""
In this script, the downloaded dataset will be segmented into individual videos, each containing a single word. 
The segmentation is based on the time codes provided in the ELAN annotations (EAF files). 

Input:
 - dataset/
    - eaf/
        - ...
    videos/
        - ...

Output:

csv file with all segments

path, word, start_time, end_time, transcript, gloss


"""
from pathlib import Path
import re
import pympi

def clean_trancript(transcript):
    """Clean the transcript by removing unwanted characters and formatting."""
    transcript = transcript.lstrip('$')                # leading gesture/pointer marker
    transcript = re.sub(r'[\^\*]+$', '', transcript)    # trailing ^ / * markers
    transcript = re.sub(r'\d+[A-Z]?$', '', transcript)  # trailing variant number/letter

    return transcript

# set the path to the eaf file 
dataset_dir = Path(__file__).resolve().parent / "dataset"
eaf_path = dataset_dir / "eaf" / "1204239.eaf"
eaf = pympi.Elan.Eaf(str(eaf_path))

# set transciript tier
tier_name = "Gebärde_r_A"

# annotations for the tier
annotations = eaf.get_annotation_data_for_tier(tier_name)

# get number, start, end and translation for each annotation
for i, annotation in enumerate(annotations, start=1):
    start, end, transcription = annotation[:3]
    print(i, start, end, clean_trancript(transcription))
