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
import csv
import re
import pympi

def clean_trancript(transcript):
    """Clean the transcript by removing unwanted characters and formatting."""
    transcript = transcript.lstrip('$')                # leading gesture/pointer marker
    transcript = re.sub(r'[\^\*]+$', '', transcript)    # removing ^ / * markers
    transcript = re.sub(r'\d+[A-Z]?$', '', transcript)  # removing variant number/letter

    return transcript

def create_csv_from_eaf(eaf_path, participants):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eaf_file", "participant", "index", "start_ms", "end_ms", "transcript"])

        for eaf_path in sorted(eaf_dir.glob("*.eaf")):
            eaf = pympi.Elan.Eaf(str(eaf_path))
            tier_names = eaf.get_tier_names()

            found_any = False
            for participant in participants:
                tier_name = f"Lexem_Gebärde_r_{participant}"
                if tier_name not in tier_names:
                    continue
                found_any = True

                annotations = eaf.get_annotation_data_for_tier(tier_name)
                for i, (start, end, transcription) in enumerate(annotations, start=1):
                    writer.writerow([eaf_path.name, participant, i, start, end, clean_trancript(transcription)])

            if not found_any:
                print(f"skipping {eaf_path.name}: no 'Lexem_Gebärde_r_A/B' tier")

        print(f"wrote segments to {output_csv}")

# set up paths of dataset and output files
dataset_dir = Path(__file__).resolve().parent / "dataset"
eaf_dir = dataset_dir / "eaf"
output_csv = dataset_dir / "segments.csv"

participants = ["A", "B"]

create_csv_from_eaf(eaf_dir, participants)