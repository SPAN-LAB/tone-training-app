import os
import random
import shutil

# Define source and destination directories
source_dir = r'R:\projects\trial_classification\Tones_database\tones\tone_perfect'
dest_dir = r'R:\projects\trial_classification\Tones_database\stratified_audio_files'

# Define the strata: syllables, tones, and speakers
syllables = ['bu', 'di', 'lu', 'ma', 'mi']
tones = ['1', '2', '3', '4']
speakers = ['MV1', 'MV2', 'MV3', 'FV1', 'FV2', 'FV3']

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Stratified sample
sampled_files = []

# For each tone, sample 3 files per syllable
for tone in tones:
    for syllable in syllables:
        # Find all files matching the current syllable and tone
        matching_files = [f for f in os.listdir(source_dir) if f.startswith(f"{syllable}{tone}_")]

        # If fewer than 3 files are available, take all of them (to avoid errors)
        sample_count = min(3, len(matching_files))

        # Randomly sample 3 files
        sampled_files.extend(random.sample(matching_files, sample_count))

# Copy the sampled files to the destination folder
for file in sampled_files:
    source_file = os.path.join(source_dir, file)
    dest_file = os.path.join(dest_dir, file)
    shutil.copy(source_file, dest_file)
    print(f"Copied: {file}")

print("Stratified sampling complete.")

