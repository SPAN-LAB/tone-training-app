import os
import shutil
import re

# Define source and destination directories
source_dir = r'R:\projects\trial_classification\Tones_database\tones\tone_perfect'
dest_dir = r'R:\projects\tone-training-app\resources\sounds'

# Define the syllables you want to extract
target_syllables = ['bu', 'di', 'lu', 'ma', 'mi']

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Loop through files in the source directory
for filename in os.listdir(source_dir):
    # Check if the filename starts with any of the target syllables
    if re.match(r"^(bu|di|lu|ma|mi)[1-4]_(MV[1-3]|FV[1-3])", filename):
        # Construct full file path
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        # Copy file to destination
        shutil.copy(source_file, dest_file)
        print(f"Copied: {filename}")

print("Extraction and copying complete.")
