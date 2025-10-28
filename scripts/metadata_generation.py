import os
import csv
import xml.etree.ElementTree as ET

# Extracts information from the XML files for the audio files sampled and being used for tone training. This info is
# placed in a metadata CSV file in the tone training app resources directory
def generate_metadata_csv(audio_directory, xml_directory, output_csv):
    # Get a list of relevant audio file names without extensions and remove '_MP3' from base names
    audio_files = {os.path.splitext(f)[0].replace('_MP3', '') for f in os.listdir(audio_directory) if f.endswith(".mp3")}
    
    # Debugging print to check audio files
    print(f"Audio files (cleaned base names): {audio_files}")

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'tone', 'syllable', 'gender', 'speaker']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Process XML files that end with "_CUSTOM.xml"
        for filename in os.listdir(xml_directory):
            if filename.endswith("_CUSTOM.xml"):
                file_base = os.path.splitext(filename)[0].replace('_CUSTOM', '')  # This removes _CUSTOM.xml

                # Debugging print to check file base and comparison
                # print(f"Processing file: {filename}, file_base: {file_base}")

                # Check if XML file base name matches one of the audio files
                if file_base in audio_files:
                    file_path = os.path.join(xml_directory, filename)
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    # Extract data based on provided XML structure
                    tone = root.find('tone').text if root.find('tone') is not None else 'N/A'
                    syllable = root.find('syllable').text if root.find('syllable') is not None else 'N/A'
                    gender = root.find('gender').text if root.find('gender') is not None else 'N/A'
                    speaker = root.find('speaker').text if root.find('speaker') is not None else 'N/A'

                    # Write extracted data to CSV
                    writer.writerow({
                        'filename': file_base,
                        'tone': f"tone {tone}",  # Formatting tone as "tone X"
                        'syllable': syllable,
                        'gender': gender,
                        'speaker': speaker
                    })
                    print(f"Written to CSV: {file_base}, tone {tone}, {syllable}, {gender}, {speaker}")

    print(f"Metadata CSV generated at: {output_csv}")

# Example usage
generate_metadata_csv(
    r'R:\projects\tone-training-app\resources\sounds', 
    r'R:\projects\trial_classification\Tones_database\tones\xml', 
    r'R:\projects\tone-training-app\resources\metadata.csv'
)