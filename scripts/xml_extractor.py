import os
import csv
import xml.etree.ElementTree as ET

def extract_data_from_xml(directory, output_csv):
    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        # Add 'audio_file' column to the CSV
        fieldnames = ['tone', 'syllable', 'gender', 'speaker', 'audio_file']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Traverse through XML files in the directory
        for filename in os.listdir(directory):
            if filename.endswith("_CUSTOM.xml"):  # Filter only _CUSTOM XML files
                file_path = os.path.join(directory, filename)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    # Extract relevant data from the XML structure
                    tone = root.find('tone').text
                    syllable = root.find('syllable').text
                    gender = root.find('gender').text
                    speaker = root.find('speaker').text

                    # Construct the audio file name using the extracted data
                    audio_file = f"{syllable}_{speaker}_MP3"  # Construct the audio file name

                    # Write data to CSV with the new audio_file column
                    writer.writerow({
                        'tone': tone,
                        'syllable': syllable,
                        'gender': gender,
                        'speaker': speaker,
                        'audio_file': audio_file
                    })

                except ET.ParseError as e:
                    print(f"Error parsing {file_path}: {e}")
                except AttributeError as e:
                    print(f"Missing field in {file_path}: {e}")

    print(f"Data extraction complete. CSV saved as: {output_csv}")

# Example usage
extract_data_from_xml(r'R:\projects\trial_classification\Tones_database\tones\xml', 'output_data.csv')
