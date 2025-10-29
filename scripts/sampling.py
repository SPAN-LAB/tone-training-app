import csv
import random
from collections import defaultdict

def load_csv_data(csv_file):
    data = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def select_evenly_distributed_trials(data, num_per_syllable_tone=6):
    # Group trials by syllable and tone
    grouped_data = defaultdict(list)
    
    # Organize data by syllable-tone and gender
    for trial in data:
        syllable = trial['syllable']
        tone = trial['tone']
        #gender = trial['gender'].lower()  # Normalize gender to lowercase
        group_key = f"{syllable}_tone{tone}"
        #if gender in grouped_data[group_key]:
        grouped_data[group_key].append(trial)

    # Create an evenly distributed sample of trials
    selected_trials = []
    
    for group_key, trials_list in grouped_data.items():
        #male_speakers = speakers['male']
        #female_speakers = speakers['female']
        
        # Check if we have enough speakers in each category
        if len(trials_list) < num_per_syllable_tone: 
            raise ValueError(f"Not enough speakers for {group_key}")
        
        # Randomly select an equal number of male and female speakers
        selected_trials.extend(random.sample(trials_list, num_per_syllable_tone))
        #selected_trials.extend(random.sample(female_speakers, num_per_syllable_tone // 2))
    
    return selected_trials

def save_selected_trials_to_csv(trials, output_file):
    # Write the selected trials to a new CSV file
    if trials:
        fieldnames = trials[0].keys()  # Use the field names from the first trial (same as the original CSV)
        with open(output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trials)
        print(f"Selected trials saved to {output_file}")

# Example usage:
csv_file = 'output_data.csv'
output_file = 'evenly_distributed_trials.csv'

data = load_csv_data(csv_file)
trials = select_evenly_distributed_trials(data, num_per_syllable_tone=6)  # 6 total speakers per syllable-tone combination (3 male, 3 female)

save_selected_trials_to_csv(trials, output_file)  # Save the selected trials to a new CSV file

