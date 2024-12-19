import pandas as pd

file_path = 'evenly_distributed_trials.csv'
df = pd.read_csv(file_path)

df['syllable_trimmed'] = df['syllable'].apply(lambda x: x[:-1])  # This removes the last character

distinct_trimmed_syllables = df['syllable_trimmed'].unique()


syllable_count = len(distinct_trimmed_syllables)

with open('output_syllables.txt', 'w') as f:
        for syllable in distinct_trimmed_syllables:
            f.write(f"{syllable}\n")  # Write each syllable in a new line
        f.write(f"\nTotal number of distinct syllables: {syllable_count}\n")

print(f"Total number of distinct syllables: {syllable_count}")
