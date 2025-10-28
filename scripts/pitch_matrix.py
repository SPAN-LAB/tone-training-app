# Use pyin to extract Fundamental frequency (F0) and use min-max normalization based on speakers
import librosa
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

# Helper function to extract tone from audio files
def extract_tone_num(file):
    match = re.search(r"\d+", file)
    return int(match.group()) if match else float("inf")    

# Helper function to extract speaker from audio files
def extract_speaker(file):
    match = re.findall(r"[FM][V]\d+", file)[0]
    return match

### obtain audio files
main_path = os.path.join("/Volumes", "gurindapalli", "projects", "Plasticity_training", "stratified_audio_files")
files = os.listdir(main_path)
files.sort(key=extract_tone_num)

f0_dict = {}
tone_list = []
speaker_list = []

### make fundamental frequency (f0) dataframe
for f in files:

    # read audio files
    audio = os.path.join(main_path,f)
    y, sr = librosa.load(audio)

    # extract f0
    f0, _, voiced_prob = librosa.pyin(
        y=y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # remove unsound pitch
    threshold = 0.5
    f0 = np.where(voiced_prob > threshold, f0, np.nan)
    f0 = f0[~np.isnan(f0)]

    # get time frame for each frequency
    times = librosa.times_like(f0, sr=sr)
    f0_dict[f] = pd.Series(f0, index=times)

    # obtain tone number 
    tone = re.findall(r'\d+', f)[0]
    tone_list.append(tone)

    # obtain speaker index
    speaker = re.findall(r"[FM][V]\d+", f)[0]
    speaker_list.append(speaker)

# create dataframe
f0_df = pd.DataFrame(f0_dict).T
f0_df["tone"] = tone_list 
f0_df["speaker"] = speaker_list

# create subset without tone column
f0_df_subset = f0_df.drop(["tone"], axis=1)

# obtain min and max f0 value for each speaker and convert to dictionary
f0_max = f0_df_subset.groupby("speaker").max().max(axis=1).to_dict()
f0_min = f0_df_subset.groupby("speaker").min().min(axis=1).to_dict()

# create subset with only numeric columns
f0_df_subset = f0_df.select_dtypes(include=[np.number]).columns

# min-max normalization
f0_df_normalized = f0_df.copy()
for col in f0_df_subset:
    f0_df_normalized[col] = f0_df_normalized.apply(
        lambda row: (row[col] - f0_min[row['speaker']]) / 
                    (f0_max[row['speaker']] - f0_min[row['speaker']]),
        axis=1
    )

# convert to long format dataframe (if needed)
# speakers = f0_df_normalized["speaker"].unique().sort()  # obtain speakers
# tones = f0_df_normalized["tone"].unique().sort()        # obtain tones

#  # pivot df to long format
# f0_df_normalized = f0_df_normalized.melt(  
#     id_vars=["tone", "speaker"], 
#     var_name="timepoint", 
#     value_name="f0")   

# # convert timepoint to integer
# f0_df_normalized['timepoint'] = f0_df_normalized['timepoint'].astype(float)

# save as CSV
f0_df_normalized.to_csv("tone_matrix.csv", index=False)