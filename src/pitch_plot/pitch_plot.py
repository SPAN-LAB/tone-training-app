# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the audio file of Mandarin Tone 1
# audio_path = './tests/di4_FV1_MP3.mp3'  # Replace with the actual path to your audio file
# y, sr = librosa.load(audio_path)

# # Create a spectrogram
# plt.figure(figsize=(12, 6))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram of Mandarin Tone 3')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')

# # Extract the pitch (fundamental frequency)
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# times = librosa.times_like(f0)

# # Overlay pitch contour on spectrogram
# plt.plot(times, f0, label='Pitch Contour (F0)', color='red', linewidth=2)
# plt.legend(loc='upper right')

# # Save the plot
# plt.savefig('mandarin_tone4_spectrogram.png', format='png', dpi=300)

# # Display the plot
# plt.show()

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the audio file of Mandarin Tone 1
# audio_path = './tests/di1_FV3_MP3.mp3'  # Replace with the actual path to your audio file
# y, sr = librosa.load(audio_path)

# # Extract the pitch (fundamental frequency) over time
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# times = librosa.times_like(f0)

# # Plot the pitch contour as a line plot
# plt.figure(figsize=(10, 4))
# plt.plot(times, f0, color='b', linewidth=2, label='Pitch Contour (F0)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Pitch Contour of Mandarin Tone 1')
# plt.legend(loc='upper right')
# plt.grid()
# plt.savefig('mandarin_tone1_pitch_contour2.png', format='png', dpi=300)  # Save the plot
# plt.show()

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np

# # List of audio files for each tone (replace with actual paths)
# for j in range(4):
#     audio_files = [
#         f"./resources/sounds/bu{j+1}_FV1_MP3.mp3",
#         f"./resources/sounds/di{j+1}_FV1_MP3.mp3",
#         f"./resources/sounds/lu{j+1}_FV1_MP3.mp3",
#         f"./resources/sounds/ma{j+1}_FV1_MP3.mp3",
#     ]
#     tone_titles = ["Bu", "Di", "Lu", "Ma"]

#     # Set up a 2x2 subplot grid
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle("Pitch Contours of Mandarin Tones", fontsize=16)

#     # Loop through each audio file and plot the pitch contour
#     for i, audio_path in enumerate(audio_files):
#         # Load audio and extract pitch
#         y, sr = librosa.load(audio_path)
#         f0, voiced_flag, voiced_probs = librosa.pyin(
#             y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
#         )
#         times = librosa.times_like(f0)

#         # Determine the current subplot location
#         ax = axs[i // 2, i % 2]

#         # Plot the pitch contour
#         ax.plot(times, f0, color="b", linewidth=2)
#         ax.set_title(tone_titles[i])
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Frequency (Hz)")
#         ax.grid()

#     # Adjust layout and save the combined figure
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the main title
#     plt.savefig("mandarin_tones_pitch_contours.png", format="png", dpi=300)
#     plt.show()
# import librosa
# import matplotlib.pyplot as plt
# import numpy as np

# # Words and tone numbers
# words = ["bu", "di", "lu", "ma"]
# tones = [1, 2, 3, 4]
# speakers = {"FV": "Female", "MV": "Male"}
# colors = {"FV": "red", "MV": "blue"}

# # Set up a 4x4 grid of subplots for the words and tones
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# fig.suptitle("Pitch Contours of Mandarin Tones by Word and Speaker", fontsize=18)

# # Loop through each word and tone
# for row, word in enumerate(words):
#     for col, tone in enumerate(tones):
#         ax = axs[row, col]  # Get the subplot for the current word and tone
#         for gender_code, gender in speakers.items():
#             for speaker_id in range(1, 4):  # Loop through 3 speakers per gender
#                 # Construct the file path
#                 audio_path = f"./resources/sounds/{word}{tone}_{gender_code}{speaker_id}_MP3.mp3"
                
#                 # Load audio and extract pitch
#                 y, sr = librosa.load(audio_path)
#                 f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
#                 times = librosa.times_like(f0)
                
#                 # Plot the pitch contour
#                 ax.plot(times, f0, label=f"{gender} Speaker {speaker_id}", color=colors[gender_code], linewidth=1)

#         # Set titles and labels for subplots
#         ax.set_title(f"{word.capitalize()}, Tone {tone}")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Frequency (Hz)")
#         ax.grid()

# # Adding a single legend for the whole figure
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper right", fontsize=12)

# # Adjust layout and save the combined figure
# plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust for the main title and legend
# plt.savefig("mandarin_words_tones_pitch_contours.png", format="png", dpi=300)
# plt.show()



# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Create "plots" directory if it doesn't exist
# os.makedirs("plots", exist_ok=True)

# # Words and tone numbers
# words = ["bu", "di", "lu", "ma"]
# tones = [1, 2, 3, 4]
# speakers = {"FV": "Female", "MV": "Male"}
# colors = {"FV": "red", "MV": "blue"}

# # Set up a 4x4 grid of subplots for the words and tones
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# fig.suptitle("Pitch Contours of Mandarin Tones by Word and Speaker", fontsize=18)

# # Loop through each word and tone
# for row, word in enumerate(words):
#     for col, tone in enumerate(tones):
#         ax = axs[row, col]  # Get the subplot for the current word and tone
#         for gender_code, gender in speakers.items():
#             for speaker_id in range(1, 4):  # Loop through 3 speakers per gender
#                 # Construct the file path
#                 audio_path = f"./resources/sounds/{word}{tone}_{gender_code}{speaker_id}_MP3.mp3"
                
#                 # Load audio and extract pitch
#                 y, sr = librosa.load(audio_path)
#                 f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
#                 times = librosa.times_like(f0)

#                 # Trim data to start from 200 ms and end at 600 ms
#                 start_idx = np.searchsorted(times, 0.2)  # Find index closest to 200 ms
#                 end_idx = np.searchsorted(times, 0.6)    # Find index closest to 600 ms
#                 trimmed_times = times[start_idx:end_idx]
#                 trimmed_f0 = f0[start_idx:end_idx]
                
#                 # Plot the pitch contour
#                 ax.plot(trimmed_times, trimmed_f0, label=f"{gender} Speaker {speaker_id}", color=colors[gender_code], linewidth=1)

#         # Set titles and labels for subplots
#         ax.set_title(f"{word.capitalize()}, Tone {tone}")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Frequency (Hz)")
#         ax.grid()

# # Adding a single legend for the whole figure
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper right", fontsize=12)

# # Adjust layout and save the combined figure
# plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust for the main title and legend
# output_path = os.path.join("./plots/mandarin_words_tones_pitch_contours2.png")
# plt.savefig(output_path, format="png", dpi=300)
# plt.show()

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Create "plots" directory if it doesn't exist
# os.makedirs("plots", exist_ok=True)

# # Words and tone numbers
# words = ["bu", "di", "lu", "ma"]
# tones = [1, 2, 3, 4]
# speakers = {"FV": "Female", "MV": "Male"}
# colors = {"FV": "red", "MV": "blue"}

# # Set up a 4x4 grid of subplots for the words and tones
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# fig.suptitle("Pitch Contours of Mandarin Tones by Word and Speaker", fontsize=18)

# # Loop through each word and tone
# for row, word in enumerate(words):
#     for col, tone in enumerate(tones):
#         ax = axs[row, col]  # Get the subplot for the current word and tone
#         for gender_code, gender in speakers.items():
#             for speaker_id in range(1, 4):  # Loop through 3 speakers per gender
#                 # Construct the file path
#                 audio_path = f"./resources/sounds/{word}{tone}_{gender_code}{speaker_id}_MP3.mp3"
                
#                 # Load audio and extract pitch
#                 y, sr = librosa.load(audio_path)
#                 f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
#                 times = librosa.times_like(f0)

#                 # Trim data to start from 200 ms and end at 600 ms
#                 start_idx = np.searchsorted(times, 0.2)  # Find index closest to 200 ms
#                 end_idx = np.searchsorted(times, 0.6)    # Find index closest to 600 ms
#                 trimmed_times = times[start_idx:end_idx]
#                 trimmed_f0 = f0[start_idx:end_idx]
                
#                 # Check if trimmed_f0 has any non-NaN values before interpolating
#                 if np.isnan(trimmed_f0).all():
#                     # Skip plotting if all values are NaN
#                     continue
#                 else:
#                     # Replace NaN values with interpolated values
#                     trimmed_f0 = np.interp(trimmed_times, trimmed_times[~np.isnan(trimmed_f0)], trimmed_f0[~np.isnan(trimmed_f0)])

#                 # Plot the pitch contour
#                 ax.plot(trimmed_times, trimmed_f0, label=f"{gender} Speaker {speaker_id}", color=colors[gender_code], linewidth=1)

#         # Set titles and labels for subplots
#         ax.set_title(f"{word.capitalize()}, Tone {tone}")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Frequency (Hz)")
#         ax.grid()

# # Adding a single legend for the whole figure
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper right", fontsize=12)

# # Adjust layout and save the combined figure
# plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust for the main title and legend
# output_path = os.path.join("/Volumes/gurindapalli/projects/tone-training-app/src/pitch_plot/plots", "mandarin_words_tones_pitch_contours.png")
# plt.savefig(output_path, format="png", dpi=300)
# plt.show()

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Create "plots" directory if it doesn't exist
# os.makedirs("plots", exist_ok=True)

# # Words and tone numbers
# words = ["bu", "di", "lu", "ma"]
# tones = [1, 2, 3, 4]
# speakers = {"FV": "Female", "MV": "Male"}
# colors = {"FV": "red", "MV": "blue"}

# # Set up a 4x4 grid of subplots for the words and tones
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# fig.suptitle("Pitch Contours of Mandarin Tones by Word and Speaker (Normalized)", fontsize=18)

# # Loop through each word and tone
# for row, word in enumerate(words):
#     for col, tone in enumerate(tones):
#         ax = axs[row, col]  # Get the subplot for the current word and tone
#         for gender_code, gender in speakers.items():
#             for speaker_id in range(1, 4):  # Loop through 3 speakers per gender
#                 # Construct the file path
#                 audio_path = f"./resources/sounds/{word}{tone}_{gender_code}{speaker_id}_MP3.mp3"
                
#                 # Load audio and extract pitch
#                 y, sr = librosa.load(audio_path)
#                 f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
#                 times = librosa.times_like(f0)

#                 # Trim data to start from 200 ms and end at 600 ms
#                 start_idx = np.searchsorted(times, 0.2)  # Find index closest to 200 ms
#                 end_idx = np.searchsorted(times, 0.6)    # Find index closest to 600 ms
#                 trimmed_times = times[start_idx:end_idx]
#                 trimmed_f0 = f0[start_idx:end_idx]
                
#                 # Check if trimmed_f0 has any non-NaN values before proceeding
#                 if np.isnan(trimmed_f0).all():
#                     # Skip plotting if all values are NaN
#                     continue
#                 else:
#                     # Replace NaN values with interpolated values
#                     trimmed_f0 = np.interp(trimmed_times, trimmed_times[~np.isnan(trimmed_f0)], trimmed_f0[~np.isnan(trimmed_f0)])
                
#                 # Normalize the pitch values using min-max normalization
#                 min_f0 = np.min(trimmed_f0)
#                 max_f0 = np.max(trimmed_f0)
#                 normalized_f0 = (trimmed_f0 - min_f0) / (max_f0 - min_f0) if max_f0 > min_f0 else trimmed_f0  # Avoid division by zero
                
#                 # Plot the normalized pitch contour
#                 ax.plot(trimmed_times, normalized_f0, label=f"{gender} Speaker {speaker_id}", color=colors[gender_code], linewidth=1)

#         # Set titles and labels for subplots
#         ax.set_title(f"{word.capitalize()}, Tone {tone}")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Normalized Frequency (0-1)")
#         ax.grid()

# # Adding a single legend for the whole figure
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper right", fontsize=12)

# # Adjust layout and save the combined figure
# plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust for the main title and legend
# output_path = os.path.join("/Volumes/gurindapalli/projects/tone-training-app/src/pitch_plot/plots", "mandarin_words_tones_pitch_contours_normalized.png")
# plt.savefig(output_path, format="png", dpi=300)
# plt.show()

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

# Create "plots" directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Words and tone numbers
words = ["bu", "di", "lu", "ma"]
tones = [1, 2, 3, 4]
speakers = {"FV": "Female", "MV": "Male"}
colors = {"FV": "red", "MV": "blue"}

# Set up a 4x4 grid of subplots for the words and tones
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle("Pitch Contours of Mandarin Tones by Word and Speaker (Normalized)", fontsize=18)

# Loop through each word and tone
for row, word in enumerate(words):
    for col, tone in enumerate(tones):
        ax = axs[row, col]  # Get the subplot for the current word and tone
        for gender_code, gender in speakers.items():
            for speaker_id in range(1, 4):  # Loop through 3 speakers per gender
                # Construct the file path
                audio_path = f"./resources/sounds/{word}{tone}_{gender_code}{speaker_id}_MP3.mp3"
                
                # Load audio and extract pitch
                y, sr = librosa.load(audio_path)
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
                times = librosa.times_like(f0)

                # Check if f0 has any non-NaN values before proceeding
                if np.isnan(f0).all():
                    # Skip plotting if all values are NaN
                    continue
                else:
                    # Replace NaN values with interpolated values for continuity
                    f0 = np.interp(times, times[~np.isnan(f0)], f0[~np.isnan(f0)])
                
                # Normalize the pitch values using min-max normalization
                min_f0 = np.min(f0)
                max_f0 = np.max(f0)
                normalized_f0 = (f0 - min_f0) / (max_f0 - min_f0) if max_f0 > min_f0 else f0  # Avoid division by zero
                
                # Plot the normalized pitch contour
                ax.plot(times, normalized_f0, label=f"{gender} Speaker {speaker_id}", color=colors[gender_code], linewidth=1)

        # Set titles and labels for subplots
        ax.set_title(f"{word.capitalize()}, Tone {tone}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Frequency (0-1)")
        ax.grid()

# Adding a single legend for the whole figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=12)

# Adjust layout and save the combined figure
plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust for the main title and legend
output_path = os.path.join("/Volumes/gurindapalli/projects/tone-training-app/src/pitch_plot/plots", "mandarin_words_tones_pitch_contours_normalized3.png")
plt.savefig(output_path, format="png", dpi=300)
plt.show()