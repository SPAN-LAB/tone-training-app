import os
import joblib
import numpy as np
import pandas as pd
import re
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import librosa

def extract_pitch_features(audio, sr, f0):
    """Extract meaningful features from pitch contour"""
    features = []
    
    # remove zeros from padding
    f0_clean = f0[f0 > 0]
    
    if len(f0_clean) > 0:
        features.extend([
            np.mean(f0_clean),           # Average pitch
            np.std(f0_clean),            # Pitch variability
            np.median(f0_clean),         # Robust central tendency
            np.max(f0_clean),            # Maximum pitch
            np.min(f0_clean),            # Minimum pitch
            np.ptp(f0_clean),            # Pitch range (max - min)
        ])
    else:
        # If no voiced frames, use zeros
        features.extend([0, 0, 0, 0, 0, 0])
    
    # Additional features: quartiles
    if len(f0_clean) > 3:
        features.extend(np.percentile(f0_clean, [25, 50, 75, 90]).tolist())
    else:
        features.extend([0, 0, 0, 0])
    
    return np.array(features)

def read_audio_files_yin(folder, samples_per_class=2250):
    """Load files and extract pitch FEATURES instead of raw contours"""
    directory = "/Volumes/gurindapalli/projects/Plasticity_training/tones/tone_perfect"
    all_files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

    # Extract tone number label from filename
    files_with_labels = []
    for f in all_files:
        m = re.search(r'(\d+)', f)
        if m:
            tone_num = int(m.group(1))
            files_with_labels.append((f, tone_num))

    # Group files by tone label
    from collections import defaultdict
    files_by_label = defaultdict(list)
    for f, label in files_with_labels:
        files_by_label[label].append(f)

    # Stratified sampling
    selected_files = []
    for tone in range(1, 5):
        files = files_by_label.get(tone, [])
        if len(files) >= samples_per_class:
            selected_files.extend(random.sample(files, samples_per_class))
        else:
            selected_files.extend(files)

    # extract pitch features
    feature_data = []
    labels = []

    for f in selected_files:
        path = os.path.join(directory, f)
        print("Loading:", path, "Size:", os.path.getsize(path))
        audio, sr = librosa.load(os.path.join(directory, f), sr=None, mono=True)

        # Extract pitch contour
        f0 = librosa.yin(audio, fmin=60, fmax=500, sr=sr)
        f0 = np.nan_to_num(f0)  # handle undefined frames
        
        # Extract features from contour instead of using raw data
        features = extract_pitch_features(audio, sr, f0)
        feature_data.append(features)

        label = int(re.search(r'(\d+)', f).group(1)) - 1  # zero-based label
        labels.append(label)

    return np.array(feature_data, dtype=np.float32), np.array(labels, dtype=int)

def main():
    folder = "tone_perfect"
    X, y = read_audio_files_yin(folder, samples_per_class=2250)
    print(f"audio files read")

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
    )

    # normalization - fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    best_accuracy = 0
    best_gamma = None

    for gamma in [0.001, 0.01, 0.1, 1, 10]:
        svm = SVC(kernel='rbf', C=1, gamma=gamma, random_state=42)
        svm.fit(X_train_scaled, y_train)

        val_pred = svm.predict(X_val_scaled)
        acc = accuracy_score(y_val, val_pred)

        print(f"Validation Accuracy (gamma={gamma}): {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_gamma = gamma

    print(f"\nBest gamma: {best_gamma}")

    # Final model on combined train+val
    X_trainval_scaled = scaler.transform(X_trainval)  # Use the scaler fitted on X_train
    svm_final = SVC(kernel='rbf', C=1, gamma=best_gamma, random_state=42)
    svm_final.fit(X_trainval_scaled, y_trainval)

    test_pred = svm_final.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.title(f"Confusion Matrix (Test Acc: {test_accuracy:.4f})")
    plt.xlabel("Predicted Tone")
    plt.ylabel("True Tone")
    plt.tight_layout()
    plt.savefig("confusion_matrix_yin_improved.png")
    plt.show()

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        svm_final, X_trainval_scaled, y_trainval, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()

    joblib.dump(svm_final, 'tone_pred_model.pkl')


if __name__ == "__main__":
    main()