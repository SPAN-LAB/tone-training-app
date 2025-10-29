#!C:\ProgramData\Anaconda3\python.exe python


import os
import numpy as np
import pandas as pd
import re
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import librosa

def read_audio_files(folder):
    # directory = os.path.join("Volumes", "gurindapalli", "projects", "Plasticity_training", "tones", folder)
    # directory = os.path.join("tones", folder) # use relative path tones
    # directory = "../tones/tone_perfect" 
    # /Volumes/gurindapalli/projects/Plasticity_training/tones/tone_perfect
    directory = "/Volumes/gurindapalli/projects/Plasticity_training/tones/tone_perfect"

    print(os.getcwd())
    print("All directories:")
    print(os.walk(directory))

    # mp3_files = [file for file in os.listdir(directory) if file.endswith('.mp3')]

    #randomly choose 8000 audio files
    files = os.listdir(directory)
    index = random.sample(range(len(files)), 9000)

    mp3_files = [files[i] for i in index if files[i].endswith('.mp3')]
    
    features = []
    labels = []
    
    max_length = None

    for file in mp3_files:

        # Load audio file
        data, sr = librosa.load(os.path.join(directory, file))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

        # Truncate or pad MFCCs to ensure consistent length
        if max_length is None:
            max_length = max([mfccs.shape[1] for mfccs in features]) if features else mfccs.shape[1]
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        
        features.append(mfccs)
        labels.append(re.search(r'\d', file)[0])
    
    return np.array(features), np.array(labels)

# Autoencoder Model
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Create autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    # Create encoder model for feature extraction
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder

def custom_shuffle(data, labels):
    assert len(data) == len(labels), "Data and labels must have the same length."
    
    # Create an array of indices
    indices = np.arange(len(data))
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Use the shuffled indices to reorder data and labels
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]
    
    return shuffled_data, shuffled_labels



def main():

    #randomly choose 8000 audio files
    folder = "tone_perfect"
    
    features, labels = read_audio_files(folder)
    print("files read")

    # Parameters for the autoencoder
    windowSize = 10
    input_dim = features.shape[1] * windowSize  # 13 * 10, Number of features per time step (MFCCs)
    encoding_dim = 64              # Compressed feature size

    # Create autoencoder and encoder models
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)

    slice_results = np.zeros(9)

    """
    Model Training
    """

    # Initialize global datasets
    global_encoded_data = []
    global_labels = []
    for i in range(len(labels)):  # Process all files
        currentData = []

        for windowStart in range(features.shape[2] - windowSize + 1):  # Sliding window
            currentWindow = []
            for fileIdx in range(windowStart, windowStart + windowSize):
                currentWindow.append(features[i, :, fileIdx])

            currentData.append(np.array(currentWindow))

        currentData = np.array(currentData)
        currentData_flat = currentData.reshape((currentData.shape[0], -1))  # Flatten for autoencoder

        # Train autoencoder
        autoencoder.fit(currentData_flat, currentData_flat, epochs=50, batch_size=32, verbose=0)

        # Encode data
        encoded_data = encoder.predict(currentData_flat).flatten()

        # Append to global datasets
        global_encoded_data.append(encoded_data)

        global_labels.append(labels[i])  # Extend with the current label

    global_encoded_data = np.array(global_encoded_data)
    inc_factor = 10

    global_encoded_data = np.repeat(global_encoded_data, repeats=inc_factor, axis=0)
    global_labels = np.repeat(global_labels, repeats=inc_factor, axis=0)

    global_encoded_data, global_labels = custom_shuffle(global_encoded_data, global_labels)   

    """Model Validation
    """
    
    # Initialize result storage
    numSlices = 10
    slice_results = np.zeros(numSlices)

    k = 5  # Number of folds for cross-validation

    for i in range(numSlices):  # You have 9 slices
        print(f"Slice: {i + 1}")

        # Filter data for the current slice (if applicable, otherwise use all data)
        X = global_encoded_data
        y = global_labels

        # Initialize storage for accuracies across windows
        accuracyResults = []

        # K-Fold Cross-Validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        bestAccuracy = 0
        bestHyperparameters = {}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"Processing Fold: {fold + 1}")

            # Split data into train and test
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Further split train into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            print(X_train.shape)

            # Skip if the training set contains a single class
            if len(np.unique(y_train)) <= 1:
                print(f"Skipping fold due to single class in y_train: {y_train}")
                continue

            # Hyperparameter search
            for kernelScale in [0.001, 0.01, 0.1, 1]:
                # Train SVM on encoded features
                svm = SVC(kernel='rbf', C=1, gamma=kernelScale)
                svm.fit(X_train, y_train)

                # Validate SVM
                val_predictions = svm.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)

                # # Plot confusion matrix for validation set
                # cm = confusion_matrix(y_val, val_predictions)
                # print(f"Confusion Matrix for Validation Set, Fold {fold + 1}, Validation Accuracy: {val_accuracy:.4f}")
                # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                # plt.show()

                # Track best hyperparameters
                if val_accuracy > bestAccuracy:
                    bestAccuracy = val_accuracy
                    bestHyperparameters['KernelScale'] = kernelScale

                print(f"Fold {fold + 1}, KernelScale: {kernelScale}, Validation Accuracy: {val_accuracy:.4f}")

            # Train final model on train + validation data
            svm_final = SVC(kernel='rbf', C=1, gamma=bestHyperparameters['KernelScale'])
            svm_final.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

            # Test final model
            test_predictions = svm_final.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            print(y_test, test_predictions)

            accuracyResults.append(test_accuracy)

            # # Plot confusion matrix for test set
            # cm = confusion_matrix(y_test, test_predictions)
            # print(f"Confusion Matrix for Test Set, Fold {fold + 1}, Test Accuracy: {test_accuracy:.4f}")
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.show()

        # Average accuracy across folds for the slice
        slice_avg_accuracy = np.mean(accuracyResults)
        print(f"Slice {i + 1} Average Accuracy: {slice_avg_accuracy:.4f}")
        slice_results[i] = slice_avg_accuracy

    # Display final results
    print("Final Results for All Slices:", slice_results)

    # Plot results
    plt.figure()
    plt.plot(range(1, 11), slice_results, '-o')  # Updated for 9 audio files
    plt.xlabel('Slice')
    plt.ylabel('Accuracy')
    plt.axhline(y=np.mean(slice_results), color='r', linestyle='--', label=f'Average Accuracy: {np.mean(slice_results):.4f}')
    plt.ylim([0, 1])
    plt.title('Model Accuracy across slices')


if __name__ == "__main__":
    main()