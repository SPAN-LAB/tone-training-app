import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import joblib
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Extra wide pitch bounds for manadarin tones specifically*
GENDER_PITCH_BOUNDS = {
    "male": (70, 200), # 80, 160
    "female": (100, 300) #160, 260
}


@dataclass
class AudioExample:
    pitch_contour: np.ndarray  # f0 values (Zero padding)
    label: int  # 0-based indexing
    speaker_id: str
    gender: str


def extract_speaker_id(filename: str) -> str:
    """
    Best-effort speaker ID extraction from filename.
    """
    stem = Path(filename).stem
    tokens = re.split(r"[_\-]+", stem)
    for token in tokens:
        if token and not token.isdigit():
            return token.lower()
    return "unknown"


def load_speaker_metadata(metadata_path: Path) -> Dict[str, str]:
    """
    Optional CSV metadata with columns: speaker_id, gender
    """
    if not metadata_path.exists():
        return {}

    lookup: Dict[str, str] = {}
    with metadata_path.open() as f:
        for line in f:
            if not line.strip() or "speaker" in line.lower():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            speaker, gender = parts[0].lower(), parts[1].lower()
            lookup[speaker] = gender
    return lookup


def pitch_bounds_for_gender(gender: str) -> Tuple[float, float]:
    if gender in GENDER_PITCH_BOUNDS:
        return GENDER_PITCH_BOUNDS[gender]
    return 70.0, 250.0 #80, 260


def interpolate_out_of_bounds_pitch(
    f0: np.ndarray, fmin: float, fmax: float
) -> np.ndarray:
    """
    Replace out-of-range voiced frames via monotone interpolation.
    """
    if f0.size == 0:
        return f0

    voiced_mask = f0 > 0
    valid_mask = voiced_mask & (f0 >= fmin) & (f0 <= fmax)
    if valid_mask.all():
        return f0

    f0_clean = f0.astype(np.float32, copy=True)
    valid_idx = np.nonzero(valid_mask)[0]
    if valid_idx.size == 0:
        f0_clean[voiced_mask] = np.clip(f0_clean[voiced_mask], fmin, fmax)
        return f0_clean

    if valid_idx.size == 1:
        fill_value = np.clip(f0_clean[valid_idx[0]], fmin, fmax)
        f0_clean[voiced_mask & ~valid_mask] = fill_value
        return f0_clean

    interpolator = PchipInterpolator(valid_idx, f0_clean[valid_idx], extrapolate=True)
    interpolated = interpolator(np.arange(f0_clean.shape[0]))

    replace_mask = voiced_mask & ~valid_mask
    f0_clean[replace_mask] = np.clip(interpolated[replace_mask], fmin, fmax)
    return f0_clean


def read_audio_files_yin(
    folder: str,
    samples_per_class: int = 2250,
    metadata_path: Optional[Path] = None,
) -> List[AudioExample]:
    """
    Load tone_perfect audio files and extract YIN pitch contours.
    """
    default_dir = Path(
        os.environ.get(
            "TONE_PERFECT_DIR", "/Volumes/gurindapalli/projects/Plasticity_training/tones/tone_perfect"
        )
    )
    user_dir = Path(folder)
    directory = user_dir if user_dir.exists() else default_dir
    speaker_metadata = load_speaker_metadata(metadata_path) if metadata_path else {}

    all_files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

    files_with_labels: List[Tuple[str, int]] = []
    for f in all_files:
        m = re.search(r"(\d+)", f)
        if m:
            tone_num = int(m.group(1))
            files_with_labels.append((f, tone_num))

    from collections import defaultdict

    files_by_label = defaultdict(list)
    for f, label in files_with_labels:
        files_by_label[label].append(f)

    selected_files: List[str] = []
    for tone in range(1, 5):
        files = files_by_label.get(tone, [])
        if len(files) >= samples_per_class:
            selected_files.extend(random.sample(files, samples_per_class))
        else:
            selected_files.extend(files)

    examples: List[AudioExample] = []
    for f in selected_files:
        path = directory / f
        speaker_id = extract_speaker_id(f)
        gender = speaker_metadata.get(speaker_id, "unknown")
        fmin, fmax = pitch_bounds_for_gender(gender)

        audio, sr = librosa.load(path, sr=None, mono=True)
        f0 = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr)
        f0 = np.nan_to_num(f0, nan=0.0)
        f0 = interpolate_out_of_bounds_pitch(f0, fmin, fmax)

        label = int(re.search(r"(\d+)", f).group(1)) - 1  # zero-based label
        examples.append(AudioExample(pitch_contour=f0.astype(np.float32), label=label, speaker_id=speaker_id, gender=gender))

    return examples


def split_examples(
    examples: List[AudioExample], test_size: float = 0.2, val_size: float = 0.2
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample]]:
    labels = [ex.label for ex in examples]
    train_val, test = train_test_split(
        examples, test_size=test_size, random_state=SEED, stratify=labels
    )

    train_labels = [ex.label for ex in train_val]
    adjusted_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val_size, random_state=SEED, stratify=train_labels
    )
    return train, val, test



def normalize_and_pad(
    examples: List[AudioExample],
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score normalize PER CLIP (Instance Normalization) and pad sequences.
    This removes speaker pitch bias (deep voice vs high voice).
    """
    normalized_sequences: List[np.ndarray] = []
    labels: List[int] = []

    for ex in examples:
        seq = ex.pitch_contour.astype(np.float32)
        mask = seq > 0 # only voiced frames
        
        norm_seq = np.zeros_like(seq)
        if np.any(mask):
            # calculate based on audio clip stats, NOT speaker/global stats
            clip_mean = np.mean(seq[mask])
            clip_std = np.std(seq[mask])
            
            # Prevent div by zero
            if clip_std < 1e-6:
                clip_std = 1e-6
                
            norm_seq[mask] = (seq[mask] - clip_mean) / clip_std

        normalized_sequences.append(norm_seq)
        labels.append(ex.label)

    padded = pad_sequences(
        normalized_sequences, maxlen=max_length, padding="post", dtype="float32", truncating="post"
    )
    # Bi-LSTM expects feature dimension
    padded = np.expand_dims(padded, axis=-1)
    return padded, np.array(labels, dtype=np.int32)


def build_bilstm_model(max_length: int, num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(max_length, 1)),
            layers.Bidirectional(layers.LSTM(96, return_sequences=True)), 
            layers.Bidirectional(layers.LSTM(96)),
            layers.Dense(96, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    data_dir = Path(
        os.environ.get(
            "TONE_PERFECT_DIR", "/Volumes/gurindapalli/projects/Plasticity_training/tones/tone_perfect"
        )
    )
    metadata_path = data_dir / "speaker_metadata.csv"
    metadata_arg = metadata_path if metadata_path.exists() else None

    print("Loading audio and extracting pitch contours with YIN...")
    examples = read_audio_files_yin(
        str(data_dir), samples_per_class=2250, metadata_path=metadata_arg
    )
    print(f"Loaded {len(examples)} examples")

    train_examples, val_examples, test_examples = split_examples(examples)
    print(
        f"Split -> train: {len(train_examples)}, val: {len(val_examples)}, test: {len(test_examples)}"
    )


    
    train_lengths = [len(ex.pitch_contour) for ex in train_examples]
    max_length = max(train_lengths) if train_lengths else 0
    if max_length == 0:
        raise ValueError("No training data available after loading.")


    X_train, y_train = normalize_and_pad(train_examples, max_length)
    X_val, y_val = normalize_and_pad(val_examples, max_length)
    X_test, y_test = normalize_and_pad(test_examples, max_length)

    model = build_bilstm_model(max_length=max_length, num_classes=4)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    batch_size = 64

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_metrics[0]:.4f} | Test accuracy: {test_metrics[1]:.4f}")

    output_dir = Path(".")
    model.save(output_dir / "tone_bilstm_model.keras")
    

    normalization_bundle = {
        "model_path": str(output_dir / "tone_bilstm_model"),
        "max_length": max_length,
        "normalization_type": "instance_level" # informative flag
    }
    joblib.dump(normalization_bundle, output_dir / "tone_bilstm.pkl")
    np.save(output_dir / "bilstm_training_history.npy", history.history)
    print("Saved model and instance-normalization bundle.")

    # Confusion Matrix, Training & Validation Progress
    y_pred = np.argmax(model.predict(X_test, batch_size=64, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4'],
                yticklabels=['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4'])
    plt.title(f'Confusion Matrix (Accuracy: {test_metrics[1]:.4f})')
    plt.xlabel('Predicted Tone')
    plt.ylabel('True Tone')
    plt.tight_layout()
    plt.savefig("confusion_matrix_pitch_contours.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()