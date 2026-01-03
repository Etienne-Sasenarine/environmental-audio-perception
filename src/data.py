import os
import numpy as np
import soundata
import librosa

from src.features import extract_features

def load_dataset() -> dict:
    dataset = soundata.initialize("urbansound8k")
    return dataset.load_clips()

def load_features(clips: dict, folds: list) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for clip in clips.values():
        if clip.fold not in folds:
            continue
        audio, sr = librosa.load(clip.audio_path, sr=None)
        features = extract_features(audio, sr)
        X.append(features)
        y.append(clip.class_id)
    return np.array(X), np.array(y)

def add_noise(audio: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def build_noisy_features(clips: dict, folds: list, noise_level: float) -> tuple[np.ndarray, np.ndarray]:
    X_noisy, y = [], []
    for clip in clips.values():
        if clip.fold not in folds:
            continue
        audio, sr = librosa.load(clip.audio_path, sr=None)
        noisy_audio = add_noise(audio, noise_level)
        features = extract_features(noisy_audio, sr)
        X_noisy.append(features)
        y.append(clip.class_id)
    return np.array(X_noisy), np.array(y)

def build_augmented_train_set(clips: dict, folds: list, noise_levels: list) -> tuple[np.ndarray, np.ndarray]:
    X_aug, y_aug = [], []
    for clip in clips.values():
        if clip.fold not in folds:
            continue
        audio, sr = librosa.load(clip.audio_path, sr=None)
        clean_features = extract_features(audio, sr)
        X_aug.append(clean_features)
        y_aug.append(clip.class_id)
        for nl in noise_levels:
            noisy_audio = add_noise(audio, nl)
            noisy_features = extract_features(noisy_audio, sr)
            X_aug.append(noisy_features)
            y_aug.append(clip.class_id)
    return np.array(X_aug), np.array(y_aug)