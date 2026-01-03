import numpy as np
import librosa
import noisereduce as nr

def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    width = max(width, 3)
    delta = librosa.feature.delta(mfcc, width=width)
    delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    mfcc_features = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1)
    ])
    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=audio)
    spec_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_bands=6, fmin=50.0
    )
    spec_features = np.concatenate([
        np.mean(spec_cent, axis=1), np.std(spec_cent, axis=1),
        np.mean(spec_bw, axis=1), np.std(spec_bw, axis=1),
        np.mean(spec_rolloff, axis=1), np.std(spec_rolloff, axis=1),
        np.mean(spec_flat, axis=1), np.std(spec_flat, axis=1),
        np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1)
    ])
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_features = np.concatenate([
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1)
    ])
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_features = np.array([np.mean(zcr), np.std(zcr)])
    # RMS energy
    rms = librosa.feature.rms(y=audio)
    rms_features = np.array([np.mean(rms), np.std(rms)])
    return np.concatenate([
        mfcc_features,
        spec_features,
        chroma_features,
        zcr_features,
        rms_features
    ])