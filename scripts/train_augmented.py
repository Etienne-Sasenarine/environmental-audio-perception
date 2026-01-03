import os
import joblib
import numpy as np

from src.data import load_dataset, build_augmented_train_set
from src.models import build_fixed_svm
from src.utils import (
    TRAIN_FOLDS, TRAIN_NOISE_LEVELS,
    CACHE_DIR, X_AUG_PATH, 
    Y_AUG_PATH, AUGMENTED_MODEL_PATH
)

clips = load_dataset()

if os.path.exists(X_AUG_PATH) and os.path.exists(Y_AUG_PATH):
    X_train_aug = np.load(X_AUG_PATH)
    y_train_aug = np.load(Y_AUG_PATH)
else:
    X_train_aug, y_train_aug = build_augmented_train_set(clips, TRAIN_FOLDS, TRAIN_NOISE_LEVELS)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.save(X_AUG_PATH, X_train_aug)
    np.save(Y_AUG_PATH, y_train_aug)

if os.path.exists(AUGMENTED_MODEL_PATH):
    model_aug = joblib.load(AUGMENTED_MODEL_PATH)
else:
    # Using hyperparameters found from baseline training
    best_C = 10
    best_gamma = 0.01
    model_aug = build_fixed_svm(best_C, best_gamma)
    model_aug.fit(X_train_aug, y_train_aug)
    joblib.dump(model_aug, AUGMENTED_MODEL_PATH)