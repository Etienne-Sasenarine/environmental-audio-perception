import os

TRAIN_FOLDS = {1, 2, 3, 4, 5, 6, 7, 8}
TEST_FOLDS = {9, 10}

CACHE_DIR = "cache"

X_TRAIN_PATH = os.path.join(CACHE_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(CACHE_DIR, "y_train.npy")
X_TEST_PATH  = os.path.join(CACHE_DIR, "X_test.npy")
Y_TEST_PATH  = os.path.join(CACHE_DIR, "y_test.npy")

X_AUG_PATH = os.path.join(CACHE_DIR, "X_train_aug.npy")
Y_AUG_PATH = os.path.join(CACHE_DIR, "y_train_aug.npy")

BASELINE_MODEL_PATH = os.path.join(CACHE_DIR, "svm_baseline.joblib")
AUGMENTED_MODEL_PATH = os.path.join(CACHE_DIR, "svm_augmented.joblib")

CLASS_NAMES = [
    "Air Conditioner",
    "Car Horn",
    "Children Playing",
    "Dog Bark",
    "Drilling",
    "Engine Idling",
    "Gun Shot",
    "Jackhammer",
    "Siren",
    "Street Music"
]

EVAL_NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]
TRAIN_NOISE_LEVELS = [0.005, 0.01, 0.02]
