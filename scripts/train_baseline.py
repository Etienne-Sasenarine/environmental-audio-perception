import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from src.data import load_dataset, load_features
from src.models import train_svm
from src.utils import (
    TRAIN_FOLDS, TEST_FOLDS, X_TRAIN_PATH, 
    Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH,
    CACHE_DIR, BASELINE_MODEL_PATH, CLASS_NAMES
)

clips = load_dataset()

if os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH) and os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
else:
    X_train, y_train = load_features(clips, TRAIN_FOLDS)
    X_test, y_test = load_features(clips, TEST_FOLDS)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

if os.path.exists(BASELINE_MODEL_PATH):
    model = joblib.load(BASELINE_MODEL_PATH)
else:
    model, best_params = train_svm(X_train, y_train)
    joblib.dump(model, BASELINE_MODEL_PATH)
    print(f"Best hyperparameters: {best_params}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline Test Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(16, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Clean Audio")
plt.savefig("/figures/confusion_matrix_clean.png")
plt.close()