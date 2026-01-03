import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from src.data import build_noisy_features, load_dataset
from src.utils import TEST_FOLDS, EVAL_NOISE_LEVELS, BASELINE_MODEL_PATH, AUGMENTED_MODEL_PATH

def evaluate_under_noise(model, clips, folds, noise_levels):
    results = {}
    for nl in noise_levels:
        X_noisy, y_noisy = build_noisy_features(clips, folds, nl)
        y_pred = model.predict(X_noisy)
        acc = accuracy_score(y_noisy, y_pred)
        results[nl] = acc
        print(f"Noise level {nl}: accuracy = {acc:.4f}")
    return results

clips = load_dataset()

clean_model = joblib.load(BASELINE_MODEL_PATH)
augmented_model = joblib.load(AUGMENTED_MODEL_PATH)

clean_results = evaluate_under_noise(clean_model, clips, TEST_FOLDS, EVAL_NOISE_LEVELS)
augmented_results = evaluate_under_noise(augmented_model, clips, TEST_FOLDS, EVAL_NOISE_LEVELS)

plt.plot(clean_results.keys(), clean_results.values(), marker="o", label="Clean-Trained")
plt.plot(augmented_results.keys(), augmented_results.values(), marker="s", label="Noise-Augmented")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.title("SVM Robustness to Acoustic Noise")
plt.legend()
plt.grid(True)
plt.savefig("/figures/svm_noise_aug_comparison.png")
plt.close()