import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> tuple[SVC, dict]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1]
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def build_fixed_svm(C: float, gamma: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma))
    ])
