import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_and_predict(random_state: int = 42):
    X_train, X_test, y_train, y_test = load_data(random_state=random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return preds, y_test


def get_accuracy(random_state: int = 42) -> float:
    preds, y_test = train_and_predict(random_state=random_state)
    return accuracy_score(y_test, preds)


def train_model(random_state: int = 42):
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)

    return model, iris.target_names


def predict_from_features(features):
    model, target_names = train_model()

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]

    return {
        "predicted_class_index": int(prediction),
        "predicted_class_name": target_names[prediction]
    }