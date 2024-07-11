import os
import warnings

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TimeStepClassificationSchema
from typing import Tuple

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TimeStepClassifier:
    """KNeighbors TimeStepClassifier.

    This class provides a consistent interface that can be used with other
    TimeStepClassifier models.
    """

    MODEL_NAME = "KNeighbors_TimeStepClassifier"

    def __init__(
        self,
        n_classes: int,
        encode_len: int,
        padding_value: float,
        n_neighbors: int = 5,
        **kwargs,
    ):
        """
        Construct a new KNeighbors TimeStepClassifier.

        Args:
            n_classes (int): Number of target classes.
            encode_len (int): Encoding (history) length.
            padding_value (float): Padding value.
            n_neighbors (int): Number of neighbors to use.
        """
        self.n_classes = n_classes
        self.encode_len = int(encode_len)
        self.padding_value = padding_value
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> KNeighborsClassifier:
        """Build a new KNeighbors classifier."""
        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=n_jobs,
            **self.kwargs,
        )
        return model

    def _get_X_and_y(
        self, data: np.ndarray, is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        if is_train:
            if T != self.encode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len}"
                    f" length on axis 1. Found length {T}"
                )
            # we excluded the first 2 dimensions (id, time) and the last dimension (target)
            X = data[:, :, 2:-1].reshape(N, -1)  # shape = [N, T*D]
            y = data[:, :, -1].astype(int)  # shape = [N, T]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, :, 2:].reshape(N, -1)
            y = data[:, :, 0:2]
        return X, y

    def fit(self, train_data):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, data):
        X, window_ids = self._get_X_and_y(data, is_train=False)
        preds = self.model.predict_proba(X)
        for i, pred in enumerate(preds):
            if pred.shape[1] > self.n_classes:
                preds[i] = pred[:, :-1]
        preds = np.array(preds)
        preds = preds.transpose(1, 0, 2)

        prob_dict = {}

        for index, prediction in enumerate(preds):
            series_id = window_ids[index][0][0]
            for step_index, step in enumerate(prediction):
                step_id = window_ids[index][step_index][1]
                step_id = (series_id, step_id)
                prob_dict[step_id] = prob_dict.get(step_id, []) + [step]

        prob_dict = {
            k: np.mean(np.array(v), axis=0)
            for k, v in prob_dict.items()
            if k[1] != self.padding_value
        }

        sorted_dict = {key: prob_dict[key] for key in sorted(prob_dict.keys())}
        probabilities = np.vstack(list(sorted_dict.values()))
        return probabilities

    def evaluate(self, test_data, truth_labels):
        """Evaluate the model and return the loss and metrics"""
        predictions = self.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(truth_labels, predictions, average="weighted")
        return f1

    def save(self, model_dir_path: str) -> None:
        """Save the KNeighborsTimeStepClassifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TimeStepClassifier":
        """Load the KNeighborsTimeStepClassifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TimeStepClassifier: A new instance of the loaded KNeighborsTimeStepClassifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TimeStepClassificationSchema,
    hyperparameters: dict,
    padding_value: float,
) -> TimeStepClassifier:
    """
    Instantiate and train the TimeStepClassifier model.

    Args:
        train_data (np.ndarray): The train split from training data.
        data_schema (TimeStepClassificationSchema): The data schema.
        hyperparameters (dict): Hyperparameters for the TimeStepClassifier.
        padding_value (float): The padding value.

    Returns:
        'TimeStepClassifier': The TimeStepClassifier model
    """
    model = TimeStepClassifier(
        n_classes=len(data_schema.target_classes),
        padding_value=padding_value,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TimeStepClassifier, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_data (np.ndarray): The test input data for classification.

    Returns:
        np.ndarray: The predictions.
    """
    return model.predict(test_data)


def save_predictor_model(model: TimeStepClassifier, predictor_dir_path: str) -> None:
    """
    Save the TimeStepClassifier model to disk.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TimeStepClassifier:
    """
    Load the TimeStepClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TimeStepClassifier: A new instance of the loaded TimeStepClassifier model.
    """
    return TimeStepClassifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: TimeStepClassifier, test_split: np.ndarray, truth_labels: np.ndarray
) -> float:
    """
    Evaluate the TimeStepClassifier model and return the r-squared value.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_split (np.ndarray): Test data.
        truth_labels (np.ndarray): The true labels.

    Returns:
        float: The r-squared value of the TimeStepClassifier model.
    """
    return model.evaluate(test_split, truth_labels)
