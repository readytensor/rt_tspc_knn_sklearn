import os
import warnings

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TSAnnotationSchema
from preprocessing.custom_transformers import PADDING_VALUE
from typing import Tuple

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TSAnnotator:
    """KNN Timeseries Annotator.

    This class provides a consistent interface that can be used with other
    TSAnnotator models.
    """

    MODEL_NAME = "KNN_Timeseries_Annotator"

    def __init__(
        self,
        feat_dim:int,
        n_classes:int,
        n_neighbors: int = 7,
        **kwargs,
    ):
        """
        Construct a new KNN TSAnnotator.

        Args:
            feat_dim (int): Number of features.
            n_classes (int): Number of target classes.
            n_neighbors (int): Number of neighbors to use.
        """
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> KNeighborsClassifier:
        """Build a new KNN regressor."""
        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=n_jobs,
        )
        return model

    def _get_X_and_y(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        window_ids = data[:, :, :2]
        X = data[:, :, 2:self.feat_dim + 2].reshape(N, -1)
        if D == self.feat_dim + 3:
            y = data[:, :, -1].astype(int)
        else:
            y = None
        return X, y, window_ids

    def fit(self, train_data):
        train_X, train_y, window_ids = self._get_X_and_y(train_data)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, data):
        X, y, window_ids = self._get_X_and_y(data)
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
            if k[1] != PADDING_VALUE
        }

        sorted_dict = {key: prob_dict[key] for key in sorted(prob_dict.keys())}
        probabilities = np.vstack(list(sorted_dict.values()))
        return probabilities

    def evaluate(self, test_data, truth_labels):
        """Evaluate the model and return the metric"""
        predictions = self.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(truth_labels, predictions, average="weighted")
        return f1

    def save(self, model_dir_path: str) -> None:
        """Save the KNN TSAnnotator to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TSAnnotator":
        """Load the KNN TSAnnotator from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TSAnnotator: A new instance of the loaded KNN TSAnnotator.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TSAnnotationSchema,
    hyperparameters: dict,
) -> TSAnnotator:
    """
    Instantiate and train the TSAnnotator model.

    Args:
        train_data (np.ndarray): The train split from training data.
        hyperparameters (dict): Hyperparameters for the TSAnnotator.

    Returns:
        'TSAnnotator': The TSAnnotator model
    """
    model = TSAnnotator(
        feat_dim=len(data_schema.features),
        n_classes=len(data_schema.target_classes),
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TSAnnotator, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TSAnnotator, predictor_dir_path: str) -> None:
    """
    Save the TSAnnotator model to disk.

    Args:
        model (TSAnnotator): The TSAnnotator model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TSAnnotator:
    """
    Load the TSAnnotator model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TSAnnotator: A new instance of the loaded TSAnnotator model.
    """
    return TSAnnotator.load(predictor_dir_path)


def evaluate_predictor_model(model: TSAnnotator, test_split: np.ndarray, truth_labels: np.ndarray) -> float:
    """
    Evaluate the TSAnnotator model and return the r-squared value.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_split (np.ndarray): Test data.
        truth_labels (np.ndarray): The truth labels.

    Returns:
        float: The r-squared value of the TSAnnotator model.
    """
    return model.evaluate(test_split, truth_labels)
