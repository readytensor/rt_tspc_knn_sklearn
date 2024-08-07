from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects or drops specified columns."""

    def __init__(self, columns: List[str], selector_type: str = "keep"):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the column selection.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """
        if self.selector_type == "keep":
            retained_cols = [col for col in self.columns if col in X.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == "drop":
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        return X


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns using label encoding."""

    def __init__(self, columns: List[str]):
        """
        Initializes a new instance of the `LabelEncoder` class.

        Args:
            columns : list of str
                List of column names to encode.
        """
        self.columns = columns
        self.encoders = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the label encoders for the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            self
        """
        for col in self.columns:
            if col in X.columns:
                self.encoders[col] = {
                    label: idx for idx, label in enumerate(sorted(X[col].unique()))
                }
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the label encoding to the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """

        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].map(self.encoders[col])
        return X

    def inverse_transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Applies the inverse of the label encoding to the specified columns.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The inverse transformed data.
        """
        X = X.copy()
        for col in self.columns:
            inv_encoders = {v: k for k, v in self.encoders[col].items()}
            X[col] = X[col].map(inv_encoders)
        return X


class TypeCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the specified variables in the input data
    to a specified data type.
    """

    def __init__(self, vars: List[str], cast_type: str):
        """
        Initializes a new instance of the `TypeCaster` class.

        Args:
            vars : list of str
                List of variable names to be transformed.
            cast_type : data type
                Data type to which the specified variables will be cast.
        """
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns]
        for var in applied_cols:
            if data[var].notnull().any():  # check if the column has any non-null values
                data[var] = data[var].astype(self.cast_type)
            else:
                # all values are null. so no-op
                pass
        return data


class TimeColCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the time col in the input data
    to either a datetime type or the integer type, given its type
    in the schema.
    """

    def __init__(self, time_col: str, data_type: str):
        """
        Initializes a new instance of the `TimeColCaster` class.

        Args:
            time_col (str): Name of the time field.
            cast_type (str): Data type to which the specified variables
                             will be cast.
        """
        super().__init__()
        self.time_col = time_col
        self.data_type = data_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        if self.data_type == "INT":
            data[self.time_col] = data[self.time_col].astype(int)
        elif self.data_type in ["DATETIME", "DATE"]:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
        else:
            raise ValueError(f"Invalid data type for time column: {self.data_type}")
        return data


class DataFrameSorter(BaseEstimator, TransformerMixin):
    """
    Sorts a pandas DataFrame based on specified columns and their corresponding sort orders.
    """

    def __init__(self, sort_columns: List[str], ascending: List[bool]):
        """
        Initializes a new instance of the `DataFrameSorter` class.

        Args:
            sort_columns : list of str
                List of column names to sort by.
            ascending : list of bool
                List of boolean values corresponding to each column in `sort_columns`.
                Each value indicates whether to sort the corresponding column in ascending order.
        """
        assert len(sort_columns) == len(
            ascending
        ), "sort_columns and ascending must be of the same length"
        self.sort_columns = sort_columns
        self.ascending = ascending

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Sorts the DataFrame based on specified columns and order.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The sorted DataFrame.
        """
        X = X.sort_values(by=self.sort_columns, ascending=self.ascending)
        return X


class PaddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_col: str, target_col: str, padding_value: float) -> None:
        super().__init__()
        self.id_col = id_col
        self.target_col = target_col
        self.padding_value = padding_value

    def fit(self, X, y=None):
        self.max_length = X.groupby(self.id_col).size().max()
        return self

    def transform(self, X):
        padded_X = None
        grouped = X.groupby(self.id_col)
        if len(grouped) == 1:
            return X
        for group_id, group_data in grouped:

            padding_length = self.max_length - group_data.shape[0]
            padding_df = pd.DataFrame(
                {
                    col: [self.padding_value] * padding_length
                    for col in group_data.columns
                }
            )
            padding_df[self.id_col] = group_id

            if padded_X is None:
                padded_X = pd.concat([group_data, padding_df], ignore_index=True)
            else:
                padded_X = pd.concat(
                    [padded_X, group_data, padding_df], ignore_index=True
                )

        if self.target_col in padded_X.columns:
            labels = X[self.target_col].unique().tolist()
            num_padding_entries = padded_X[
                padded_X[self.target_col] == self.padding_value
            ].shape[0]
            random_labels = np.random.choice(labels, num_padding_entries)
            padded_X.loc[
                padded_X[self.target_col] == self.padding_value, self.target_col
            ] = random_labels
            label_dtype = X[self.target_col].dtype
            padded_X[self.target_col] = padded_X[self.target_col].astype(label_dtype)

        return padded_X


class ReshaperToThreeD(BaseEstimator, TransformerMixin):
    def __init__(self, id_col, time_col, value_columns, target_column=None) -> None:
        super().__init__()
        self.id_col = id_col
        self.time_col = time_col
        self.id_columns = [self.id_col, self.time_col]
        if not isinstance(value_columns, list):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns
        # ensure id and time columns are included to be able to join the windows during inference
        self.cols_to_reshape = [id_col, time_col] + self.value_columns
        self.target_column = target_column
        self.id_vals = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.id_vals = X[[self.id_col]].drop_duplicates().sort_values(by=self.id_col)
        self.id_vals.reset_index(inplace=True, drop=True)
        reshaped_columns = [c for c in self.cols_to_reshape if c in X.columns]
        if self.target_column in X.columns:
            reshaped_columns.append(self.target_column)

        value_counts = X[[self.id_col]].value_counts()

        # Check if all value counts are the same
        if not value_counts.nunique() == 1:
            raise ValueError(
                "The counts are not the same for all ids. " "Did you pad the data?"
            )
        T = value_counts.iloc[0]
        N = self.id_vals.shape[0]
        D = len(reshaped_columns)

        X = X[reshaped_columns].values.reshape((N, T, D))
        return X

    def inverse_transform(self, preds_df):

        time_cols = list(preds_df.columns)
        preds_df = pd.concat([self.id_vals, preds_df], axis=1, ignore_index=True)

        cols = self.id_columns + time_cols
        preds_df.columns = cols

        # unpivot given dataframe
        preds_df = pd.melt(
            preds_df,
            id_vars=self.id_columns,
            value_vars=time_cols,
            var_name=self.time_col,
            value_name="prediction",
        )

        return preds_df


class TimeSeriesWindowGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer for generating windows from time-series data.
    """

    def __init__(
        self,
        window_size: int,
        padding_value: float,
        stride: int = 1,
        max_windows: int = None,
        mode: str = "train",
    ):
        """
        Initializes the TimeSeriesWindowGenerator.

        Args:
            window_size (int): The size of each window (W).
            padding_value (float): The value to use for padding.
            stride (int): The stride between each window.
            max_windows (int): The maximum number of windows to generate.
            mode (str): The mode of the transformer. Must be either 'train' or 'inference'.
        """
        self.window_size = window_size
        self.padding_value = padding_value
        self.stride = stride
        self.max_windows = max_windows
        self.mode = mode

    def fit(self, X, y=None):
        """
        No-op. This transformer does not require fitting.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transforms the input time-series data into windows using vectorized operations.

        Args:
            X (numpy.ndarray): Input time-series data of shape [N, T, D].

        Returns:
            numpy.ndarray: Transformed data of shape [N', W, D] where N' is the number of windows.
        """
        n_series, time_length, n_features = X.shape

        # Validate window size and stride
        if self.window_size > time_length:
            print(
                "Window size must be less than or equal to the time dimension length. \n"
                f"Given window size {self.window_size} will be trimmed to length {time_length - 1}."
            )
            self.window_size = time_length - 1

        # Calculate the total number of windows per series
        n_windows_per_series = 1 + (time_length - self.window_size) // self.stride

        # Create an array of starting indices for each window
        start_indices = np.arange(0, n_windows_per_series * self.stride, self.stride)
        if (
            self.mode == "inference"
            and start_indices[-1] + self.window_size < time_length
        ):
            # Add an additional window that extends to the end of the series
            last_window_index = time_length - self.window_size
            start_indices = np.append(start_indices, last_window_index)

        # Use broadcasting to generate window indices
        window_indices = start_indices[:, None] + np.arange(self.window_size)

        # Generate windows for each series using advanced indexing
        windows = X[:, window_indices, :]
        windows = windows.reshape(-1, self.window_size, n_features)

        # Remove windows that are full of padding values
        all_padding = windows[:, :, 1] == self.padding_value
        all_padding = all_padding.all(axis=1)

        windows = windows[~all_padding]

        # Limit the number of windows if specified by randomly sampling
        if self.max_windows is not None and len(windows) > self.max_windows:
            indices = np.random.choice(len(windows), self.max_windows, replace=False)
            windows = windows[indices]
        return windows


class SeriesLengthTrimmer(BaseEstimator, TransformerMixin):
    """Transformer that trims the length of each series in the dataset.

    This transformer retains only the latest data points along the time dimension
    up to the specified length.

    Attributes:
        trimmed_len (int): The length to which each series should be trimmed.
    """

    def __init__(self, trimmed_len: int):
        """
        Initializes the SeriesLengthTrimmer.

        Args:
            trimmed_len (int): The length to which each series should be trimmed.
        """
        self.trimmed_len = trimmed_len

    def fit(self, X: np.ndarray, y: None = None) -> "SeriesLengthTrimmer":
        """Fit method for the transformer.

        This transformer does not learn anything from the data and hence fit is a no-op.

        Args:
            X (np.ndarray): The input data.
            y (None): Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            SeriesLengthTrimmer: The fitted transformer.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Trims each series in the input data to the specified length.

        Args:
            X (np.ndarray): The input time-series data of shape [N, T, D], where N is the number of series,
                            T is the time length, and D is the number of features.

        Returns:
            np.ndarray: The transformed data with each series trimmed to the specified length.
        """
        _, time_length, _ = X.shape
        if time_length > self.trimmed_len:
            X = X[:, -self.trimmed_len :, :]
        return X


class TimeSeriesMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales the history and forecast parts of a time-series based on history data.

    The scaler is fitted using only the history part of the time-series and is
    then used to transform both the history and forecast parts. Values are scaled
    to a range and capped to an upper bound.

    Attributes:
        encode_len (int): The length of the history (encoding) window in the time-series.
        upper_bound (float): The upper bound to which values are capped after scaling.
    """

    def __init__(self, columns: List = []):
        """
        Initializes the TimeSeriesMinMaxScaler.

        Args:
            columns (List): The columns to scale.
        """
        self.scaler = MinMaxScaler()
        self.columns = columns
        self.fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "TimeSeriesMinMaxScaler":
        """
        No-op

        Args:
            X (pd.DataFrame): The input dataframe.
            y: Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            TimeSeriesMinMaxScaler: The fitted scaler.
        """
        if self.fitted:
            return self
        X_scaled = X.copy()[self.columns]
        self.scaler.fit(X_scaled)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the MinMax scaling transformation to the input data.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X_scaled = X.copy()
        X_scaled.loc[:, self.columns] = self.scaler.transform(
            X_scaled.loc[:, self.columns]
        )
        return X_scaled
