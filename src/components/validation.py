import numpy as np
import pandas as pd
import re


class Validation:
    @staticmethod
    def validate_path(path: str, file_types: list[str] = None):
        if not isinstance(path, str):
            raise ValueError(
                f"The path must be a string. Got {type(path)} instead.")
        if len(file_types) > 0:
            if not any(path.endswith(ext) for ext in file_types):
                raise ValueError(
                    f"The file must be one of {file_types}. Got '{path}' instead.")

    @staticmethod
    def validate_strings(**kwargs):
        for name, value in kwargs.items():
            Validation.validate_type(value, str, name)

    @staticmethod
    def validate_type(value, expected_types, name: str):
        if not isinstance(value, expected_types):
            raise TypeError(
                f"{name} must be one of {expected_types}. Got {type(value)} instead.")

    @staticmethod
    def validate_type_in_list(value, expected_types, name: str):
        if not isinstance(value, list) or not all(isinstance(v, expected_types) for v in value):
            raise ValueError(
                f"{name} must be a list of {expected_types}. Got {type(value)} instead.")

    @staticmethod
    def validate_positive(value, name: str, zero_allowed: bool = False):
        if zero_allowed and value < 0:
            raise ValueError(
                f"{name} must be a non-negative number. Got {value} instead.")
        elif not zero_allowed and value <= 0:
            raise ValueError(
                f"{name} must be a positive number. Got {value} instead.")

    @staticmethod
    def validate_float_in_range(value, min_value: float, max_value: float, name: str):
        if not isinstance(value, (float, int)):
            raise ValueError(
                f"{name} must be a float or int. Got {type(value)} instead.")
        if value < min_value or value > max_value:
            raise ValueError(
                f"{name} must be between {min_value} and {max_value}. Got {value} instead.")

    @staticmethod
    def validate_in_list(value, valid_list, name: str):
        if value not in valid_list:
            raise ValueError(
                f"{name} must be one of {valid_list}. Got '{value}' instead.")

    @staticmethod
    def validate_array(array, shape: tuple = None, name: str = "Array"):
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"{name} must be a numpy array. Got {type(array)} instead.")
        if shape and array.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}. Got {array.shape} instead.")

    @staticmethod
    def validate_array_int_float(array, shape: tuple = None, name: str = "Array"):
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"{name} must be a numpy array. Got {type(array)} instead.")
        if shape and array.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}. Got {array.shape} instead.")
        # raise error if the array is not of type int or float
        if array.dtype not in [int, np.float32, np.float64]:
            raise ValueError(
                f"{name} must have data type 'int'. Got '{array.dtype} instead.")

    @staticmethod
    def validate_list_int(list_values, shape: tuple = None, name: str = "List"):
        if not isinstance(list_values, list):
            raise ValueError(
                f"{name} must be a list. Got {type(list_values)} instead.")
        if shape and len(list_values) != shape[0]:
            raise ValueError(
                f"{name} must have length {shape[0]}. Got {len(list_values)} instead.")
        if not all(isinstance(value, int) for value in list_values):
            raise ValueError(f"All values in {name} must be integers.")

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None, name: str = "DataFrame", optional: bool = False):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"{name} must be a pandas DataFrame. Got {type(df)} instead.")

        column_mapping = {}  # To store matched columns for renaming
        if len(required_columns) > 0:
            missing_columns = []
            for group in required_columns:
                # If the group is a string, treat it as a single required column
                if isinstance(group, str):
                    group = [group]
                # Check if at least one pattern in the group matches any column
                matched_column = next((column for pattern in group for column in df.columns if re.search(
                    pattern, column, re.IGNORECASE)), None)
                if matched_column:
                    # Map the first pattern in the group to the matched column
                    column_mapping[group[0]] = matched_column
                else:
                    missing_columns.append(group)

            if missing_columns and not optional:
                raise ValueError(
                    f"{name} is missing required columns matching any of: {missing_columns}")
        return column_mapping
