from src.components.validation import Validation as Val
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import json
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class OutlierImputer:
    """
    A class for detecting and imputing outliers in time series data using velocity-based detection
    and iterative model-based imputation.

    This class provides methods to:
        - Detect outliers in paired (x, y) coordinate data using velocity thresholds.
        - Select the best regression model for each column using grid search.
        - Impute missing/outlier values using iterative imputation.
        - Log model performance and selections to a JSON file.

    Attributes:
        models (dict): Dictionary of available regression models for imputation.
        param_grids (dict): Dictionary of hyperparameter grids for each model.
        best_models (dict): Stores the best model selected for each column after grid search.
        log_file (str): Path to the JSON file where model performance is logged.

    Args:
        log_file (str, optional): Path to the JSON file for logging model performance. Defaults to "model_performance.json".

    Raises:
        TypeError: If log_file is not a string.
        ValueError: If log_file does not have a .json extension.
    """
    models = {
        "RFR": RandomForestRegressor(n_estimators=100, random_state=42),
        "HGBR": HistGradientBoostingRegressor(),
        "KNR": KNeighborsRegressor(n_neighbors=10),
        "XGB": XGBRegressor(n_estimators=100, learning_rate=0.1),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        "BR": BayesianRidge(),
        "Poly": make_pipeline(PolynomialFeatures(degree=2), BayesianRidge())
    }

    param_grids = {
        "RFR": {"n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]},
        "HGBR": {"max_iter": [100, 200],
                 "learning_rate": [0.05, 0.1, 0.2]},
        "KNR": {"n_neighbors": [5, 10, 15],
                "weights": ["uniform", "distance"]},
        "XGB": {"n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.2]},
        "SVR": {"C": [1, 10, 100],
                "gamma": ["scale", 0.1, 1],
                "epsilon": [0.01, 0.1, 0.2]},
        "BR": {"alpha_1": [1e-6, 1e-5, 1e-4],
               "lambda_1": [1e-6, 1e-5, 1e-4]},
        "Poly": {"bayesianridge__alpha_1": [1e-6, 1e-5],
                 "polynomialfeatures__degree": [2]}
    }

    def __init__(self, log_file="model_performance.json") -> None:
        """
        Initialize the OutlierImputer.

        Args:
            log_file (str, optional): Path to the JSON file for logging model performance. Defaults to "model_performance.json".

        Raises:
            TypeError: If log_file is not a string.
            ValueError: If log_file does not have a .json extension.
        """
        Val.validate_type(log_file, str, "Log File")
        Val.validate_path(log_file, file_types=[".json"])

        self.best_models = {}
        self.log_file = log_file

    @staticmethod
    def transform_to_derivative(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the absolute derivative (velocity) of each column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with numeric columns (must have even number of columns).

        Returns:
            pd.DataFrame: DataFrame of absolute differences (velocity) with the first row set to 0.

        Raises:
            TypeError: If df is not a DataFrame or contains non-numeric data.
            ValueError: If the DataFrame does not have an even number of columns.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_dataframe_numeric(df, "DataFrame")
        if df.shape[1] % 2 != 0:
            raise ValueError(
                "The DataFrame must have an even number of columns (x and y pairs).")

        derivative_df = df.diff().abs()
        derivative_df.iloc[0, :] = 0 # Set the first row to 0
        return derivative_df

    def detect_outliers_velocity(self, df: pd.DataFrame, threshold: int | float = 2.0) -> pd.DataFrame:
        """
        Detect and remove outliers in the DataFrame based on velocity thresholds.

        Outliers are detected where the velocity exceeds mean ± threshold * std or is above an absolute threshold.

        Args:
            df (pd.DataFrame): Input DataFrame with numeric columns.
            threshold (int | float, optional): Number of standard deviations for outlier detection. Defaults to 2.0.

        Returns:
            pd.DataFrame: DataFrame with outliers replaced by NaN.

        Raises:
            TypeError: If df is not a DataFrame or threshold is not numeric.
            ValueError: If threshold is not positive.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(threshold, (int, float), "Threshold")
        Val.validate_positive(threshold, "Threshold")

        df_velocity = self.transform_to_derivative(df.copy())
        mean, std = df_velocity.mean(), df_velocity.std()
        outlier_mask = (df_velocity < (mean - threshold * std)
                        ) | (df_velocity > (mean + threshold * std))
        outlier_mask.iloc[0, :] = False
        outlier_mask |= (df_velocity > 50)
        df[outlier_mask] = np.nan
        return df

    def _grid_search_models_per_col(self, df: pd.DataFrame, model_name: str = None) -> None:
        """
        Perform grid search to select the best regression model for each column.

        For each column, fits all candidate models (or a specified model) using grid search and stores the best model.

        Args:
            df (pd.DataFrame): DataFrame with missing values to impute.
            model_name (str, optional): Name of a specific model to use. If None, tries all models.

        Returns:
            None

        Raises:
            TypeError: If df is not a DataFrame or model_name is not a string.
            ValueError: If model_name is not in the list of available models.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        if model_name:
            Val.validate_type(model_name, str, "Model Name")
            if model_name not in self.models:
                raise ValueError(
                    f"Invalid model name '{model_name}'. Available models: {list(self.models.keys())}")

        self.best_models = {}
        for target_col in df.columns:
            train_df = df.dropna(subset=[target_col]).dropna(how="any")
            if train_df.empty:
                self.best_models[target_col] = None
                continue

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            models_to_try = {
                model_name: self.models[model_name]} if model_name else self.models
            best_model, best_score = None, float("inf")

            for name, model in models_to_try.items():
                param_grid = self.param_grids.get(name, {})

                grid = GridSearchCV(model,
                                    param_grid,
                                    scoring="neg_mean_squared_error",
                                    cv=3)
                grid.fit(X_train, y_train)
                mse = -grid.best_score_
                if mse < best_score:
                    best_score, best_model = mse, grid.best_estimator_

            self.best_models[target_col] = best_model

    def iterative_imputation(self, df: pd.DataFrame, max_iter=1000) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame using iterative model-based imputation.

        Uses the best model(s) found by grid search to iteratively impute missing values.

        Args:
            df (pd.DataFrame): DataFrame with missing values to impute.
            max_iter (int, optional): Maximum number of imputation iterations. Defaults to 1000.

        Returns:
            pd.DataFrame: DataFrame with imputed values.

        Raises:
            TypeError: If df is not a DataFrame or max_iter is not an integer.
            ValueError: If max_iter is not positive.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(max_iter, int, "Max Iterations")
        Val.validate_positive(max_iter, "Max Iterations")

        df_copy = df.copy()
        estimators = [model for model in self.best_models.values() if model]
        estimator = estimators[0] if estimators else RandomForestRegressor()

        imputer = IterativeImputer(estimator=estimator,
                                   max_iter=max_iter,
                                   random_state=101)
        imputed_array = imputer.fit_transform(df_copy)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)

    def impute_outliers(self, df: pd.DataFrame, std_threshold: int | float = 2.0, model_name: str = None) -> pd.DataFrame:
        """
        Detect outliers, select the best model(s), and impute missing/outlier values.

        This method:
            1. Detects and removes outliers using velocity-based thresholding.
            2. Runs grid search to select the best model(s) for each column.
            3. Imputes missing values using iterative imputation.
            4. Logs the selected models to a JSON file.

        Args:
            df (pd.DataFrame): Input DataFrame with numeric columns.
            std_threshold (int | float, optional): Number of standard deviations for outlier detection. Defaults to 2.0.
            model_name (str, optional): Name of a specific model to use for imputation. If None, tries all models.

        Returns:
            pd.DataFrame: DataFrame with outliers imputed.

        Raises:
            TypeError: If df is not a DataFrame, std_threshold is not numeric, or model_name is not a string.
            ValueError: If std_threshold is not positive or model_name is invalid.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(std_threshold, (int, float), "STD Threshold")
        Val.validate_positive(std_threshold, "STD Threshold")

        df_copy = self.detect_outliers_velocity(df.copy(), std_threshold)
        self._grid_search_models_per_col(df_copy, model_name=model_name)
        df_copy = self.iterative_imputation(df_copy)

        with open(self.log_file, "w") as f:
            json.dump({col: str(model)
                      for col, model in self.best_models.items()}, f, indent=4)

        return df_copy
