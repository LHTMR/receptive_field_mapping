from src.post_processing.validation import Validation as Val
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
        Val.validate_type(log_file, str, "Log File")
        Val.validate_path(log_file, file_types=[".json"])

        self.best_models = {}
        self.log_file = log_file

    @staticmethod
    def transform_to_derivative(df: pd.DataFrame) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        if df.shape[1] % 2 != 0:
            raise ValueError("The DataFrame must have an even number of columns (x and y pairs).")

        derivative_df = df.diff().abs()
        derivative_df.iloc[0, :] = 0
        return derivative_df

    def detect_outliers_velocity(self, df: pd.DataFrame, threshold: int | float = 2.0) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(threshold, (int, float), "Threshold")
        Val.validate_positive(threshold, "Threshold")

        df_velocity = self.transform_to_derivative(df.copy())
        mean, std = df_velocity.mean(), df_velocity.std()
        outlier_mask = (df_velocity < (mean - threshold * std)) | (df_velocity > (mean + threshold * std))
        outlier_mask.iloc[0, :] = False
        outlier_mask |= (df_velocity > 50)
        df[outlier_mask] = np.nan
        return df

    def _grid_search_models_per_col(self, df: pd.DataFrame, model_name: str = None) -> None:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        if model_name:
            Val.validate_type(model_name, str, "Model Name")
            if model_name not in self.models:
                raise ValueError(f"Invalid model name '{model_name}'. Available models: {list(self.models.keys())}")

        self.best_models = {}
        for target_col in df.columns:
            train_df = df.dropna(subset=[target_col]).dropna(how="any")
            if train_df.empty:
                self.best_models[target_col] = None
                continue

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            models_to_try = {model_name: self.models[model_name]} if model_name else self.models
            best_model, best_score = None, float("inf")

            for name, model in models_to_try.items():
                param_grid = self.param_grids.get(name, {})
                try:
                    grid = GridSearchCV(model,
                                        param_grid,
                                        scoring="neg_mean_squared_error",
                                        cv=3)
                    grid.fit(X_train, y_train)
                    mse = -grid.best_score_
                    if mse < best_score:
                        best_score, best_model = mse, grid.best_estimator_
                except:
                    continue

            self.best_models[target_col] = best_model if best_model else None

        if not any(self.best_models.values()):
            for col in df.columns:
                self.best_models[col] = [RandomForestRegressor(n_estimators=100),
                                         HistGradientBoostingRegressor(),
                                         KNeighborsRegressor(n_neighbors=10)]

    def iterative_imputation(self, df: pd.DataFrame, max_iter=100) -> pd.DataFrame:
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
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(std_threshold, (int, float), "STD Threshold")
        Val.validate_positive(std_threshold, "STD Threshold")

        df_copy = self.detect_outliers_velocity(df.copy(), std_threshold)
        self._grid_search_models_per_col(df_copy, model_name=model_name)
        df_copy = self.iterative_imputation(df_copy)

        with open(self.log_file, "w") as f:
            json.dump({col: str(model) for col, model in self.best_models.items()}, f, indent=4)

        return df_copy
