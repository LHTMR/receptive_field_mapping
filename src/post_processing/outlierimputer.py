import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from validation import Validation as Val

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

    def __init__(self,
                 log_file="model_performance.json") -> None:
        Val.validate_type(log_file, str, "Log File")
        Val.validate_path(log_file, file_types=[".json"])
        
        self.best_models = {}
        self.log_file = log_file

    @staticmethod
    def transform_to_derivative(df: pd.DataFrame) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        # Ensure the DataFrame has an even number of columns (x and y pairs)
        if df.shape[1] % 2 != 0:
            raise ValueError("The DataFrame must have an even number of columns (x and y pairs).")

        # Calculate the derivative (difference between consecutive rows)
        derivative_df = df.diff().abs()  # Use `.diff()` and drop the first row (NaN)
        derivative_df.iloc[0, :] = 0  # Set the first row to zero to avoid NaNs

        # Rename columns to indicate they are derivatives
        derivative_df.columns = [f"{col}_derivative" for col in df.columns]

        return derivative_df

    def detect_outliers_velocity(self,
                                 df: pd.DataFrame,
                                 threshold: int|float = 2.0) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(threshold, (int, float), "Threshold")
        Val.validate_positive(threshold, "Threshold")

        df_velocity = self.transform_to_derivative(df.copy())

        mean, std = df_velocity.mean(), df_velocity.std()
        outlier_mask = (df_velocity < (mean - threshold * std)) |\
                       (df_velocity > (mean + threshold * std))
        outlier_mask.iloc[0, :] = False  # First row should not be an outlier

        outlier_mask |= (df_velocity > 50)
        df[outlier_mask] = np.nan  # Mark outliers as NaN
        return df

    def _select_best_models_per_col(self,
                                    df: pd.DataFrame) -> None:
        Val.validate_type(df, pd.DataFrame, "DataFrame")

        self.best_models = {}
        feature_cols = df.columns

        for target_col in feature_cols:
            train_df = df.dropna(subset=[target_col])
            train_df = train_df.dropna(how="any")
            if train_df.empty:
                continue
            
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            best_model, best_score = None, float("inf")

            for model_name, model in self.models.items():
                param_grid = self.param_grids.get(model_name, {})
                try:
                    grid = GridSearchCV(model,
                                        param_grid,
                                        scoring="neg_mean_squared_error", cv=3)
                    grid.fit(X_train, y_train)
                    mse = -grid.best_score_

                    if mse < best_score:
                        best_score, best_model = mse, grid.best_estimator_
                except:
                    continue  # Skip model if it fails

            self.best_models[target_col] = best_model if best_model else None

        # Default models if no good models found
        if not any(self.best_models.values()):
            for col in df.columns:
                self.best_models[col] = [RandomForestRegressor(n_estimators=100),
                                         HistGradientBoostingRegressor(),
                                         KNeighborsRegressor(n_neighbors=10)]

    def iterative_imputation(self,
                             df: pd.DataFrame,
                             max_iter=100) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(max_iter, int, "Max Iterations")
        Val.validate_positive(max_iter, "Max Iterations")
        
        df_copy = df.copy()
        self._select_best_models_per_col(df_copy)

        estimators = [model for model in self.best_models.values() if model]
        estimator = estimators[0] if estimators else RandomForestRegressor()
        
        imputer = IterativeImputer(estimator=estimator,
                                   max_iter=max_iter,
                                   random_state=101)
        imputed_array = imputer.fit_transform(df_copy)

        df_imputed = pd.DataFrame(imputed_array,
                                  columns=df.columns,
                                  index=df.index)
        return df_imputed

    def impute_outliers(self,
                        df: pd.DataFrame,
                        std_threshold: int|float = 2.0) -> pd.DataFrame:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(std_threshold, (int, float), "STD Threshold")
        Val.validate_positive(std_threshold, "STD Threshold")

        df_copy = self.detect_outliers_velocity(df.copy(), std_threshold)
        df_copy = self.iterative_imputation(df_copy)

        with open(self.log_file, "w") as f:
            json.dump({col: str(model) for col, model in self.best_models.items()}, f, indent=4)

        return df_copy
