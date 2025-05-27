"""
Experiment Classification Model and Supporting Functions
"""
from typing import Collection
from scipy.optimize import curve_fit
import numpy as np
import json


def load_model_config(config_file: str) -> dict:
    """
    Load model configuration from a JSON file.
    View ExperimentClassifier for expected structure.

    Args:
    - config_file (str): Path to the JSON configuration file.

    Returns:
    - dict: Parsed configuration dictionary.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def format_input(data: str) -> tuple[list[str], np.ndarray]:
    """
    Validate input data. Then,
    Format input data into experiment name and time series.

    Args:
    - data (str): Input data string in the format "experiment_name: time_series".
    Returns:
    - experiment_names: np.array 1D
    - time_series: np.array 2D
    """
    lines = data.split("\n")
    names = lines[0].split(",")

    time_series = []
    for line in lines[1:]:
        if line.strip():
            time_series.append(list(map(float, line.strip().split(","))))

    time_series = np.array(time_series).T

    if time_series.shape[1] != 100:
        raise ValueError("Each time series must have exactly 100 observations.")
    if time_series.shape[0] != len(names):
        raise ValueError("Number of time series must match number of experiment names.")
    return names, time_series


def format_output(names: list[str], results: list[bool]) -> str:
    """
    Format output data into a string.

    Args:
    - names: List of experiment names.
    - results: List of classification results.

    Returns:
    - str: Formatted output string.
    """
    return f"""{','.join(names)}
{','.join(['perfect' if res else 'imperfect' for res in results])}"""


class ExperimentClassifier:
    def __init__(self, model_configs):
        """
        Args:
        - model_parameters (dict): A dictionary containing the model parameters 
            for classification. Example structure shown below.

        Example:
        {
            "validation": {
                "min-lower-bound": 0.0,
                "min-upper-bound": 0.5,
                "max-lower-bound": 2.0,
                "max-upper-bound": 14.0,
                "rmse-upper-bound": 5.0,
            },
            "5PL": {
                "cparam-lower-bound": 35.0,
                "cparam-upper-bound": 85.0,
                "rmse-threshold": 0.018,
            }
        }
        """
        self.val_configs = model_configs["validation"]
        self.pred_configs = model_configs["prediction"]
        self.time_axis = np.arange(100)

    @staticmethod
    def logistic_5pl(x, a, b, c, d, g):
        return d + (a - d) / ((1 + (x / c)**b)**g)

    def __call__(self, time_series: Collection[float]) -> bool:
        """
        Classify the time series as perfect or imperfect based on the model parameters.
        Raise error if validation fails.

        Args:
        - time_series: A time series of observations of size 100.
        Returns:
        - bool: True if the time series is classified as perfect, False if imperfect.
        """
        ### data validations
        min_val = min(time_series)
        max_val = max(time_series)
        
        if (min_val < self.val_configs["min-lower-bound"]):
            raise ValueError(f"Minimum value {min_val} must be greater than or equal to {self.val_configs['min-lower-bound']}.")
        if (min_val > self.val_configs["min-upper-bound"]):
            raise ValueError(f"Minimum value {min_val} must be less than {self.val_configs['min-upper-bound']}.")
        if (max_val < self.val_configs["max-lower-bound"]):
            raise ValueError(f"Maximum value {max_val} must be greater than {self.val_configs['max-lower-bound']}.")
        if (max_val > self.val_configs["max-upper-bound"]):
            raise ValueError(f"Maximum value {max_val} must be less than {self.val_configs['max-upper-bound']}.")
        
        popt, _ = curve_fit(self.logistic_5pl, self.time_axis, time_series, maxfev=10000)
        fitted_values = self.logistic_5pl(self.time_axis, *popt)
        rmse = np.sqrt(np.mean((fitted_values - time_series) ** 2))

        if rmse > self.val_configs["rmse-upper-bound"]:
            raise ValueError(f"Curve fitting error. RMSE {rmse} exceeds the upper bound {self.val_configs['rmse-upper-bound']}. ")
        
        ### classify using cparam
        if (popt[2] < self.pred_configs["cparam-lower-bound"] or 
            popt[2] > self.pred_configs["cparam-upper-bound"]):
            return False
        
        ### classify using rmse
        return rmse < self.pred_configs["rmse-threshold"]
