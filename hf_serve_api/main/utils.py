# Utility functions for the model inference api

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from lime.lime_text import LimeTextExplainer

# load yaml files to get model meta data.
model_metadata: dict[str, Any]
try:
    with open(Path("model/registered_model_meta"), 'r') as f:
        model_metadata = yaml.safe_load(f) or {}
except:
    raise FileNotFoundError("Failed to load file having model metadata")


# Intialize lime explainer with class names
_global_explainer = LimeTextExplainer(class_names=["hate", "non-hate"], bow=False)


class LimeExplainer:
    def __init__(self, model: Any):
        """
        Initializes an instance of LimeExplainer.

        Sets the class names for the explainer and initializes the LimeTextExplainer.
        Also initializes the model prediction attribute to None.
        """
        self.explainer = _global_explainer
        self.prediction = None
        self.model = model

    def _get_prediction_explaination(self, tweet) -> np.ndarray:
        """
        Internal function to get prediction from the model and class probability scores 
        for lime explainer.
        """
        input_df = pd.DataFrame({
            "comments": tweet
        })
        self.prediction = self.model.predict(context=None, model_input=input_df)
        return np.array(self.prediction["class_probability_scores"])
    
    def explain(self, tweet) -> dict:
        """
        Generate lime explanation for a given tweet.

        Parameters
            tweet: str : Input tweet or comment to be classified.

        Returns
            dict : A dictionary with words as keys and their corresponding weightage.
        """
        explanation = self.explainer.explain_instance(
            tweet,
            self._get_prediction_explaination,
            num_features=5,
            num_samples=20
        )
        return round_dict_values(dic = dict(explanation.as_list()))


def load_model():
    """Loads ML model from location path and returns the model."""
    try:
        with open(Path("model/python_model.pkl"), "rb") as f:
            model = joblib.load(f)
        return model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model from hub: {e}")


def get_model_registry() -> str:
    """Fetches the model registry name and returns it."""
    model_registry = model_metadata['model_name']
    return model_registry


def get_model_version() -> str:
    """Fetches the model version and returns it."""
    model_version = model_metadata['model_version']
    return model_version


def round_dict_values(dic) -> dict:
    """Rounds all values in a dictionary to 4 decimal places."""
    return {str(k): round(v, 4) for k, v in dic.items()}