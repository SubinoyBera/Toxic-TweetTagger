# Utility functions for the model inference api

import yaml
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import xgboost as xgb
from lime.lime_text import LimeTextExplainer

# load yaml files to get model meta data.
try:
    with open(Path("model/registered_model_meta"), 'r') as f:
        model_metadata = yaml.safe_load(f) 
except:
    raise FileNotFoundError("Failed to load file having model metadata")


# Intialize lime explainer with class names
_global_explainer = LimeTextExplainer(class_names=["hate", "non-hate"], bow=False)


class LimeExplainer:
    def __init__(self, vectorizer: Any, model: Any):
        """
        Initializes an instance of LimeExplainer.

        Sets the class names for the explainer and initializes the LimeTextExplainer.
        Also initializes the model prediction attribute to None.
        """
        self.explainer = _global_explainer
        self.prediction = None
        self.model = model
        self.vectorizer = vectorizer

    def _get_prediction(self, tweet) -> np.ndarray:
        """
        Internal function to get prediction from the model and class probability scores 
        for lime explainer.
        """
        X = self.vectorizer.transform([tweet])
        pred = self.model.predict(X)
        input_df = pd.DataFrame({
            "comments": tweet
        })
        self.prediction = self.model.predict(context=None, model_input=input_df)
        return np.array(self.prediction["class_probability_scores"])
    
    def explain(self, tweet):
        """
        Generate lime explanation for a given tweet.

        Parameters
            tweet: str : Input tweet or comment to be classified.

        Returns
            dict : A dictionary with words as keys and their corresponding weightage.
        """
        explanation = self.explainer.explain_instance(
            tweet,
            self._get_prediction,
            num_features=10,
            num_samples=20
        )
        html_content = explanation.as_html()
        
        return html_content
        

def load_model_artifacts():
    """Loads ML model from location path and returns the model."""
    try:
        with open(Path("model/artifacts/vectorizer.joblib"), 'rb') as f:
            vectorizer = joblib.load(f)

        xbg_booster = xgb.Booster()
        xbg_booster.load_model(Path("model/artifacts/booster.json"))

        with open(Path("model/artifacts/metrics.json"), 'r') as f:
            metrics = json.load(f)
            eval_threshold = metrics.get("threshold", 0.5)

        return vectorizer,  xbg_booster, eval_threshold
    
    except FileNotFoundError:
        raise FileNotFoundError("Model artifacts not found in the directory.")


def get_model_registry() -> str:
    """Fetches the model registry name and returns it."""
    model_registry = model_metadata['model_name']
    return model_registry


def get_model_version() -> int:
    """Fetches the model version and returns it."""
    if not model_metadata:
        return 0
    else:
        return int(model_metadata.get("model_version", 0))