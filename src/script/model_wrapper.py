# This script wraps the ML model and the vectorizer into a single Python model object and 
# prepares it for registeration in MLflow. 
# This implementation is deprecated and is not used in the current version of the API (V2)

import pandas as pd
from mlflow.pyfunc.model import PythonModel

# Custom model class to wrap the ML model and vectorizer
class CustomModel(PythonModel):
    def __init__(self, model, vectorizer):
        """
        Initializes the CustomModel instance with a machine learning model and a vectorizer.

        Args:
            model: The machine learning model to be used for prediction.
            vectorizer: The vectorizer to transform input data for the model.
        """
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, context, model_input: pd.DataFrame):
        
        """
        Predicts the class probability scores and class labels for the given input data.

        Args:
            context (dict): Context containing additional information that may be useful for prediction.
            model_input (pd.DataFrame): Input data containing the text column.
        
        Returns:
            dict: A dictionary containing the class probability scores and class labels.
        """
        texts = model_input["comments"]
        if self.vectorizer is not None and self.model is not None:
            X = self.vectorizer.transform(texts)
            class_probability_scores = self.model.predict_proba(X)
            class_label = self.model.predict(X)

        return {
            "class_probability_scores" : class_probability_scores,
            "class_label" : class_label
        }