# This file contains tests for the model loading and performance validation.
import pytest
import os
import json
import numpy as np
import pandas as pd
import mlflow
import dagshub
from mlflow.client import MlflowClient


@pytest.fixture(scope="module")
def new_model():
    """
    Fixture to load the latest version of the 'ToxicTagger-Models' from the MLflow model registry.

    This fixture sets up the MLflow tracking environment using credentials from environment
    variables and retrieves the latest version of the specified model in the "Staging" stage.
    It returns the loaded model for use in tests.

    Returns:
        mlflow.pyfunc.PyFuncModel

    """
    uri = os.getenv("MLFOW_URI")
    repo_owner = os.getenv("OWNER")
    repo_name = os.getenv("REPO")
    
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Dagshub Token environment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

   #  For local use
   #  ==============================================================================
   # if repo_owner is None:
   #	 raise ValueError("Missing dagshub logging environment credentials.")
   #
   # dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)    
   #  ==============================================================================

    mlflow.set_tracking_uri(uri)            # type: ignore

    # load the new model from MLflow model registry
    model_name = "ToxicTagger-Models"
    client = MlflowClient()
    get_latest_version = client.get_latest_versions(name=model_name, stages=["Staging"])
    new_model_version = get_latest_version[0].version
    new_model_uri = f'models:/{model_name}/{new_model_version}'

    new_model = mlflow.pyfunc.load_model(new_model_uri)
    return new_model
    

def test_model_signature(new_model):
    """
    Tests the model's signature and output validity.
    This test creates a dummy input DataFrame mimicking expected input and uses the loaded model 
    to make a prediction. It checks that the model returns a valid class label (0 or 1) and a set 
    of probability scores. It also verifies that the sum of probability scores almost equals to 1.0.
   
    Assertions:
        - The predicted class label should be either 0 or 1.
        - Probability scores should not be None.
        - The sum of the probability scores should be approximately 1.0
          to ensure correct probability distribution.
    
    """
    input_text = ["the book is so bad, i hate it!"]
    input_df = pd.DataFrame(input_text, columns=["comments"])

    model_response = new_model.predict(input_df)

    label = int(model_response["class_label"][0])
    probability_scores = model_response["class_probability_scores"]
    scores = np.array(probability_scores).flatten()

    assert label in [0, 1]
    assert probability_scores is not None
    assert pytest.approx(np.sum(scores), 0.01) == 1.0


def test_model_performance():
    """
    Tests the performance of the model against expected thresholds.
    This test loads the model's performance metrics from the 'reports/metrics.json'
    file and asserts that the metrics meet the expected thresholds.
    Assertions:
        - Accuracy should at least meet the expected accuracy.
        - Other metrics like Precision, Recall, F1 score, ROC_AUC should at least meet the 
          expected values provided.
    
    """
    # load new model metrics
    with open("reports/metrics.json", 'r') as f:
        metrics = json.load(f)

    # expected thresholds of the performance metrics for model promotion
    assert metrics["accuracy"] >= 0.819
    assert metrics["precision"] >= 0.80
    assert metrics["recall"] >= 0.80
    assert metrics["f1 score"] >= 0.80
    assert metrics["roc_auc"] >= 0.85