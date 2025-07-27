import os
import mlflow
import dagshub
import pandas as pd
from src.components.data_preprocessing import HelperFunctions
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv("MLFLOW_URI")
repo_owner = os.getenv("OWNER")
repo_name = os.getenv("REPO")

mlflow.set_tracking_uri(uri)    # type: ignore

if repo_owner is None:
	raise ValueError("Missing dagshub logging environment credentials.")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)       # type: ignore


class Model:
    def __init__(self) -> None:
        pass

    def load_model(self):
        try:
            model_name = "ToxicTagger-Models"
            client = mlflow.MlflowClient()
            get_latest_version = client.get_latest_versions(name=model_name, stages=["Staging"])
            model_version = get_latest_version[0].version
            model_uri = f"models:/{model_name}/{model_version}"
            self.model = mlflow.pyfunc.load_model(model_uri)

            return self.model, model_version
        
        except Exception as e:
            raise (e)
    

    def get_model_name(self):
        classifier_model = self.model._model_impl.python_model.model
        return type(classifier_model).__name__
    
    def get_vectorizer_name(self):
        vectorizer = self.model._model_impl.python_model.vectorizer
        return type(vectorizer).__name__


def preprocess(tweet: str):
    obj = HelperFunctions()
    df = pd.DataFrame({
        "texts": [tweet]
    })
    df["texts"] = df["texts"].apply(obj.lower_case)
    df["texts"] = df["texts"].apply(obj.remove_punctuations)
    df["texts"] = df["texts"].apply(obj.remove_stopwords)
    df["texts"] = df["texts"].apply(obj.lemmatization)
    
    return df
