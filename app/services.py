import os
import mlflow
import dagshub
import pandas as pd
import emoji
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')

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

# Helper Functions from text preprocessing
class HelperFunctions:
    def __init__(self):
        """
        Initializes an instance of the HelperFunctions class.

        This class contains methods for various text preprocessing tasks such as
        lowercasing, removing punctuations, removing stopwords, and lemmatization.
        """
        pass
    
    def lower_case(self, text):
        """Converts the given text to lowercase."""
        return text.lower()
    
    def remove_punctuations(self, text):
        """Removes all punctuation marks from the given text."""
        exclude = string.punctuation
        return text.translate(str.maketrans("", "", exclude))
    
    def remove_stopwords(self, text):
        """Removes all English stopwords from the given text."""
        stop_words = set(stopwords.words('english'))
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)
    
    def emojis_to_texts(self, text):
        """Replace emojis with meaning from the given text."""
        return emoji.demojize(text)
    
    def lemmatization(self, text):
        """Converts all words in the given text to their base form using WordNet lemmatization."""
        lemmatizer = WordNetLemmatizer()
        text_words = text.split()
        text = [lemmatizer.lemmatize(word) for word in text_words]
        return " ".join(text)


def preprocess(tweet: str):
    """
    Preprocesses a given tweet by performing the following tasks:
        1. Lowercasing the tweet
        2. Removing all punctuation marks from the tweet
        3. Removing all English stopwords from the tweet
        4. Replacing all emojis with their meaning from the tweet
        4. Lemmatizing all words in the tweet to their base form using WordNet lemmatization

    Returns a dataframe containing the preprocessed tweet
    """
    obj = HelperFunctions()
    df = pd.DataFrame({
        "comments": [tweet]
    })
    df["comments"] = df["comments"].apply(obj.lower_case)
    df["comments"] = df["comments"].apply(obj.remove_punctuations)
    df["comments"] = df["comments"].apply(obj.remove_stopwords)
    df['comments'] = df['comments'].apply(obj.emojis_to_texts)
    df["comments"] = df["comments"].apply(obj.lemmatization)
    
    return df