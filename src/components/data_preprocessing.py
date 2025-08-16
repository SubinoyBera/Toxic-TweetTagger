import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

import sys
import string
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from src.core.logger import logging
from src.core.exception import AppException
from src.core.configuration import AppConfiguration
import gc


class HelperFunctions:
    def __init__(self):
        """
        Initializes an instance of the HelperFunctions class.

        This class contains methods for various text preprocessing tasks such as
        lowercasing, removing punctuations, removing stopwords, and lemmatization.
        """
        pass
    
    def lower_case(self, text) -> str:
        """Converts the given text to lowercase."""
        return text.lower()
    
    def remove_punctuations(self, text) -> str:
        """Removes all punctuation marks from the given text."""
        exclude = string.punctuation
        return text.translate(str.maketrans("", "", exclude))
    
    def remove_stopwords(self, text) -> str:
        """Removes all English stopwords from the given text."""
        stop_words = set(stopwords.words('english'))
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)

    def lemmatization(self, text) -> str:
        """
        Does parts of speech tagging and then converts words to their base form 
        using WordNetLemmatizer.
        """
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        # Perform POS tagging and lemmatization
        pos_text = pos_tag(text.split())
        text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text]
        return " ".join(text)


class DataPreprocessing:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the DataPreprocessing object by creating the data prepocessing configuration.
        Args:
            config (AppConfiguration): The configuration object containing the application configuration.
        """
        try:
            self.data_preprocessing_config = config.data_preprocessing_config()

        except Exception as e:
            logging.error(f"Failed to create data preprocessing configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def preprocess(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Preprocesses the given dataframe by performing lowercasing, removing punctuation, 
        removing stopwords and performing lemmatization with parts of speech tagging.
        Args:
            df (pd.DataFrame): The dataframe to be preprocessed
            filename (str): The filename to save the preprocessed dataframe

        Returns:
            pd.DataFrame: The preprocessed dataframe
        """
        fn = HelperFunctions()
        # Preprocessing steps
        try:
            df.dropna(how='any', inplace=True)
            tqdm.pandas()

            logging.info("Performing lowercasing")
            df['Content'] = df['Content'].progress_apply(fn.lower_case)

            logging.info("Removing punctuations")
            df['Content'] = df['Content'].progress_apply(fn.remove_punctuations)

            logging.info("Performing pos tagging and lemmatization")
            df['Content'] = df['Content'].progress_apply(fn.lemmatization)

            logging.info("Removing stopwords")
            df['Content'] = df['Content'].progress_apply(fn.remove_stopwords)

            logging.info("Finished preprocessing operations successfully")

            preprocessed_data_dir = self.data_preprocessing_config.preprocessed_data_dir
            df.to_feather(Path(preprocessed_data_dir, filename))
            
            logging.info(f"Data successfully saved at {preprocessed_data_dir}")
            return df
        
        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}", exc_info=True)
            raise AppException(e, sys)
    
def initiate_data_preprocessing():
    """
    Main function to initiate the data preprocessing workflow. It reads ingested dataset,
    performs different preprocessing operations and saves the preprocessed data as a CSV file.

    Raises:
        AppException: If an error occurs during data preprocessing.
    """
    obj = DataPreprocessing()
    try:
        logging.info(f"{'='*20}Data Preprocessing{'='*20}")
        data_path = obj.data_preprocessing_config.data_path
        if not data_path:
            raise ValueError("No data path found")
        
        df = pd.read_parquet(data_path)
        preprocessed_dataset_name = obj.data_preprocessing_config.preprocessed_data_filename
        obj.preprocess(df, preprocessed_dataset_name)
        del df
        gc.collect()
        logging.info(f"{'='*20}Data Preprocessing Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error in Data Preprocessing process: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":
    initiate_data_preprocessing()