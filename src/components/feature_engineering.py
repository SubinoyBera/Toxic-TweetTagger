import sys
from src.constant.constants import PARAMS_FILE
from src.core.configuration import AppConfiguration
from src.core.logger import logging
from src.core.exception import AppException
from src.utils import read_yaml, save_obj
import gc
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineering:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the FeatureEngineering object by creating a feature engineering configuration.
        Args:
            config (AppConfiguration): The configuration object containing the application configuration.
        """
        try:
            self.eng_config = config.feature_engineering_config()

        except Exception as e:
            logging.error(f"Failed to create feature engineering Configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def perform_feature_engineering(self, df: pd.DataFrame):
        """
        Performs feature engineering on the given dataframe by extracting features with TF-IDF vectorization 
        and also splits data into training and testing sets.
        Saves the vectorizer object and training dataset.
        """
        try:
            config_params = read_yaml(PARAMS_FILE)
            params = config_params.feature_engineering
            vectorizer_name = params.vectorizer

            vectorizer = TfidfVectorizer(max_features=params.max_features,
                                        min_df=params.min_df, ngram_range=(params.ngrams.min, params.ngrams.max)
                                    )

            logging.info("Performing TF-IDF vectorization")
            X_tfidf = vectorizer.fit_transform(df['Content'])
            X_tfidf = csr_matrix(X_tfidf)

            training_data = pd.DataFrame(X_tfidf.toarray())
            training_data['Label'] = df['Label'].values

            save_model_path = self.eng_config.models_dir
            save_obj(location_path=save_model_path, obj_name=f"vectorizer.joblib", obj=vectorizer)
            
            with open(Path(save_model_path, "vectorizer_meta.txt"), 'w') as f:
                f.write(f"{vectorizer_name} has been created and fitted on the training data\n\n {params}")
            
            logging.info("Saving training dataset")
            training_data.to_feather(self.eng_config.training_data_path)

            logging.info("Feature engineering operation done")

        except Exception as e:
            logging.error(f"Error - feature engineering operation terminated: {e}", exc_info=True)
            raise AppException(e, sys)


def initiate_feature_engineering():
    """
    Main function to initiate the feature engineering workflow. It reads preprocessed data, 
    performs feature engineering on the data, and splits data into training and testing sets.

    Raises:
        AppException: If an error occurs during feature engineering.
    """
    obj = FeatureEngineering()
    try:
        logging.info(f"{'='*20}Feature Engineering{'='*20}")
        data_path = obj.eng_config.preprocessed_data_path
        if not data_path:
            logging.error("Dataset path after preprocessing stage not found")
        df = pd.read_feather(data_path)
        df.dropna(how='any', inplace=True)
        obj.perform_feature_engineering(df)
        del df, obj
        gc.collect()
        logging.info(f"{'='*20}Feature Engineering Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error during Feature Engineering: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":
    initiate_feature_engineering()