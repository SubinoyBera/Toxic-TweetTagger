from src.components import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

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

    def perform_feature_engineering(self, df):
        """
        Performs feature engineering on the given dataframe by extracting features with 
        TF-IDF vectorization and also splits data into training and testing sets.
        Saves the vectorizer object and train-test datasets.

        Raises:
            AppException: If error occurs during feature engineering
        """
        try:
            vectorizer = TfidfVectorizer(max_features=3000, min_df=3)

            logging.info("Performing TF-IDF vectorization")
            X_tfidf = vectorizer.fit_transform(df['Content'])
            X_tfidf = csr_matrix(X_tfidf)

            featured_data = pd.DataFrame(X_tfidf.toarray())
            featured_data['Label'] = df['Label'].values

            train_data, test_data = train_test_split(featured_data, test_size=0.2, random_state=42)
            # free memory
            del featured_data, X_tfidf

            logging.info("Saving vectorizer object and training-testing sets")
            save_models_path = self.eng_config.models_dir
            save_obj(location_path=save_models_path, obj_name='vectorizer.pkl', obj=vectorizer)

            save_training_data_path = self.eng_config.train_test_data_path
            train_data.to_csv(Path(save_training_data_path, 'train_data.csv'), index=False)
            test_data.to_csv(Path(save_training_data_path, 'test_data.csv'), index=False)
            # free memory
            del train_data, test_data
            logging.info("Feature engineering done on data. Saved vectorizer obj and training-testing datasets")

        except Exception as e:
            logging.error(f"Feature engineering operation terminated: {e}", exc_info=True)
            raise AppException(e, sys)


def initiate_feature_engineering():
    """
    Initiates the feature engineering process by reading the preprocessed dataset.
    """
    obj = FeatureEngineering()
    try:
        logging.info(f"{'='*20}Feature Engineering{'='*20}")
        data_path = obj.eng_config.preprocessed_data_path
        if not data_path:
            logging.error("Dataset path after preprocessing stage not found")
        df = pd.read_csv(data_path)
        obj.perform_feature_engineering(df)
        del df
        logging.info(f"{'='*20}Feature Engineering Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error during Feature Engineering: {e}", exc_info=True)
        raise AppException(e, sys)