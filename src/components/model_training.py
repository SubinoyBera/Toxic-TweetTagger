from src.components import *
import pandas as pd
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the ModelTrainer object by creating a model training configuration.
        Args:
            config (AppConfiguration): The configuration object containing the application configuration.
        """
        try:
            self.model_training_config = config.model_training_config()

        except Exception as e:
            logging.error(f"Failed to create model training configuration: {e}", exc_info=True)
            raise AppException(e, sys)

        
    def train(self, df):
        """
        Trains ML model on the given training data and saves the model.

        Args:
            df: The training dataframe
        """
        y_train = df['Label'].values
        X_train = df.drop(columns='Label')

        model = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=12, 
                              gamma=0.1, reg_lambda=1,subsample=0.8, n_jobs=-1,
                              random_state=42, use_label_encoder=False)
        try:
            logging.info("Model training started")
            model.fit(X_train, y_train)

            save_model_path = self.model_training_config.models_dir
            save_obj(location_path=save_model_path, obj=model, obj_name='model.pkl')
            logging.info(f"Model trained as saved at: {save_model_path}")

            # free memory
            del model, X_train, y_train
    
        except Exception as e:
            logging.error(f"Failed to train model: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
def initiate_model_training():
    """
    Initiates the model training process by reading the training dataset
    and training a model using the XGBoost classifier. The trained model
    is saved to the specified directory.

    Raises:
        AppException: If an error occurs during model training.
    """

    obj = ModelTrainer()
    try:
        logging.info(f"{'='*20}Model training{'='*20}")
        train_data_path = obj.model_training_config.train_data_path
        if not train_data_path:
            logging.error("Training dataset path not found")
            return
        df = pd.read_csv(train_data_path)
        obj.train(df)
        del df
        logging.info(f"{'='*20}Model Training Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise AppException(e, sys)