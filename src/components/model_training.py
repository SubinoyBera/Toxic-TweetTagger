import sys
import pandas as pd
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from src.constant.constants import PARAMS_FILE
from src.core.logger import logging
from src.core.exception import AppException
from src.core.configuration import AppConfiguration
from src.utils import read_yaml, save_obj
import gc

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

        try:
            config_params = read_yaml(PARAMS_FILE)
            params = config_params.model_training

            model = XGBClassifier(n_estimators=params.hyperparameters.n_estimators,
                              learning_rate=params.hyperparameters.learning_rate,
                              max_depth=params.hyperparameters.max_depth, 
                              tree_method=params.hyperparameters.tree_method,
                              max_bin=params.hyperparameters.max_bin,
                              scale_pos_weight=params.hyperparameters.scale_pos_weight,
                              gamma=params.hyperparameters.gamma,
                              reg_lambda=params.hyperparameters.reg_lambda,
                              subsample=params.hyperparameters.subsample,
                              colsample_bytree=params.hyperparameters.colsample_bytree,
                              n_jobs=-1, random_state=42, use_label_encoder=False, predictor="cpu_predictor"
                            )
        
            logging.info("Model training started")
            model.fit(X_train, y_train)
            # get XGB booster
            booster = model.get_booster()
            
            save_model_path = self.model_training_config.models_dir
            
            # save models
            booster.save_model(Path(save_model_path, "booster.json"))
            save_obj(location_path=save_model_path, obj=model, obj_name=f"model.joblib")
            logging.info(f"Model trained and saved at: {save_model_path}")

            with open(Path(save_model_path, "model_meta.txt"), 'w') as f:
                f.write(f"{params.model_name} model has been trained successfully.\n\n {params}")

            # free memory
            del X_train, y_train, model
            gc.collect()
    
        except Exception as e:
            logging.error(f"Failed to train model: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
def initiate_model_training():
    """
    Main function to initiate the model training workflow. It reads the training dataset,
    trains an ML model and saves the model.
    
    Raises:
        AppException: If an error occurs during model training.
    """
    obj = ModelTrainer()
    try:
        logging.info(f"{'='*20}Model Training{'='*20}")
        train_data_path = obj.model_training_config.training_data_path
        if not train_data_path:
            logging.error("Training dataset path not found")
            return
        df = pd.read_feather(train_data_path)
        df.dropna(how='any', inplace=True)
        obj.train(df)
        del df
        gc.collect()
        logging.info(f"{'='*20}Model Training Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise AppException(e, sys)
    

if __name__ == "__main__":  
    initiate_model_training()