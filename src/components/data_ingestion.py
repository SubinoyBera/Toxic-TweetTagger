import sys
from sklearn.model_selection import train_test_split
from src.constant.constants import DATABASE_NAME, TRAINING_COLLECTION_NAME
from src.db.mongo_client import MongoDBClient
from src.core.logger import logging
from src.core.exception import AppException
from src.core.configuration import AppConfiguration
import gc

class DataIngestion:
    def __init__(self, app_config = AppConfiguration()):
        """
        DataIngestion Intialization
        data_ingestion_config: DataIngestionConfig 
        """
        try:
            self.data_ingestion_config = app_config.data_ingestion_config()

        except Exception as e:
            logging.error(f"Data Ingestion error: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
    def download_data(self):
        """
        Downloads data from MongoDB and saves it in csv format, and also
        splits the data into training and testing sets and saves them in parquet format.

        Raises:
            AppException: If there is an error downloading data from MongoDB
        """
        try:
            save_ingested_data_path = self.data_ingestion_config.ingested_data_path
            save_train_data_path = self.data_ingestion_config.train_data_path
            save_test_data_path = self.data_ingestion_config.test_data_path

            logging.info("Connecting to MongoDB")
            client = MongoDBClient()

            data = client.get_all_docs(database_name=DATABASE_NAME, collection_name=TRAINING_COLLECTION_NAME)

            data.to_csv(save_ingested_data_path, index=False)
            logging.info(f"Raw ingested data saved successfully saved at {save_ingested_data_path}")

            # close mongo connection 
            client.close_connection()

            # shuffling data
            data = data.sample(frac=1, random_state=42).reset_index(drop=True)

            logging.info("Splitting data into training and testing sets")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            train_data.to_parquet(save_train_data_path, index=False)
            test_data.to_parquet(save_test_data_path, index=False)
            logging.info(f"Successfully saved train data and test data")
            # free memory
            del data, train_data, test_data
            gc.collect()

        except Exception as e:
            logging.error(f"Failed to download data from MongoDB: {e}", exc_info=True)
            raise AppException(e, sys)
        

def initiate_data_ingestion():
    """
    Main function to initiate the Data Ingestion workflow. It downloads data from MongoDB and
    saves it to the specified directory using the configuration provided in the
    data_ingestion_config.

    Raises:
        AppException: If an error occurs during data ingestion.
    """
    obj = DataIngestion()
    try:
        logging.info(f"{'='*20}Data Ingestion{'='*20}")
        obj.download_data()

        logging.info(f"{'='*20}Data Ingestion Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error in Data Ingestion process: {e}", exc_info=True)
        raise AppException(e, sys)


if __name__ == "__main__":
    initiate_data_ingestion()