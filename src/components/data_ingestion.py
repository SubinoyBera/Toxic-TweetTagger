import os
import sys
import requests
import zipfile
from src.logger.logging import logging
from src.exception.app_exception import AppException
from src.utils.common import create_directory
from src.config.configuration import AppConfiguration

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
        Downloads data from the given url and saves it into a zip file into the given location.

        Returns:
            str: The path of the downloaded zip file
        """
        try:
            dataset_url = self.data_ingestion_config.data_download_url
            download_dir = self.data_ingestion_config.raw_data_dir

            create_directory(download_dir)
            data_filename = os.path.basename(dataset_url)
            download_data_path = os.path.join(download_dir, data_filename)

            response = requests.get(dataset_url)
            if response.status_code == 200:
                with open(download_data_path, 'wb') as f:
                    f.write(response.content)
                    logging.info(f"Downloaded data successfully into file: {download_data_path}")

            return download_data_path

        except Exception as e:
            logging.error(f"Failed to download data: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def extract_zipfile(self, file_path):
        """
        Extracts the given zip file into a given directory.
        Args:
            zip_file_path (str): The path of the zip file to be extracted
        """
        try:
            ingested_dir = self.data_ingestion_config.ingested_dir
            create_directory(ingested_dir)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(ingested_dir)
            logging.info(f"Extracting zip file: {file_path} into dir: {ingested_dir}")

        except Exception as e:
            logging.error(f"Failed to extract Zip file: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def initiate_data_ingestion(self):
        """
        Starts the data ingestion process by downloading the data from given url and saving it into a given location. 
        Extracts the given zip file into a given directory.

        Raises:
            AppException: If error occurs during data ingestion
        """
        try:
            logging.info(f"{'='*20}Data Ingestion{'='*20}")
            data = self.download_data()
            self.extract_zipfile(data)
            logging.info(f"{'='*20}Data Ingestion Completed Successfully{'='*20} \n\n")

        except Exception as e:
            logging.error(f"Error in Data Ingestion process: {e}", exc_info=True)
            raise AppException(e, sys)