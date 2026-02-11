import sys, os
import pymongo
import pandas as pd
from urllib.parse import quote_plus
from src.core.logger import logging
from src.core.exception import AppException
from dotenv import load_dotenv
load_dotenv()

mongo_username = os.getenv("MONGO_USERNAME")
mongo_password = os.getenv("MONGO_PASSWORD")
url = os.getenv("MONGO_URL")

class MongoDBClient:
    def __init__(self):
        """
        Initializes the MongoDBClient object by connecting to the MongoDB instance.
        This method connects to the MongoDB instance using the credentials stored in environment variables.
        """
        try:
            username = quote_plus(str(mongo_username).strip())
            password = quote_plus(str(mongo_password).strip())

            mongo_connection_url = f"mongodb+srv://{username}:{password}@{url}"

            self.client = pymongo.MongoClient(mongo_connection_url)
            logging.info("MongoDB connection successfull")
        
        except Exception as e:
            logging.error(f"Failed to inititalize MongoDBClient -connection failed: {e}", exc_info=True)
            raise AppException(e, sys)
    
    def get_all_docs(self, collection_name:str, database_name:str) -> pd.DataFrame:
        """
        Fetches data from the specified MongoDB collection and database.

        Args:
            collection_name (str): The name of the MongoDB collection.
            database_name (str): The name of the MongoDB database.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the downloaded data.
        """
        try:
            database = self.client[database_name]
            collection = database[collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns="_id", inplace=True)
            logging.info("Data downloaded successfully")
            return df
        
        except Exception as e:
            logging.error(f"Failed to fetch data from MongoDB: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def insert_docs(self, collection_name:str, database_name:str, docs:list):
        """
        Inserts data into the specified MongoDB collection and database.

        Args:
            collection_name (str): The name of the MongoDB collection.
            database_name (str): The name of the MongoDB database.
            data (list): A list of dictionaries containing the data to be inserted.

        Raises:
            AppException: If an error occurs while inserting data into MongoDB.
        """
        try:
            database = self.client[database_name]
            collection = database[collection_name]

            collection.insert_many(docs)
            logging.info("Data inserted successfully")
        
        except Exception as e:
            logging.error(f"Failed to insert data into MongoDB: {e}", exc_info=True)
            raise AppException(e, sys)
        

    def close_connection(self):
        """
        Closes the MongoDB connection.
        """
        try:
            self.client.close()
            logging.info("MongoDB connection closed successfully")

        except Exception as e:
            logging.warning(f"Failed to close MongoDB connection: {e}", exc_info=True)