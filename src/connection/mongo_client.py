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
        try:
            username = quote_plus(str(mongo_username).strip())
            password = quote_plus(str(mongo_password).strip())

            mongo_connection_url = f"mongodb+srv://{username}:{password}@{url}"

            self.client = pymongo.MongoClient(mongo_connection_url)
            logging.info("MongoDB connection successfull")
        
        except Exception as e:
            logging.error(f"Failed to inititalize MongoDBClient -connection failed: {e}", exc_info=True)
            raise AppException(e, sys)
        
    
    def fetch_data(self, collection_name:str, database_name:str) -> pd.DataFrame:
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