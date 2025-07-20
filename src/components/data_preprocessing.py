from components import *
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

#nltk.download('stopwords')
#nltk.download('wordnet')

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
    
    def lemmatization(self, text):
        """Converts all words in the given text to their base form using WordNet lemmatization."""
        lemmatizer = WordNetLemmatizer()
        text_words = text.split()
        text = [lemmatizer.lemmatize(word) for word in text_words]
        return " ".join(text)


class DataPreprocessing:
    def __init__(self, config = AppConfiguration()):
        """
        Initializes the DataPreprocessing object by creating a data preprocessing configuration.

        Raises:
            AppException: If error occurs during creation of data preprocessing configuration
        """

        try:
            self.data_preprocessing_config = config.data_preprocessing_config()

        except Exception as e:
            logging.error(f"Failed to create data preprocessing configuration: {e}", exc_info=True)
            raise AppException(e, sys)


    def preprocess(self, df):
        """
        Performs different data processing steps on the given dataframe
            Lowercasing the content,
            Removing punctuations from the content,
            Removing english stopwords, and Lemmatizing the content.
        
        Saves the preprocessed data in csv file after performing data preprocessing.

        Raises:
            AppException: If error occurs during data preprocessing
        """
        # Sampling from dataframe to create a smaller working dataset
        df['Label'] = df['Label'].astype(int)
        subset_hate_df = df[df['Label']==1].sample(n=60000, random_state=42)
        subset_nornal_df = df[df['Label']==0].sample(n=60000, random_state=42)

        subset_df = pd.concat([subset_hate_df, subset_nornal_df])

        fn = HelperFunctions()
        # Preprocessing steps
        try:
            subset_df.dropna(how='any', inplace=True)

            logging.info("Performing lowercasing")
            subset_df['Content'] = subset_df['Content'].apply(fn.lower_case)

            logging.info("Removing punctuations")
            subset_df['Content'] = subset_df['Content'].apply(fn.remove_punctuations)

            logging.info("Removing stopwords")
            subset_df['Content'] = subset_df['Content'].apply(fn.remove_stopwords)

            logging.info("Performing lemmatization")
            subset_df['Content'] = subset_df['Content'].apply(fn.lemmatization)

            logging.info("Finished preprocessing operations successfully")

            processed_data_dir = self.data_preprocessing_config.preprocessed_data_dir
            create_directory(processed_data_dir)
            subset_df.to_csv(Path(processed_data_dir, 'clean_data.csv'), index=False)
            # free memory
            del subset_df
            logging.info(f"Data successfully saved at {processed_data_dir}")
        
        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}", exc_info=True)
            raise AppException(e, sys)
    
def main():
    """
    Main function to initiate the data preprocessing workflow. It reads ingested dataset,
    performs different preprocessing operations and saves the preprocessed data as a CSV file.

    Raises:
        AppException: If an error occurs during data preprocessing.
    """
    obj = DataPreprocessing()
    try:
        logging.info(f"{'='*20}Data Preprocessing{'='*20}")
        data_path = obj.data_preprocessing_config.ingested_dataset_path
        if not data_path:
            logging.error("No data path found")
            return
        df = pd.read_csv(data_path, encoding='utf-8')
        obj.preprocess(df)
        del df
        logging.info(f"{'='*20}Data Preprocessing Completed Successfully{'='*20} \n\n")

    except Exception as e:
        logging.error(f"Error in Data Preprocessing process: {e}", exc_info=True)
        raise AppException(e, sys)
    
# entry point for the data preprocessing process
if __name__ == "__main__":
    main()