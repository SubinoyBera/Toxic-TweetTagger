# app utiliy functions for fastapi app backend
import os
import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

import emoji
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Helper Functions from text preprocessing
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
    
    def emojis_to_texts(self, text):
        """Replace emojis with meaning from the given text."""
        return emoji.demojize(text)
    
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

    def remove_stopwords(self, text):
        """Removes all English stopwords from the given text."""
        stop_words = set(stopwords.words('english'))
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)
    

def preprocess(tweet: str) -> str:
    """
    Preprocesses a given tweet by performing the following tasks:
        1. Lowercasing the tweet
        2. Removing all punctuation marks from the tweet
        3. Removing all English stopwords from the tweet
        4. Replacing all emojis with their meaning from the tweet
        4. Lemmatizing all words in the tweet to their base form using WordNet lemmatization

    Returns the preprocessed tweet
    """
    obj = HelperFunctions()
    tweet = obj.lower_case(tweet)
    tweet = obj.remove_punctuations(tweet)
    tweet = obj.emojis_to_texts(tweet)
    tweet = obj.lemmatization(tweet)
    tweet = obj.remove_stopwords(tweet)
    
    return tweet