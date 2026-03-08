import pytest
from frontend.utils import HelperFunctions, preprocess

@pytest.fixture
def helper():
    return HelperFunctions()

def test_lower_case(helper):
    text = "HELLO WORLD"
    assert helper.lower_case(text) == "hello world"

def test_remove_punctuations(helper):
    text = "hello!!! world??"
    result = helper.remove_punctuations(text)

    assert "!" not in result
    assert "?" not in result

def test_emojis_to_text(helper):
    text = "I love python ❤️"
    result = helper.emojis_to_texts(text)

    assert ":red_heart:" in result

def test_remove_stopwords(helper):
    text = "this is a very good movie"
    result = helper.remove_stopwords(text)

    assert "is" not in result
    assert "a" not in result

def test_lemmatization(helper):
    text = "running dogs"
    result = helper.lemmatization(text)

    assert isinstance(result, str)

def test_preprocess_pipeline():
    tweet = "Dogs are running!!! ❤️"

    result = preprocess(tweet)

    assert isinstance(result, str)
    assert "!" not in result
    assert result == result.lower()