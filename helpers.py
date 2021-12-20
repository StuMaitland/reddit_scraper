import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def clean_tweet(tweet):
    """
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    """
    Utility function to classify the polarity of a tweet
    using textblob.
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(clean_tweet(tweet))['compound']