from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataProcessor:
    """Performs sentiment analysis on text."""
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.
        Returns a dictionary with sentiment label and score.
        """
        sentiment = self.analyzer.polarity_scores(text)
        compound_score = sentiment['compound']
        
        if compound_score >= 0.05:
            label = 'Positive'
        elif compound_score <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
            
        return {"label": label, "score": compound_score}