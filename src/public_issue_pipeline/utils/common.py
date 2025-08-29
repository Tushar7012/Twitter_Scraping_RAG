import re
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def clean_text(text: str) -> str:
    """
    Cleans tweet text by removing URLs, mentions, hashtags, and stopwords.
    """
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#','', text)
    text = text.lower() 
    text_tokens = text.split()
    filtered_words = [word for word in text_tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered_words)