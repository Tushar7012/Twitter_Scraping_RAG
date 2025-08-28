import os
import tweepy
from dotenv import load_dotenv
from public_issue_pipeline.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """Fetches data from the Twitter API."""
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        load_dotenv()
        # --- FIX: Corrected the environment variable name ---
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        # ---------------------------------------------------
        if not bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN not found in .env file!")
        self.client = tweepy.Client(bearer_token)

    def fetch_parent_tweets(self):
        """Fetches parent tweets based on the configuration."""
        print("Fetching parent tweets...")
        response = self.client.search_recent_tweets(
            query=self.config.query,
            max_results=self.config.max_results,
            tweet_fields=self.config.tweet_fields,
            user_fields=self.config.user_fields,
            expansions=["author_id"]
        )
        print("Parent tweets fetched successfully.")
        return response

    def fetch_replies(self, conversation_id: str):
        """Fetches replies for a given conversation ID."""
        print(f"  Fetching replies for conversation ID: {conversation_id}...")
        query = f"conversation_id:{conversation_id} -is:retweet lang:en"
        response = self.client.search_recent_tweets(
            query=query,
            max_results=100, # Fetch up to 100 replies per conversation
            tweet_fields=self.config.tweet_fields,
            user_fields=self.config.user_fields,
            expansions=["author_id"]
        )
        return response