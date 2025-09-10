from dataclasses import replace
from public_issue_pipeline.config.configuration import ConfigurationManager
from public_issue_pipeline.components.data_ingestion import DataIngestion
from public_issue_pipeline.components.data_processing import DataProcessor
from public_issue_pipeline.components.embedding_generation import EmbeddingGenerator
from public_issue_pipeline.components.db_storage import DatabaseStorage
from public_issue_pipeline.utils.common import clean_text

class PredictionPipeline:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.db_storage = DatabaseStorage()

    def run(self, keyword: str):
        processed_ids = []  # List to store IDs of processed tweets
        try:
            # 1. Setup Database
            self.db_storage.connect()
            self.db_storage.create_table()

            # 2. Configure and run data ingestion
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            new_query = f"{keyword} -is:retweet lang:en"
            # Set a low number for testing to avoid hitting rate limits quickly
            data_ingestion_config = replace(data_ingestion_config, query=new_query, max_results=10)
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # 3. Fetch Parent Tweets and Replies
            parent_response = data_ingestion.fetch_parent_tweets()
            if not parent_response.data:
                print(f"No parent tweets found for the keyword: '{keyword}'")
                return [] # Return an empty list if no tweets are found

            all_tweets_to_process = list(parent_response.data)
            users = {user["id"]: user for user in parent_response.includes.get("users", [])}
            
            for parent_tweet in parent_response.data:
                reply_response = data_ingestion.fetch_replies(parent_tweet.conversation_id)
                if reply_response and reply_response.data:
                    all_tweets_to_process.extend(reply_response.data)
                    reply_users = {user["id"]: user for user in reply_response.includes.get("users", [])}
                    users.update(reply_users)
            
            # 4. Process and Store each Tweet
            print(f"\nProcessing and storing {len(all_tweets_to_process)} tweets...")
            for tweet in all_tweets_to_process:
                author_info = users.get(tweet.author_id, {"username": "unknown"})
                cleaned = clean_text(tweet.text)
                
                if not cleaned: continue

                sentiment = self.data_processor.get_sentiment(cleaned)
                embedding = self.embedding_generator.generate_embedding(cleaned)

                tweet_data = {
                    "id": tweet.id,
                    "author": f"@{author_info.username}",
                    "timestamp": tweet.created_at,
                    "text": tweet.text,
                    "sentiment": sentiment['label'],
                    "score": sentiment['score'],
                    "likes": tweet.public_metrics['like_count'],
                    "embedding": embedding.tolist()
                }
                
                self.db_storage.insert_tweet(tweet_data)
                processed_ids.append(tweet.id) # Add the ID to our list

            print(f"\n✅ Success! Data has been stored in the PostgreSQL database.")
            return processed_ids # Return the list of processed IDs

        except Exception as e:
            print(f"An error occurred in the pipeline: {e}")
            raise e
        finally:
            if self.db_storage and self.db_storage.connection:
                self.db_storage.close()