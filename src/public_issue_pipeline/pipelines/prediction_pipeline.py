import json
from datetime import datetime
from dataclasses import replace
from public_issue_pipeline.config.configuration import ConfigurationManager
from public_issue_pipeline.components.data_ingestion import DataIngestion
from public_issue_pipeline.components.data_processing import DataProcessor
from public_issue_pipeline.utils.common import clean_text

class PredictionPipeline:
    def __init__(self):
        self.data_processor = DataProcessor()

    def _process_tweet_data(self, tweet, users):
        """Helper function to process a single tweet object and return a dictionary."""
        author_info = users.get(tweet.author_id, {"username": "unknown"})
        cleaned = clean_text(tweet.text)
        sentiment = self.data_processor.get_sentiment(cleaned)
        return {
            "id": tweet.id,
            "author": f"@{author_info.username}",
            "timestamp": tweet.created_at.isoformat(),
            "sentiment": sentiment['label'],
            "score": sentiment['score'],
            "likes": tweet.public_metrics['like_count'],
            "text": tweet.text
        }

    def run(self, keyword: str):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            new_query = f"{keyword} -is:retweet lang:en"
            data_ingestion_config = replace(data_ingestion_config, query=new_query, max_results=10)
            
            data_ingestion = DataIngestion(config=data_ingestion_config)

            parent_response = data_ingestion.fetch_parent_tweets()
            if not parent_response.data:
                print(f"No parent tweets found for the keyword: '{keyword}'")
                return

            all_results = []
            parent_tweets = parent_response.data
            users = {user["id"]: user for user in parent_response.includes.get("users", [])}

            print(f"\nProcessing {len(parent_tweets)} parent tweets and their replies...\n")
            for parent_tweet in parent_tweets:
                # Process the parent tweet
                parent_data = self._process_tweet_data(parent_tweet, users)
                
                # Fetch and process its replies
                parent_data["replies"] = []
                reply_response = data_ingestion.fetch_replies(parent_tweet.conversation_id)
                
                if reply_response.data:
                    reply_users = {user["id"]: user for user in reply_response.includes.get("users", [])}
                    # Update the main users dictionary with any new users from replies
                    users.update(reply_users)
                    
                    for reply_tweet in reply_response.data:
                        # Ensure we don't add the parent tweet again if it appears in replies
                        if reply_tweet.id != parent_tweet.id:
                            reply_data = self._process_tweet_data(reply_tweet, users)
                            parent_data["replies"].append(reply_data)
                
                all_results.append(parent_data)

            # Save the structured data to a JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_keyword = "".join(c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            filename = f"{safe_keyword}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            
            print(f"\n✅ Success! Saved conversations to '{filename}'")

        except Exception as e:
            print(f"An error occurred in the pipeline: {e}")
            raise e