QUERY = "potholes OR #powercut OR pollution OR governance -is:retweet lang:en"
MAX_RESULTS = 100
TWEET_FIELDS = ["created_at", "public_metrics", "conversation_id","geo"]
USER_FIELDS = ["username"]

DB_NAME = "twitter_rag"
DB_USER = "postgres"
DB_PASSWORD = "12345"
DB_HOST = "localhost"
DB_PORT = "5432"