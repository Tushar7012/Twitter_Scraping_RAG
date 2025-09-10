import psycopg2
from public_issue_pipeline.constants import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

class DatabaseStorage:
    """Handles PostgreSQL database operations with pgvector."""
    def __init__(self):
        self.connection = None
        self.db_params = {
            "dbname": DB_NAME,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "host": DB_HOST,
            "port": DB_PORT
        }

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(**self.db_params)
            print("Database connection successful.")
        except psycopg2.OperationalError as e:
            print(f"Could not connect to the database: {e}")
            raise

    def create_table(self):
        """Creates the tweets table with a vector column if it doesn't exist."""
        if not self.connection:
            self.connect()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS tweets (
            id BIGINT PRIMARY KEY,
            author TEXT,
            timestamp TIMESTAMPTZ,
            text TEXT,
            sentiment TEXT,
            sentiment_score FLOAT,
            likes INT,
            embedding VECTOR(384),
            location GEOGRAPHY(Point, 4326)
        );
        """
        with self.connection.cursor() as cur:
            # First, ensure the extensions are enabled in this session
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            # Then, create the table
            cur.execute(create_table_query)
            self.connection.commit()
            print("Table 'tweets' is ready.")

    def insert_tweet(self, tweet_data: dict):
        """
        Inserts a single processed tweet record into the database,
        including location data if available.
        """
        # Check if location data exists in the tweet data
        has_location = 'longitude' in tweet_data and 'latitude' in tweet_data

        if has_location:
            # Query to insert with location
            insert_query = """
            INSERT INTO tweets (
                id, author, timestamp, text, sentiment, sentiment_score, likes, embedding, location
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) ON CONFLICT (id) DO NOTHING;
            """
            values = (
                tweet_data['id'],
                tweet_data['author'],
                tweet_data['timestamp'],
                tweet_data['text'],
                tweet_data['sentiment'],
                tweet_data['score'],
                tweet_data['likes'],
                tweet_data['embedding'],
                tweet_data['longitude'],
                tweet_data['latitude']
            )
        else:
            # Query to insert without location (inserts NULL by default)
            insert_query = """
            INSERT INTO tweets (
                id, author, timestamp, text, sentiment, sentiment_score, likes, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING;
            """
            values = (
                tweet_data['id'],
                tweet_data['author'],
                tweet_data['timestamp'],
                tweet_data['text'],
                tweet_data['sentiment'],
                tweet_data['score'],
                tweet_data['likes'],
                tweet_data['embedding']
            )

        with self.connection.cursor() as cur:
            cur.execute(insert_query, values)
        self.connection.commit()

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")