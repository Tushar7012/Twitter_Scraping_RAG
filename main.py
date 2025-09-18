import os
import argparse
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
import tweepy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np # Import numpy

# --- Configuration Constants (loaded from .env) ---
load_dotenv()

# Twitter API
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER_TOKEN:
    raise ValueError("TWITTER_BEARER_TOKEN not found in .env file.")

# Database
DB_NAME = os.getenv("DB_NAME", "twitter_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
    raise ValueError("One or more database environment variables are not set. Check your .env file.")

# --- Global Model Initialization ---
# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Embedding Generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions (formerly in utils/common.py) ---
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    return text

def clean_text_for_storage(text):
    # More aggressive cleaning for storage/embedding, less for display
    text = clean_text(text)
    text = text.lower()
    return text

# --- Twitter API Client (simplified from DataIngestion) ---
class TwitterClient:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token)

    def search_tweets(self, query, max_results=10, tweet_fields=None, user_fields=None, expansions=None):
        try: # Added try-except block for API calls
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            )
            return response
        except tweepy.TweepyException as e:
            print(f"Twitter API error: {e}")
            return None # Return None on API error

# --- Database Operations (simplified from DatabaseStorage) ---
class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.db_params = {
            "dbname": DB_NAME, "user": DB_USER, "password": DB_PASSWORD,
            "host": DB_HOST, "port": DB_PORT
        }

    def connect(self):
        if not self.conn:
            try:
                self.conn = psycopg2.connect(**self.db_params)
                print("Database connection successful.")
            except psycopg2.OperationalError as e:
                print(f"Could not connect to the database: {e}")
                raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")

    def create_table(self):
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            create_table_query = """
            CREATE TABLE IF NOT EXISTS tweets (
                id BIGINT PRIMARY KEY,
                author TEXT,
                timestamp TIMESTAMPTZ,
                raw_text TEXT,
                text TEXT,
                sentiment TEXT,
                sentiment_score FLOAT,
                likes INT,
                embedding VECTOR(384),
                replies JSONB,
                is_issue BOOLEAN,
                issue_type TEXT,
                issue_confidence FLOAT,
                issue_explanation TEXT,
                location GEOGRAPHY(Point, 4326)
            );
            """
            cur.execute(create_table_query)
            self.conn.commit()
            print("Table 'tweets' is ready.")

    def insert_tweet(self, tweet_data: dict):
        self.connect()
        has_location = 'longitude' in tweet_data and 'latitude' in tweet_data

        if has_location:
            insert_query = """
            INSERT INTO tweets (
                id, author, timestamp, raw_text, text, sentiment, sentiment_score, likes, 
                embedding, replies, is_issue, issue_type, issue_confidence, issue_explanation, 
                location
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) ON CONFLICT (id) DO NOTHING;
            """
            values = (
                tweet_data['id'], tweet_data['author'], tweet_data['timestamp'],
                tweet_data['raw_text'], tweet_data['text'], tweet_data['sentiment'],
                tweet_data['score'], tweet_data['likes'], tweet_data['embedding'],
                tweet_data['replies'], tweet_data['is_issue'], tweet_data['issue_type'],
                tweet_data['issue_confidence'], tweet_data['issue_explanation'],
                tweet_data['longitude'], tweet_data['latitude']
            )
        else:
            insert_query = """
            INSERT INTO tweets (
                id, author, timestamp, raw_text, text, sentiment, sentiment_score, likes, 
                embedding, replies, is_issue, issue_type, issue_confidence, issue_explanation
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (id) DO NOTHING;
            """
            values = (
                tweet_data['id'], tweet_data['author'], tweet_data['timestamp'],
                tweet_data['raw_text'], tweet_data['text'], tweet_data['sentiment'],
                tweet_data['score'], tweet_data['likes'], tweet_data['embedding'],
                tweet_data['replies'], tweet_data['is_issue'], tweet_data['issue_type'],
                tweet_data['issue_confidence'], tweet_data['issue_explanation']
            )

        with self.conn.cursor() as cur:
            cur.execute(insert_query, values)
        self.conn.commit()

# --- Issue Detection (Expanded towards PDF framework) ---
PDF_ISSUE_CATEGORIES = {
    "Roads & Infrastructure": ["road", "pothole", "traffic", "bridge", "street light", "construction", "bad road", "broken road", "jam", "congestion", "flyover", "highway", "expressway", "speed breaker", "zebra crossing", "footpath", "drainage system"],
    "Water Supply & Sanitation": ["water", "sewage", "drainage", "garbage", "waste", "cleanliness", "overflow", "supply cut", "pollution", "dirty water", "pipeline leak", "borewell", "handpump", "toilet", "sanitation", "hygiene", "drinking water"],
    "Electricity & Power": ["power", "electricity", "outage", "cut", "light", "transformer", "cable", "no power", "blackout", "load shedding", "fault", "meter", "bill", "streetlamp"],
    "Public Safety & Crime": ["crime", "safety", "theft", "harassment", "police", "security", "robbery", "molestation", "accident", "unsafe", "vandalism", "extortion", "assault", "emergency", "fire", "law and order"],
    "Environment & Pollution": ["pollution", "air quality", "smog", "dumping", "greenery", "deforestation", "waste management", "plastic", "litter", "environment", "global warming", "climate change", "tree cutting", "noise pollution"],
    "Health Services": ["hospital", "doctor", "clinic", "medical", "health", "medicine", "ambulance", "patient care", "sanitation in hospitals", "nurse", "treatment", "vaccine", "disease", "epidemic", "pharmacy", "lack of facilities"],
    "Education Facilities": ["school", "college", "university", "education", "teacher", "classroom", "library", "syllabus", "fees", "exam", "admission", "infrastructure", "quality of teaching"],
    "Public Transport": ["bus", "train", "auto", "taxi", "rickshaw", "transport", "fare", "delay", "public conveyance", "station", "stop", "route", "overcrowding"],
    "Housing & Land": ["housing", "shelter", "land", "encroachment", "slum", "property", "rent", "homeless", "building collapse", "illegal construction", "zoning"],
    "Food & Public Distribution": ["food", "ration", "supply", "distribution", "hunger", "adulteration", "price hike", "storage", "quality", "availability", "PDS"],
    "Parks & Recreation": ["park", "garden", "playground", "recreation", "public space", "maintenance", "sports complex", "community center", "open gym"],
    "Digital & IT Infrastructure": ["internet", "network", "broadband", "cyber", "digital", "connectivity", "online service", "signal", "tower", "mobile network", "e-governance"],
    "Disaster Preparedness & Response": ["flood", "cyclone", "earthquake", "disaster", "relief", "rescue", "emergency", "evacuation", "rehabilitation", "warning system"],
    "Social Welfare & Equity": ["welfare", "equity", "discrimination", "rights", "senior citizen", "child labor", "poverty", "unemployment", "gender equality", "disability", "social security"],
    "Administrative & Governance": ["corruption", "bureaucracy", "government", "administration", "permit", "license", "red tape", "accountability", "transparency", "public service", "grievance"],
    "Rural & Agriculture": ["farm", "crop", "irrigation", "fertilizer", "livestock", "harvest", "farmer", "agriculture", "loan", "seed", "mandis", "storage", "rural development", "subsidy", "animal husbandry"],
    "Other Civic Issue": [] # Default if not matched
}

def detect_issue(text: str) -> dict:
    cleaned_text = text.lower()
    
    # Prioritize specific keywords first for higher confidence
    for category, keywords in PDF_ISSUE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in cleaned_text:
                # Assign confidence based on keyword match strength / specificity
                confidence = 80 if keyword in ["pothole", "sewage", "power outage", "traffic jam", "water logging", "corruption", "theft", "flood", "hospital"] else 70
                explanation = f"Detected '{keyword}' matching '{category}' category."
                return {
                    "is_issue": True,
                    "issue_type": category,
                    "confidence": confidence,
                    "explanation": explanation
                }
    
    # Fallback for generic issue phrases
    if any(phrase in cleaned_text for phrase in ["problem with", "issue with", "facing difficulty with", "bad condition", "no action", "complaint about", "urgent need", "demand for"]):
        return {
            "is_issue": True,
            "issue_type": "Other Civic Issue",
            "confidence": 60,
            "explanation": "Generic problem phrase detected."
        }
    
    # If no issue-related keywords or phrases are found
    return {
        "is_issue": False,
        "issue_type": "None",
        "confidence": 0,
        "explanation": "No specific civic issue keywords detected."
    }

# --- Main Application Logic ---
def run_pipeline(query: str):
    db_manager = DatabaseManager()
    twitter_client = TwitterClient(TWITTER_BEARER_TOKEN)
    
    processed_tweet_ids = set() # To track unique tweet IDs for this run
    all_parent_and_reply_tweets = []
    users_info = {}
    replies_for_parent_map = {} # Map parent_id -> list of reply dicts

    try:
        db_manager.create_table()

        # Fetch parent tweets
        print(f"Fetching parent tweets for query: '{query}'...")
        parent_response = twitter_client.search_tweets(
            query=f"{query} -is:retweet lang:en",
            max_results=50, # Increased from 10 to 50
            tweet_fields=["created_at", "public_metrics", "conversation_id", "geo"],
            user_fields=["username"],
            expansions=["author_id", "geo.place_id"]
        )

        if not parent_response or not parent_response.data: # More robust check for no data
            print("No parent tweets found for the given query or API call failed. Please check your query or API token.")
            return [] # Return empty list if no tweets

        # Populate users_info from parent tweets
        if parent_response.includes and "users" in parent_response.includes:
            users_info.update({user["id"]: user for user in parent_response.includes["users"]})
        
        # Add parent tweets to our list of tweets to process
        for tweet in parent_response.data:
            if tweet.id not in processed_tweet_ids:
                all_parent_and_reply_tweets.append(tweet)
                processed_tweet_ids.add(tweet.id)

            # Fetch replies for each parent tweet
            reply_response = twitter_client.search_tweets(
                query=f"conversation_id:{tweet.conversation_id} -is:retweet -is:quote lang:en",
                max_results=10, # Limit replies
                tweet_fields=["created_at", "public_metrics", "conversation_id", "geo"],
                user_fields=["username"],
                expansions=["author_id", "geo.place_id"]
            )
            
            current_tweet_replies_list = []
            if reply_response and reply_response.data:
                if reply_response.includes and "users" in reply_response.includes:
                    users_info.update({user["id"]: user for user in reply_response.includes["users"]})

                for reply_tweet in reply_response.data:
                    # Avoid adding parent tweet again if it appears in replies
                    if reply_tweet.id not in processed_tweet_ids:
                        all_parent_and_reply_tweets.append(reply_tweet)
                        processed_tweet_ids.add(reply_tweet.id)
                    
                    reply_author = users_info.get(reply_tweet.author_id, {"username": "unknown"})
                    current_tweet_replies_list.append({
                        "id": str(reply_tweet.id),
                        "author": f"@{reply_author.get('username', 'unknown')}", # .get for safety
                        "raw_text": reply_tweet.text,
                        "text": clean_text_for_storage(reply_tweet.text),
                        "timestamp": reply_tweet.created_at.isoformat(),
                        "likes": reply_tweet.public_metrics['like_count']
                    })
            replies_for_parent_map[tweet.id] = current_tweet_replies_list


        print(f"\nProcessing and storing {len(all_parent_and_reply_tweets)} unique tweets...")
        final_processed_ids = []

        for tweet in all_parent_and_reply_tweets:
            author_info = users_info.get(tweet.author_id, {"username": "unknown"})
            raw_tweet_text = tweet.text
            cleaned_tweet_text = clean_text_for_storage(raw_tweet_text)
            
            if not cleaned_tweet_text: continue

            # --- Sentiment Analysis ---
            sentiment_result = sentiment_pipeline(cleaned_tweet_text)[0]
            sentiment_label = sentiment_result['label'].capitalize()
            sentiment_score = sentiment_result['score']

            # --- Embedding Generation ---
            tweet_embedding = embedding_model.encode(cleaned_tweet_text).tolist()

            # --- Issue Detection (PDF-aligned) ---
            issue_analysis = detect_issue(raw_tweet_text)
            confidence = issue_analysis['confidence']
            
            if confidence < 65: # Threshold as per previous discussion
                print(f"Skipping tweet (low confidence {confidence}%): {raw_tweet_text[:100]}...")
                continue
            
            print(f"✅ Tweet accepted as {issue_analysis['issue_type']} with {confidence}% confidence: {raw_tweet_text[:100]}...")

            # --- Prepare tweet_data for insertion ---
            tweet_data = {
                "id": tweet.id,
                "author": f"@{author_info.get('username', 'unknown')}",
                "timestamp": tweet.created_at,
                "raw_text": raw_tweet_text,
                "text": cleaned_tweet_text,
                "sentiment": sentiment_label,
                "score": sentiment_score,
                "likes": tweet.public_metrics['like_count'],
                "embedding": tweet_embedding,
                "replies": json.dumps(replies_for_parent_map.get(tweet.id, [])),
                "is_issue": issue_analysis['is_issue'],
                "issue_type": issue_analysis['issue_type'],
                "issue_confidence": issue_analysis['confidence'],
                "issue_explanation": issue_analysis['explanation']
            }

            # --- Handle Location Data ---
            if tweet.geo and 'coordinates' in tweet.geo and 'coordinates' in tweet.geo['coordinates']:
                coordinates = tweet.geo['coordinates']['coordinates'] # [longitude, latitude]
                tweet_data['longitude'] = coordinates[0]
                tweet_data['latitude'] = coordinates[1]
            
            db_manager.insert_tweet(tweet_data)
            final_processed_ids.append(tweet.id)

        print(f"\n✅ Success! Data has been stored and analyzed for '{query}'.")
        return final_processed_ids

    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")
        raise
    finally:
        db_manager.close()

# --- Similarity Search Function (Moved here) ---
def find_similar_tweets(conn, query_embedding, top_k: int = 5):
    cur = conn.cursor()
    # If query_embedding is a list (from the fix), use it directly.
    # Otherwise, assume it's something that needs .tolist() (like numpy array from SentenceTransformer)
    embedding_for_query = query_embedding # query_embedding is already ensured to be a list before calling this function

    sql_query = """
        SELECT text, sentiment, author, 1 - (embedding <=> %s::vector) AS similarity
        FROM tweets
        ORDER BY similarity DESC
        LIMIT %s;
    """
    cur.execute(sql_query, (embedding_for_query, top_k)) # Pass the list directly
    results = cur.fetchall()
    cur.close()
    return results

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch tweets, analyze them for civic issues, store in DB, and find relevant ones."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="A descriptive phrase about the issue (e.g., 'road construction issues in Bhubaneswar')."
    )
    
    args = parser.parse_args()
    
    _db_manager_for_search = None

    try:
        # --- Step 1: Ingestion and Analysis Pipeline ---
        print("--- Step 1: Running Data Ingestion and Issue Analysis Pipeline ---")
        processed_tweet_ids = run_pipeline(query=args.query)
        print(f"--- Ingestion and Analysis complete. Stored {len(processed_tweet_ids)} new tweets. ---\n")

        # --- Step 2: Similarity Search ---
        print(f"--- Step 2: Finding tweets most similar to: '{args.query}' ---")
        
        _db_manager_for_search = DatabaseManager()
        _db_manager_for_search.connect()
        
        # Generate embedding for the user's query
        raw_query_embedding = embedding_model.encode(clean_text_for_storage(args.query))
        
        # Ensure embedding is a list before passing to find_similar_tweets
        if isinstance(raw_query_embedding, np.ndarray): # If it's a NumPy array
            query_embedding_for_search = raw_query_embedding.tolist()
        elif isinstance(raw_query_embedding, list): # If it's already a list (e.g., if it was a single vector)
            query_embedding_for_search = raw_query_embedding
        else:
            raise TypeError("Embedding model returned an unexpected type.")

        similar_tweets = find_similar_tweets(_db_manager_for_search.conn, query_embedding_for_search)
        
        print("\n--- Top 5 Most Relevant Results (from ALL stored tweets) ---\n")
        if not similar_tweets:
            print("Could not find any highly relevant tweets in the database.")
        else:
            for i, tweet in enumerate(similar_tweets):
                text, sentiment, author, similarity_score = tweet
                print(f"{i+1}. [Similarity: {similarity_score:.4f}] [Sentiment: {sentiment}] [Author: {author}]")
                print(f"   Tweet: {text.strip()}\n")
            
    except Exception as e:
        print(f"Main execution failed: {e}")
    finally:
        if _db_manager_for_search:
            _db_manager_for_search.close()