import argparse
import psycopg2
from public_issue_pipeline.pipelines.prediction_pipeline import PredictionPipeline
from public_issue_pipeline.components.embedding_generation import EmbeddingGenerator
from public_issue_pipeline.constants import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

# --- FIX: Modify function to accept a connection object ---
def find_similar_tweets(conn, query_embedding, top_k: int = 5):
    """
    Finds the most similar tweets using an existing database connection.
    """
    cur = conn.cursor()
    
    sql_query = """
        SELECT text, sentiment, 1 - (embedding <=> %s::vector) AS similarity
        FROM tweets
        WHERE id IN (SELECT id FROM new_tweets_session) -- Search only in the new tweets
        ORDER BY similarity DESC
        LIMIT %s;
    """
    
    cur.execute(sql_query, (query_embedding.tolist(), top_k))
    results = cur.fetchall()
    cur.close()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch tweets based on a keyword and find the most relevant ones."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="A descriptive phrase about the issue (e.g., 'road construction issues in Bhubaneswar')."
    )
    
    args = parser.parse_args()
    conn = None # Initialize connection object
    try:
        # --- Step 1: Ingestion Pipeline ---
        print("--- Step 1: Fetching and storing relevant tweets ---")
        pipeline = PredictionPipeline()
        processed_tweet_ids = pipeline.run(keyword=args.query)
        print(f"--- Ingestion complete. Stored {len(processed_tweet_ids)} new tweets. ---\n")

        # --- Step 2: Similarity Search ---
        if processed_tweet_ids:
            print(f"--- Step 2: Finding tweets most similar to: '{args.query}' ---")
            
            # --- FIX: Create a single connection to be shared ---
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
            )
            cur = conn.cursor()
            
            # Create and populate the temporary table on this connection
            cur.execute("CREATE TEMP TABLE new_tweets_session (id BIGINT);")
            cur.executemany("INSERT INTO new_tweets_session (id) VALUES (%s);", [(id,) for id in processed_tweet_ids])
            conn.commit()

            # Generate embedding for the user's query
            embed_generator = EmbeddingGenerator()
            query_embedding = embed_generator.generate_embedding(args.query)

            # --- FIX: Pass the existing connection to the function ---
            similar_tweets = find_similar_tweets(conn, query_embedding)
            
            print("\n--- Top 5 Most Relevant Results ---\n")
            if not similar_tweets:
                print("Could not find any highly relevant tweets in the fetched batch.")
            else:
                for i, tweet in enumerate(similar_tweets):
                    text, sentiment, similarity_score = tweet
                    print(f"{i+1}. [Similarity: {similarity_score:.4f}] [Sentiment: {sentiment}]")
                    print(f"   Tweet: {text.strip()}\n")
            
            cur.close()

    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        if conn:
            conn.close()