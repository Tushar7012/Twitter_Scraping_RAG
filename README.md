# Public Issue Tweet Analysis Pipeline

This project is an end-to-end data pipeline that fetches public tweets related to civic issues, performs sentiment analysis, generates vector embeddings, and stores the enriched data in a PostgreSQL database for similarity search.

---

## Key Features

* **Data Ingestion**: Fetches recent tweets from the X (Twitter) API v2 based on specific keywords.
* **Text Processing**: Cleans tweet text by removing URLs, mentions, and stopwords.
* **Sentiment Analysis**: Assigns a sentiment score (Positive, Negative, Neutral) to each tweet using VADER.
* **Vector Embeddings**: Generates semantic embeddings for cleaned text using Sentence-Transformers.
* **RAG Storage**: Stores all processed data, including the 384-dimensional embedding vectors, in a PostgreSQL database with the `pgvector` extension.
* **Similarity Search**: Includes a script to query the database and find tweets that are semantically similar to a given text input.

---

## Project Structure

.
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── query.py
├── template.py
└── src
└── public_issue_pipeline
├── init.py
├── components
├── config
├── constants
├── entity
├── pipelines
└── utils


---

## Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd public-issue-pipeline
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    ```
    *Activate it:*
    * Windows: `venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    * Create a `.env` file in the root directory.
    * Add your Twitter API Bearer Token and PostgreSQL credentials:
    ```
    TWITTER_BEARER_TOKEN=Your_Long_Bearer_Token_Goes_Here
    ```
    * Update your database credentials in `src/public_issue_pipeline/constants/__init__.py`.

5.  **Install as a Local Package**
    * This step makes your `src` code importable.
    ```bash
    pip install -e .
    ```

---

## How to Run

1.  **Run the Data Ingestion Pipeline**
    * This script will fetch tweets, process them, and save them to your database.
    ```bash
    python main.py
    ```

2.  **Perform a Similarity Search**
    * After storing some data, run this script to find similar tweets based on a query.
    ```bash
    python query.py
    ```