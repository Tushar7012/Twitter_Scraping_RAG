import argparse
from public_issue_pipeline.pipelines.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    # Set up argument parser to accept a keyword from the command line
    parser = argparse.ArgumentParser(description="Fetch tweets based on a keyword.")
    parser.add_argument("--keyword", type=str, required=True, help="The keyword or phrase to search for on Twitter.")
    
    args = parser.parse_args()

    try:
        pipeline = PredictionPipeline()
        pipeline.run(keyword=args.keyword)
    except Exception as e:
        print(f"Execution failed: {e}")