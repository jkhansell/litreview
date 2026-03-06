import os
import pandas as pd
import argparse
from litreview.analysis import unify_databases, clean_dataframe, run_classification_pipeline
from litreview.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Run literature review classification pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    paths = config.get("paths", {})
    raw_dir = paths.get("raw_data", "data/raw")
    processed_dir = paths.get("processed_data", "data/processed")
    
    sources = config.get("sources", [])
    models = config.get("models", [])
    labels_map = config.get("labels", {})

    # 1. Unify
    print(f"Unifying databases from {raw_dir}...")
    df = unify_databases(sources, data_dir=raw_dir)
    if df.empty:
        print(f"No data found in {raw_dir}. Please check your config or data files.")
        return

    # 2. Clean
    print("Cleaning data...")
    df = clean_dataframe(df)

    # 3. Classify optimized
    print("Running optimized classification pipeline...")
    df = run_classification_pipeline(df, models, labels_map)

    # 4. Save
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "classified.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
