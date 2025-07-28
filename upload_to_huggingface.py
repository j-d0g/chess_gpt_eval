#!/usr/bin/env python3
"""
Upload Chess Language Model Evaluation Dataset to Hugging Face
"""

import os
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from pathlib import Path
import json
import argparse
from typing import Dict, Any

def create_dataset_card() -> str:
    """Create a comprehensive dataset card for the chess evaluation dataset."""
    
    dataset_card = """---
license: mit
task_categories:
- other
language:
- en
tags:
- chess
- language-models
- evaluation
- games
- stockfish
- nanogpt
size_categories:
- 10M<n<100M
---

# Chess Language Model Evaluation Dataset

This dataset contains comprehensive evaluation results of chess-playing language models, including game records and detailed Stockfish analysis.

## Dataset Description

This dataset provides a systematic evaluation of various language models (primarily NanoGPT variants) playing chess against Stockfish at different difficulty levels. It includes both raw game data and detailed move-by-move analysis.

## Dataset Structure

### Game Data (`data/games/`)
- **Raw game files**: CSV files containing complete game records for each model vs Stockfish
- **games.csv**: Consolidated game results with metadata
- **openings.csv**: Opening analysis and statistics

### Analysis Data (`data/analysis/stockfish_analysis/`)
- **Detailed move analysis**: Stockfish evaluation of each move including:
  - Centipawn loss
  - Move classifications (blunder, mistake, inaccuracy)
  - Opening, middlegame, and endgame performance
  - Game phase analysis

### Performance Summary
- **model_performance_summary.csv**: Aggregated performance metrics across all models

## Models Evaluated

The dataset includes evaluation of the following model variants:
- small-8, small-16, small-24, small-36 (different layer configurations)
- medium-12, medium-16
- large-16
- adam_lichess (trained on Lichess data)
- adam_stockfish (trained on Stockfish games)

All models were evaluated against Stockfish levels 0-9 with 1000+ games per configuration.

## Metrics Included

- **Centipawn Loss**: Average, median, and standard deviation
- **Move Quality**: Counts of blunders, mistakes, and inaccuracies
- **Phase Performance**: Accuracy in opening, middlegame, and endgame
- **Game Length**: Average number of moves per game

## Usage

This dataset can be used for:
- Analyzing language model chess performance
- Comparing different model architectures
- Studying chess move quality across different game phases
- Training improved chess evaluation systems

## Citation

If you use this dataset in your research, please cite:

```
@dataset{chess_gpt_eval_2024,
  title={Chess Language Model Evaluation Dataset},
  author={[Your Name]},
  year={2024},
  url={https://huggingface.co/datasets/[your-username]/chess-gpt-evaluation}
}
```

## Files Overview

| File | Description | Size |
|------|-------------|------|
| model_performance_summary.csv | Aggregated performance metrics | 2.5KB |
| data/games/*.csv | Raw game records by model | ~100MB total |
| data/analysis/stockfish_analysis/*.csv | Detailed move analysis | ~50MB total |
| chess_results_analysis.png | Performance visualization | 2.9MB |

## License

This dataset is released under the MIT License.
"""
    
    return dataset_card

def prepare_dataset_info() -> Dict[str, Any]:
    """Prepare dataset configuration and metadata."""
    
    # Read model performance summary to get basic stats
    try:
        perf_df = pd.read_csv('model_performance_summary.csv')
        num_models = len(perf_df)
        total_games = perf_df['games'].sum()
    except:
        num_models = 12  # fallback
        total_games = 100000  # fallback
    
    dataset_info = {
        "dataset_name": "chess-gpt-evaluation",
        "description": "Comprehensive evaluation of chess-playing language models against Stockfish",
        "num_models": num_models,
        "total_games": total_games,
        "features": [
            "game_records",
            "move_analysis", 
            "performance_metrics",
            "opening_analysis",
            "stockfish_evaluation"
        ],
        "languages": ["en"],
        "license": "mit"
    }
    
    return dataset_info

def upload_chess_dataset(repo_name: str, private: bool = False, token: str = None):
    """
    Upload the chess evaluation dataset to Hugging Face.
    
    Args:
        repo_name: Name of the repository (format: username/dataset-name)
        private: Whether to make the repository private
        token: Hugging Face token (if not set in environment)
    """
    
    api = HfApi(token=token)
    
    print(f"Creating repository: {repo_name}")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            token=token
        )
        print(f"✅ Repository created: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Repository {repo_name} already exists, continuing with upload...")
        else:
            raise e
    
    # Create and upload README
    print("Creating dataset card...")
    readme_content = create_dataset_card()
    with open("README_dataset.md", "w") as f:
        f.write(readme_content)
    
    # Upload README
    api.upload_file(
        path_or_fileobj="README_dataset.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )
    print("✅ Dataset card uploaded")
    
    # Upload main files
    files_to_upload = [
        "model_performance_summary.csv",
        "chess_results_analysis.png",
        "requirements.txt"
    ]
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            print(f"✅ {file_path} uploaded")
    
    # Upload data directories
    data_dirs = ["data/games", "data/analysis"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"Uploading directory {data_dir}...")
            api.upload_folder(
                folder_path=data_dir,
                path_in_repo=data_dir,
                repo_id=repo_name,
                repo_type="dataset",
                token=token,
                ignore_patterns=["*.log", "*.tmp", "__pycache__/*"]
            )
            print(f"✅ {data_dir} uploaded")
    
    # Create and upload dataset info
    dataset_info = prepare_dataset_info()
    with open("dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    api.upload_file(
        path_or_fileobj="dataset_info.json",
        path_in_repo="dataset_info.json",
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )
    print("✅ Dataset info uploaded")
    
    print(f"\n🎉 Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
    print("\nNext steps:")
    print("1. Visit your dataset page to verify the upload")
    print("2. Update the dataset card with any additional information")
    print("3. Consider adding example usage code")
    
    # Cleanup temporary files
    for temp_file in ["README_dataset.md", "dataset_info.json"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description="Upload chess evaluation dataset to Hugging Face")
    parser.add_argument("repo_name", help="Repository name (format: username/dataset-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--token", help="Hugging Face token (optional if set in environment)")
    
    args = parser.parse_args()
    
    # Validate repository name format
    if "/" not in args.repo_name:
        print("❌ Repository name must be in format: username/dataset-name")
        return
    
    try:
        upload_chess_dataset(args.repo_name, args.private, args.token)
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify the repository name format: username/dataset-name")

if __name__ == "__main__":
    main() 