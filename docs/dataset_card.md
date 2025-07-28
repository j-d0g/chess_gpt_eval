---
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
configs:
- config_name: default
  data_files:
  - split: train
    path: "**/*.csv"
---

# Chess GPT Evaluation Dataset

Comprehensive evaluation of chess-playing language models from [jd0g/chess-gpt](https://huggingface.co/jd0g/chess-gpt).

## Dataset Overview

- **Size**: ~12GB of chess evaluation data
- **Models**: 12 language model variants  
- **Games**: 100,000+ with detailed Stockfish analysis
- **Structure**: Game records + move-by-move analysis

![Chess Analysis Results](chess_results_analysis.png)

## Quick Start

```python
from huggingface_hub import hf_hub_download
import pandas as pd

# Load model performance summary
summary = pd.read_csv("model_performance_summary.csv")

# Load games for a specific model
games_file = hf_hub_download(
    repo_id="jd0g/chess-language-model-evaluation",
    filename="games/small-16-600k_iters_pt_vs_stockfish_sweep.csv"
)
games = pd.read_csv(games_file)
```

## Dataset Structure

### Root Files
- `chess_results_analysis.png` - Performance visualization
- `model_performance_summary.csv` - Model comparison metrics

### Directories
- `games/` - Chess game records (CSV files with PGN transcripts)
- `analysis/` - Stockfish analysis (summaries, detailed JSON, move data)

## Models Evaluated

From [jd0g/chess-gpt](https://huggingface.co/jd0g/chess-gpt):
- **Architecture variants**: small-8/16/24/36, medium-12/16, large-16
- **Training variants**: adam_stockfish, adam_lichess (different training data)

All models evaluated against Stockfish levels 0-9.

## Citation

```bibtex
@dataset{chess_gpt_eval_2025,
  title={Chess GPT Evaluation Dataset},
  url={https://huggingface.co/datasets/jd0g/chess-language-model-evaluation},
  note={Models: https://huggingface.co/jd0g/chess-gpt}
}
``` 