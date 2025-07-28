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
