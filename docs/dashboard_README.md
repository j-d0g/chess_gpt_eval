# Chess GPT Analysis Dashboard & Benchmarking Suite

**Interactive visualization and benchmarking guide** for chess model evaluation results. For project overview and data generation, see the [main README](../README.md).

A comprehensive analysis and benchmarking framework for evaluating chess-playing language models (NanoGPT and similar architectures).

## Overview

This suite provides deep insights into chess LLM performance through:
- **Performance Analysis**: Win rates, Elo estimation, and performance vs different Stockfish levels
- **Game Analysis**: Move quality, game phases, opening/endgame performance
- **Error Analysis**: Illegal move patterns, error recovery, and failure modes
- **Stockfish Integration**: Deep position analysis with centipawn loss and blunder detection
- **Standardized Benchmarks**: Comprehensive test suite for chess LLMs
- **Interactive Dashboards**: Web-based visualizations for exploring results

## Components

### 1. Advanced Chess Dashboard (`advanced_chess_dashboard.py`)

The main analysis dashboard that creates comprehensive visualizations from game logs.

**Features:**
- Performance overview with Elo estimation
- Game length and phase analysis
- Illegal move pattern detection
- Opening performance analysis
- Model architecture comparison
- Interactive Plotly dashboards

**Usage:**
```python
from advanced_chess_dashboard import ChessAnalysisDashboard

# Initialize dashboard
dashboard = ChessAnalysisDashboard()

# Load game data
dashboard.load_data()

# Create comprehensive visualizations
dashboard.create_comprehensive_dashboard()

# Export data for mechanistic interpretability
dashboard.export_for_mechanistic_interpretability("output_dir")
```

### 2. Stockfish Analysis (`stockfish_analysis.py`)

Deep game analysis using the Stockfish chess engine.

**Features:**
- Move-by-move analysis with centipawn loss
- Blunder, mistake, and inaccuracy detection
- Position complexity scoring
- Critical moment identification
- Game phase performance breakdown

**Usage:**
```python
from stockfish_analysis import StockfishAnalyzer

# Initialize analyzer
analyzer = StockfishAnalyzer(depth=20, threads=4)

# Analyze single game
game_analysis = analyzer.analyze_game(pgn_string, game_id)

# Batch analysis
analysis_df = analyzer.batch_analyze_games(
    games_df, 
    'output.csv',
    num_processes=4
)

# Create visualizations
analyzer.create_analysis_visualizations(analysis_df)
```

### 3. Chess LLM Benchmark (`chess_llm_benchmark.py`)

Standardized benchmark suite for evaluating chess language models.

**Features:**
- 10 categories of chess tasks (tactics, strategy, endgames, etc.)
- 4 difficulty levels (easy to expert)
- Automated Elo estimation
- Comprehensive performance reports
- Leaderboard generation

**Usage:**
```python
from chess_llm_benchmark import ChessLLMBenchmark

# Initialize benchmark
benchmark = ChessLLMBenchmark()

# Export benchmark tasks
benchmark.export_benchmark_tasks()

# Evaluate model
performance = benchmark.evaluate_model(model_name, model_responses)

# Create report
benchmark.create_benchmark_report([performance1, performance2, ...])
```

## Installation

1. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn plotly chess python-chess tqdm
```

2. Install Stockfish (for deep analysis):
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# macOS
brew install stockfish

# Or download from https://stockfishchess.org/download/
```

## Data Format

The tools expect CSV files with the following columns:
- `game_id`: Unique game identifier
- `transcript`: PGN format game moves (e.g., "1.e4 e5 2.Nf3 Nc6...")
- `result`: Game result ("1-0", "0-1", "1/2-1/2")
- `player_one`: Model name
- `player_two`: Opponent (e.g., "Stockfish 5")
- `player_one_illegal_moves`: Count of illegal moves
- `number_of_moves`: Total moves in game
- Additional columns as needed

## Visualization Examples

### Performance Overview
- Model comparison across Stockfish levels
- Elo rating estimates
- Win/draw/loss distributions
- Architecture impact analysis

### Game Analysis
- Move quality over time
- Opening repertoire analysis
- Time pressure performance
- Critical position identification

### Error Analysis
- Illegal move patterns by game phase
- Error recovery rates
- Consecutive error analysis
- Error type distribution

## Advanced Features

### 1. Mechanistic Interpretability Export
Export game states for training linear probes:
```python
dashboard.export_for_mechanistic_interpretability("output_dir")
```

### 2. Custom Benchmark Tasks
Add your own benchmark positions:
```python
benchmark.tasks.append(BenchmarkTask(
    task_id='custom_001',
    category='tactics',
    difficulty='medium',
    position_fen='...',
    correct_moves=['Nxf7'],
    description='Custom tactical puzzle',
    evaluation_criteria={'move_found': 1.0}
))
```

### 3. Stockfish Annotations for Specific Games
Annotate games with detailed engine analysis:
```python
# Analyze specific games
game_ids = ['game_001', 'game_002']
annotated_games = analyzer.batch_analyze_games(
    games_df[games_df['game_id'].isin(game_ids)],
    'annotated_games.csv'
)
```

## Output Files

The suite generates various output files:

### Dashboard Outputs
- `performance_overview_[timestamp].png`: Main performance visualizations
- `game_analysis_[timestamp].png`: Detailed game analysis
- `error_analysis_[timestamp].png`: Error pattern analysis
- `opening_analysis_[timestamp].png`: Opening performance
- `interactive_dashboard_[timestamp].html`: Interactive web dashboard
- `comparison_report_[timestamp].md`: Detailed markdown report

### Stockfish Analysis Outputs
- `stockfish_analysis_results.csv`: Summary statistics
- `stockfish_analysis_results_detailed.json`: Move-by-move analysis
- `centipawn_loss_distribution.png`: CPL visualization
- `phase_performance.png`: Performance by game phase

### Benchmark Outputs
- `benchmark_tasks.json`: All benchmark positions
- `overall_comparison_[timestamp].png`: Model comparison
- `leaderboard_[timestamp].png`: Visual leaderboard
- `detailed_report_[timestamp].md`: Comprehensive report
- `raw_results_[timestamp].json`: Raw benchmark data

## Best Practices

1. **Data Quality**: Ensure game transcripts are properly formatted PGN
2. **Sample Size**: Use at least 1000 games per model for reliable statistics
3. **Stockfish Analysis**: Adjust depth based on time/accuracy tradeoff
4. **Benchmarking**: Run benchmarks multiple times to account for variance

## Future Enhancements

Potential areas for expansion:
- Real-time game analysis during play
- Integration with chess GUI (lichess/chess.com style)
- Neural network probe training pipeline
- Automated hyperparameter optimization
- Multi-model ensemble analysis
- Opening book generation from model games
- Endgame tablebase comparison
- Style analysis (aggressive/positional/tactical)

## Citation

If you use this analysis suite in your research, please cite:
```
@software{chess_gpt_analysis,
  title={Chess GPT Analysis Dashboard & Benchmarking Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/chess_gpt_eval}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Adam Karvonen's Chess-GPT work
- Uses the python-chess library for game handling
- Stockfish chess engine for position analysis 