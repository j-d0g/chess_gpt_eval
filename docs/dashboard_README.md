# Interactive Chess Analysis Dashboard

**Interactive web-based dashboard** for comprehensive chess model evaluation results. For project overview and data generation, see the [main README](../README.md).

An advanced analysis dashboard featuring 8 analysis modes, dynamic filtering, and deep granular insights for chess-playing language models.

## Overview

The Interactive Chess Analysis Dashboard (`src/visualization/interactive_dashboard.py`) provides deep insights into chess LLM performance through:

- **8 Analysis Modes**: Overview, Move Buckets, Performance Heatmaps, Learning Patterns, Game Phases, Move Quality, Advanced Comparisons, and Position Complexity
- **Dynamic Move Grouping**: Configurable move grouping (n=1 to 20) with real-time performance trends
- **Interactive Filtering**: Move range filtering, centipawn loss thresholds, and model selection
- **Statistical Analysis**: T-tests, effect size calculations, and significance testing
- **Real-time Visualization**: Plotly-based interactive charts and heatmaps

## Quick Start

### Launch the Dashboard
```bash
# Using the launcher (recommended)
python launch_dashboards.py interactive

# Direct execution
python src/visualization/interactive_dashboard.py

# Access at: http://localhost:8050
```

### Prerequisites
- Analysis data in `data/analysis/` directory with `*_summary_*.csv` and `*_moves_*.csv` files
- Python dependencies from `requirements.txt`

## Dashboard Features

### 🎯 Move Bucket Analysis (Core Feature)
- **Dynamic Move Grouping**: Group moves by n=1 to 20 for trend analysis
- **Move Bucket Heatmap**: Performance across 6 game phases (early opening, late opening, early middle, late middle, early end, late end)
- **Variance Analysis**: Compare individual vs grouped move performance
- **Centipawn Loss Distribution**: Histograms by move buckets with customizable thresholds

### 🔥 Performance Heatmaps
- **Multi-Metric Heatmap**: CP loss, blunders, mistakes, accuracy across game phases
- **Move Quality Heatmap**: Best/good/suboptimal move distribution by model

### 📈 Learning Patterns
- **Performance Evolution**: Rolling average trends over games
- **Consistency Analysis**: Performance vs consistency scatter plots with coefficient of variation

### 🎲 Game Phase Deep-Dive
- **Phase Performance Comparison**: Opening, middlegame, endgame analysis
- **Complexity vs Performance**: Position complexity impact by game phase

### ⚡ Move Quality Analysis
- **Quality Distribution**: Percentage of best, good, suboptimal, mistake, blunder moves
- **Quality Score Correlation**: Move quality scores vs centipawn loss

### 🧪 Advanced Comparisons
- **Statistical Significance Testing**: T-tests between models with p-values and effect sizes
- **Performance Rankings**: Automated statistical comparisons

### 🎨 Position Complexity Analysis
- **Complexity Scatter Plots**: Position complexity vs performance correlation
- **Complexity Distribution**: Histogram analysis of position difficulty

### 📊 Overview Dashboard
- **Performance Summary**: Box plots, move quality bars, and key metrics
- **Enhanced Summary Table**: Comprehensive statistics with 10+ metrics per model

## Interactive Controls

### Model Selection
- **Multi-select dropdown**: Choose from available models with clean names
- **Automatic filtering**: Only models with valid data are shown

### Dynamic Filtering
- **Move Range Slider**: Filter analysis to specific move ranges (1-100)
- **Move Grouping Size**: Adjust grouping from individual moves (n=1) to larger buckets (n=20)
- **Centipawn Loss Threshold**: Set maximum CP loss for analysis focus

### Analysis Type Selection
8 radio button options for different analysis modes:
- Overview Dashboard
- Move Bucket Analysis  
- Performance Heatmaps
- Learning Patterns
- Game Phase Deep-Dive
- Move Quality Analysis
- Advanced Comparisons
- Position Complexity

## Data Requirements

### Input Files
```bash
data/analysis/
├── {model}_summary_{timestamp}.csv     # Game-level statistics
└── {model}_moves_{timestamp}.csv       # Move-by-move analysis (sampled to 50k moves)
```

### Required Columns

**Summary Files:**
- `average_centipawn_loss`, `blunders`, `mistakes`, `inaccuracies`
- `best_moves`, `good_moves`, `suboptimal_moves`
- `opening_accuracy`, `middlegame_accuracy`, `endgame_accuracy`

**Moves Files:**
- `move_number`, `move_quality`, `centipawn_loss`, `game_phase`
- `position_complexity`, `move_quality_score`

## Usage Examples

### Basic Analysis
```python
from src.visualization.interactive_dashboard import AdvancedChessAnalysisDashboard

# Initialize dashboard
dashboard = AdvancedChessAnalysisDashboard()

# Run dashboard (opens in browser)
dashboard.run(port=8050)
```

### Custom Data Directories
```python
# Custom data paths
dashboard = AdvancedChessAnalysisDashboard(
    results_dir='custom/path/analysis',
    moves_dir='custom/path/analysis'
)
dashboard.run()
```

## Performance Features

### Data Loading
- **Intelligent Sampling**: Loads first 50k moves per model for performance
- **Data Validation**: Checks for required columns before loading
- **Error Handling**: Graceful handling of missing or corrupted files

### Statistical Analysis
- **Move Bucket Statistics**: Performance across 6 predefined game phases
- **Enhanced Statistics**: Calculates 20+ metrics per model
- **Real-time Computation**: Dynamic recalculation based on filters

### Visualization Performance
- **Efficient Plotting**: Optimized Plotly charts with hover details
- **Responsive UI**: Real-time updates based on control changes
- **Memory Management**: Smart data sampling for large datasets

## Dashboard Architecture

### Class Structure
```python
class AdvancedChessAnalysisDashboard:
    def load_data()                           # Load and validate analysis data
    def calculate_enhanced_stats()            # Compute model statistics
    def setup_layout()                        # Define dashboard UI
    def setup_callbacks()                     # Handle interactivity
    
    # Analysis Functions (8 modes)
    def create_move_bucket_analysis()         # Core move grouping feature
    def create_performance_heatmaps()         # Multi-metric heatmaps
    def create_learning_patterns()            # Trend analysis
    def create_game_phase_analysis()          # Phase-specific insights
    def create_move_quality_analysis()        # Move classification
    def create_advanced_comparisons()         # Statistical testing
    def create_position_complexity_analysis() # Complexity correlation
    def create_overview_dashboard()           # Summary view
```

### Callback System
- **Real-time Updates**: All visualizations update dynamically
- **Input Validation**: Handles missing data gracefully
- **Performance Optimization**: Efficient data filtering and processing

## Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check data availability
ls data/analysis/*_summary_*.csv
ls data/analysis/*_moves_*.csv
```

**Missing models:**
- Ensure CSV files follow naming convention: `{model}_vs_stockfish_*`
- Check that files contain required columns

**Performance issues:**
- Dashboard samples moves data (50k per model) for performance
- Reduce move range or number of selected models if needed

### Data Format Issues
```bash
# Check file format
head -5 data/analysis/model_summary_timestamp.csv
head -5 data/analysis/model_moves_timestamp.csv
```

## Advanced Features

### Move Bucket Analysis Details
The core feature provides granular analysis of move performance:

1. **Dynamic Grouping**: Groups moves by configurable size (1-20)
2. **Performance Trends**: Shows how grouping affects variance
3. **Phase Mapping**: Maps moves to 6 game phases automatically
4. **Statistical Depth**: Provides mean, variance, and distribution analysis

### Integration with Analysis Pipeline
The dashboard automatically loads data from the Stockfish analysis pipeline:
```bash
# Generate data first
python src/analysis/mass_stockfish_processor.py --input-dir data/games

# Then launch dashboard
python src/visualization/interactive_dashboard.py
```

---

**Ready to explore your chess model performance with advanced interactive analysis!** 🚀 