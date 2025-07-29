# Enhanced Chess Language Model Evaluation Suite

A streamlined, comprehensive system for evaluating chess-playing language models with two main workflows: **Game Playing** and **Stockfish Analysis**. Optimized for high-performance computing clusters like CSF3.

## 🎯 **Two Main Workflows**

### 1. **🎮 Game Playing Workflow**
Play NanoGPT models against Stockfish (levels 0-9) and record comprehensive game data.

### 2. **📊 Stockfish Analysis Workflow**  
Analyze recorded games with Stockfish engine to extract move quality, centipawn loss, and tactical insights.

## 🚀 **Quick Start**

### **Workflow 1: Game Playing**
```bash
# Direct execution
python src/models/main.py large-16 0 10 1000
```

### **Workflow 2: Stockfish Analysis**
```bash
# Run comprehensive analysis (recommended)
sbatch scripts/run_stockfish_analysis.sh data/games 64 150000

# Direct execution
python src/analysis/mass_stockfish_processor.py --input-dir data/games --workers 64
```

### **View Results**
```bash
# Chess results visualization
python src/visualization/visualize_chess_results.py

# Interactive analysis dashboard (recommended)
python launch_dashboards.py interactive

# Or direct execution
python src/visualization/interactive_dashboard.py
```

## 📁 **Streamlined Repository Structure**

```
chess_gpt_eval/
├── 🎮 WORKFLOW 1: GAME PLAYING
│   ├── src/models/
│   │   ├── main.py                     # Main game playing engine
│   │   ├── nanogpt/                    # NanoGPT model implementation
│   │   └── stockfish/                  # Stockfish engine binaries
│   └── data/games/                     # Game result CSV files
│
├── 📊 WORKFLOW 2: STOCKFISH ANALYSIS
│   ├── src/analysis/
│   │   ├── mass_stockfish_processor.py # Batch analysis processor
│   │   ├── stockfish_analysis.py       # Single game analysis
│   │   └── generate_analysis_summary.py # Analysis summaries
│   ├── scripts/
│   │   └── run_stockfish_analysis.sh   # Stockfish analysis script
│   └── data/analysis/                  # Analysis output files
│
├── 📈 VISUALIZATION & REPORTING
│   ├── src/visualization/
│   │   ├── visualize_chess_results.py  # Chess results visualization
│   │   └── interactive_dashboard.py    # Interactive web dashboard
│   └── scripts/
│       └── launch_analysis_dashboard.sh # Dashboard launcher script
│
├── 🔧 INFRASTRUCTURE
│   ├── docs/                           # Documentation
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # This file
│
└── 🗂️ ARCHIVE
    ├── experimental/                   # Experimental scripts
    ├── legacy/                         # Legacy implementations
    └── visualization/                  # Archived visualization tools
```

## 🌟 **Enhanced Features**

### **Chess Results Analysis**
- **Comprehensive visualizations** of model performance
- **Elo estimation** based on Stockfish performance
- **Opening analysis** with categorization
- **Time pressure analysis** (performance degradation)
- **Move-bucket performance analysis**
- **Consistency metrics** and phase specialists
- **Architecture impact analysis**

### **Interactive Analysis Dashboard**
- **8 analysis modes**: Overview, Move Buckets, Performance Heatmaps, Learning Patterns, Game Phases, Move Quality, Advanced Comparisons, Position Complexity
- **Dynamic move grouping** with configurable grouping sizes (n=1-20)
- **Interactive filtering** by move range, centipawn loss thresholds, and model selection
- **Statistical analysis** with t-tests, effect sizes, and significance testing
- **Real-time visualization** updates with Plotly charts

### **Superior Analysis Capabilities**
- **Comprehensive metrics**: 25+ statistics per game
- **Move classifications**: Best, Good, Suboptimal, Inaccuracies, Mistakes, Blunders
- **Position complexity scoring** (0-100 scale)
- **Game phase detection**: Opening, Middlegame, Endgame
- **Performance correlation analysis**
- **Error pattern identification**

## 🔧 **Detailed Usage**

### **Game Playing Workflow**

#### **Basic Usage**
```bash
# Play games directly (adjust parameters as needed)
python src/models/main.py large-16 0 9 1000

# Quick test (100 games, levels 0-2) 
python src/models/main.py small-8 0 3 100
```

#### **Output Files**
```bash
data/games/
├── large-16-600k_iters_pt_vs_stockfish_sweep.csv
├── medium-16-600k_iters_pt_vs_stockfish_sweep.csv
└── small-8-600k_iters_pt_vs_stockfish_sweep.csv
```

### **Stockfish Analysis Workflow**

#### **Basic Usage**
```bash
# Comprehensive analysis (all games, 64 workers)
sbatch scripts/run_stockfish_analysis.sh data/games 64 150000

# Quick analysis (4 workers, 1000 nodes)
python src/analysis/mass_stockfish_processor.py --workers 4 --nodes 1000 --max-games 100
```

#### **Output Files**
```bash
data/analysis/
├── {model}_summary_{timestamp}.csv      # Game-level statistics
├── {model}_moves_{timestamp}.csv        # Move-by-move analysis
└── {model}_detailed_{timestamp}.json    # Complete analysis data
```

### **Visualization & Analysis**

#### **Chess Results Visualization**
```bash
# Generate comprehensive analysis plots
python src/visualization/visualize_chess_results.py

# Output: Analysis plots and visualizations
```

#### **Interactive Dashboard**
```bash
# Launch web dashboard (recommended)
python launch_dashboards.py interactive

# Direct execution
python src/visualization/interactive_dashboard.py

# Access at: http://localhost:8050
```

## 📊 **Analysis Capabilities**

### **Game Playing Analysis**
- **Performance vs Stockfish levels** with error bars and Elo estimates
- **Opening repertoire analysis** with success rates by category
- **Time pressure effects** (short vs long games)
- **Architecture impact** (model size vs performance)
- **Move-bucket analysis** (performance by game phase)
- **Consistency scoring** and performance variance

### **Stockfish Analysis**
- **Move quality distribution** (Best/Good/Suboptimal/Inaccuracies/Mistakes/Blunders)
- **Centipawn loss analysis** with distributions and trends
- **Position complexity scoring** (0-100 scale)
- **Game phase performance** (Opening/Middlegame/Endgame accuracy)
- **Error pattern identification** and correlation analysis
- **Critical moment detection** and tactical evaluation

### **Comparative Analysis**
- **Model rankings** by estimated Elo
- **Performance correlation matrices**
- **Strength vs consistency scatter plots**
- **Improvement potential analysis**
- **Phase specialist identification**
- **Architecture optimization insights**

## 🎯 **Key Improvements**

### **Current Features**
- ✅ **Streamlined structure** - Clear separation of two workflows
- ✅ **Comprehensive visualizations** - Multiple analysis plots and charts
- ✅ **Elo estimation** - Proper chess rating estimates
- ✅ **Opening analysis** - Categorized opening performance
- ✅ **Interactive dashboard** - 8 analysis modes with advanced filtering
- ✅ **Consistency metrics** - Performance variance analysis
- ✅ **Move-bucket analysis** - Dynamic performance grouping
- ✅ **Statistical testing** - T-tests and significance analysis
- ✅ **Real-time filtering** - Interactive data exploration

### **Archived Components**
- 🗂️ **Experimental scripts** moved to `archive/experimental/`
- 🗂️ **Legacy implementations** moved to `archive/legacy/`
- 🗂️ **Legacy visualization tools** moved to `archive/visualization/`
- 🗂️ **Generated outputs** moved to `archive/outputs/`

## 🚀 **Performance Expectations**

### **Game Playing**
- **Throughput**: 5-15 games/second per job
- **1000 games**: 2-10 minutes depending on model
- **Memory usage**: 2-5GB per job

### **Stockfish Analysis**
- **Throughput**: 50-100 games/minute with 64 workers
- **10,000 games**: 2-4 hours
- **80,000 games**: 15-30 hours
- **Memory usage**: 8GB per worker

## 🛠️ **Setup & Installation**

### **Prerequisites**
```bash
# On CSF3 (University of Manchester)
module load apps/binapps/anaconda3/2023.09
module load tools/gcc/11.2.0

# Other systems
# module load apps/anaconda3/2022.10

# Install dependencies
pip install -r requirements.txt
```

### **Download Dataset**
```bash
# Download complete dataset from HuggingFace (11GB)
python data/download_data.py

# Download only specific subsets
python data/download_data.py --subset games    # Game CSV files only
python data/download_data.py --subset analysis # Analysis results only

# See all options
python data/download_data.py --help
```

### **Stockfish Setup**
```bash
# Stockfish binaries included in src/models/stockfish/ directory
# Verify installation
./src/models/stockfish/stockfish-ubuntu-x86-64-avx2

# For automated setup, see docs/stockfish.md
```

### **First Run**
```bash
# Test game playing
python src/models/main.py small-8 0 3 10

# Test analysis  
python src/analysis/mass_stockfish_processor.py --max-games 5 --workers 2
```

## 📚 **Documentation**

**Getting Started:**
- **[CSF3 Setup Guide](docs/README_CSF3_Setup.md)** - University of Manchester CSF3 cluster setup
- **[CSF3 Resource Limits](docs/CSF3_Resource_Limits.md)** - Resource optimization for CSF3

**Analysis & Visualization:**
- **[Dashboard Guide](docs/dashboard_README.md)** - Interactive visualization and dashboard usage  
- **[Stockfish Features](docs/stockfish_missing_features.md)** - Detailed analysis capabilities
- **[Stockfish Setup Guide](docs/setup_stockfish.md)** - Installing and configuring Stockfish engine

**For CSF3 Users:** See **[docs/README.md](docs/README.md)** for CSF3-specific workflows and optimization.

## 🔍 **Example Workflows**

### **Complete Analysis Pipeline**
    ```bash
# 1. Generate games
python src/models/main.py large-16 0 9 1000

# 2. Analyze games
sbatch scripts/run_stockfish_analysis.sh data/games 64 150000

# 3. Generate visualizations
python src/visualization/visualize_chess_results.py

# 4. Launch dashboard
python src/visualization/interactive_dashboard.py
```

### **Quick Model Comparison**
```bash
# Generate games for multiple models
for model in small-8 medium-12 large-16; do
    python src/models/main.py $model 0 5 500
done

# Analyze all results
sbatch scripts/run_stockfish_analysis.sh data/games 32 100000

# View comparison
python src/visualization/interactive_dashboard.py
```

## 🎯 **Model Performance Rankings**

Based on comprehensive analysis of 80,000+ games:

1. **large-16-600k_iters** - ~1548 Elo (Best overall)
2. **medium-16-600k_iters** - ~1527 Elo (Best consistency)
3. **medium-12-600k_iters** - ~1482 Elo (Balanced performance)
4. **small-16-600k_iters** - ~1469 Elo (Good efficiency)
5. **small-8-600k_iters** - ~1377 Elo (Baseline)

## 📄 **License**

This project builds upon the original chess evaluation framework and adds comprehensive analysis capabilities with streamlined workflows.

---

**Ready to evaluate your chess models with enhanced analysis!** 🚀 