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
# Run game evaluation (recommended)
sbatch scripts/game_playing/run_game_evaluation.sh large-16 0 9 1000

# Direct execution
python src/game_engine/main.py large-16 0 10 1000
```

### **Workflow 2: Stockfish Analysis**
```bash
# Run comprehensive analysis (recommended)
sbatch scripts/analysis/run_stockfish_analysis.sh data/games 64 150000

# Direct execution
python src/analysis/mass_stockfish_processor.py --input-dir data/games --workers 64
```

### **View Results**
```bash
# Enhanced game results visualization
python src/visualization/enhanced_game_results.py

# Interactive analysis dashboard
python src/visualization/enhanced_analysis_dashboard.py
```

## 📁 **Streamlined Repository Structure**

```
chess_gpt_eval/
├── 🎮 WORKFLOW 1: GAME PLAYING
│   ├── src/models/
│   │   ├── main.py                     # Main game playing engine
│   │   ├── nanogpt/                    # NanoGPT model implementation
│   │   └── llama/                      # LLaMA model implementation
│   ├── scripts/game_playing/
│   │   └── run_game_evaluation.sh      # Unified game playing script
│   └── data/games/                     # Game result CSV files
│
├── 📊 WORKFLOW 2: STOCKFISH ANALYSIS
│   ├── src/analysis/
│   │   ├── mass_stockfish_processor.py # Batch analysis processor
│   │   └── stockfish_analysis.py       # Single game analysis
│   ├── scripts/analysis/
│   │   └── run_stockfish_analysis.sh   # Unified analysis script
│   └── data/analysis/                  # Analysis output files
│
├── 📈 VISUALIZATION & REPORTING
│   ├── src/visualization/
│   │   ├── enhanced_game_results.py    # 20-plot comprehensive analysis
│   │   ├── enhanced_analysis_dashboard.py # Interactive web dashboard
│   │   ├── generate_analysis_summary.py   # Analysis summaries
│   │   └── model_comparative_analysis.py  # Model comparisons
│
├── 🔧 INFRASTRUCTURE
│   ├── engines/                        # Stockfish engine binaries
│   ├── docs/                           # Documentation
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # This file
│
└── 🗂️ ARCHIVE
    ├── experimental/                   # Experimental scripts
    ├── legacy/                         # Legacy implementations
    └── outputs/                        # Generated output files
```

## 🌟 **Enhanced Features**

### **Enhanced Game Results Analysis**
- **20 comprehensive visualizations** (vs. previous 16)
- **Elo estimation** based on Stockfish performance
- **Opening analysis** with categorization
- **Time pressure analysis** (performance degradation)
- **Move-bucket advantage heatmaps** (NEW!)
- **Consistency metrics** and phase specialists
- **Architecture impact analysis**

### **Enhanced Analysis Dashboard**
- **5 analysis modes**: Overview, Move Quality, Game Phases, Error Patterns, Comparative
- **Interactive filtering** by game phase, metrics, confidence intervals
- **Intelligent insights generation** with automated recommendations
- **Enhanced summary tables** with 10+ metrics
- **Real-time visualization** updates

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
# Play 1000 games per Stockfish level (0-9)
sbatch scripts/game_playing/run_game_evaluation.sh large-16 0 9 1000

# Quick test (100 games, levels 0-2)
sbatch scripts/game_playing/run_game_evaluation.sh small-8 0 3 100
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
sbatch scripts/analysis/run_stockfish_analysis.sh data/games 64 150000

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

#### **Enhanced Game Results**
```bash
# Generate 20-plot comprehensive analysis
python src/visualization/enhanced_game_results.py

# Output: enhanced_chess_analysis.png (24x30 inches, 300 DPI)
```

#### **Interactive Dashboard**
```bash
# Launch web dashboard
python src/visualization/enhanced_analysis_dashboard.py

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

### **From Previous Version**
- ✅ **Streamlined structure** - Clear separation of two workflows
- ✅ **Enhanced visualizations** - 20 plots vs 16, with superior analysis
- ✅ **Elo estimation** - Proper chess rating estimates
- ✅ **Opening analysis** - Categorized opening performance
- ✅ **Interactive dashboard** - 5 analysis modes with intelligent insights
- ✅ **Consistency metrics** - Performance variance analysis
- ✅ **Move-bucket heatmaps** - Performance by game phase
- ✅ **Unified entry scripts** - Simple workflow execution
- ✅ **Intelligent insights** - Automated recommendations

### **Archived Features**
- 🗂️ **Experimental scripts** moved to `archive/experimental/`
- 🗂️ **Legacy implementations** moved to `archive/legacy/`
- 🗂️ **Generated outputs** moved to `archive/outputs/`
- 🗂️ **Utility notebooks** moved to `archive/legacy/utils/`

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

### **Stockfish Setup**
```bash
# Stockfish binaries included in engines/ directory
# Verify installation
./engines/stockfish-ubuntu-x86-64-avx2

# For automated setup, see docs/setup_stockfish.md
```

### **First Run**
```bash
# Test game playing
sbatch scripts/game_playing/run_game_evaluation.sh small-8 0 3 10

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
sbatch scripts/game_playing/run_game_evaluation.sh large-16 0 9 1000

# 2. Analyze games
sbatch scripts/analysis/run_stockfish_analysis.sh data/games 64 150000

# 3. Generate visualizations
python src/visualization/enhanced_game_results.py

# 4. Launch dashboard
python src/visualization/enhanced_analysis_dashboard.py
```

### **Quick Model Comparison**
```bash
# Compare multiple models
for model in small-8 medium-12 large-16; do
    sbatch scripts/game_playing/run_game_evaluation.sh $model 0 5 500
done

# Analyze all results
sbatch scripts/analysis/run_stockfish_analysis.sh data/games 32 100000

# View comparison
python src/visualization/enhanced_analysis_dashboard.py
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