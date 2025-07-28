# Chess GPT Evaluation on CSF3

**CSF3-specific workflows and optimization guide** for the University of Manchester's cluster. For general project overview and usage, see the [main README](../README.md).

## 🎯 CSF3-Optimized Workflows

This guide covers CSF3-specific optimizations, legacy file paths, and cluster-specific configurations.

### **Current vs Legacy File Paths**
```bash
# Current structured approach (recommended)
python src/models/main.py large-16 0 9 1000
sbatch scripts/run_stockfish_analysis.sh data/games 64 150000

# Legacy file paths (if using older checkpoints)
python scripts/parallel_chess_eval.py --model large-24-600K_iters.pt --games 1000
sbatch run_mass_stockfish.sh
python mass_stockfish_processor.py --max-games 2 --workers 4
```

### **Current Recommended Usage**
For current structured approach, see [main README workflows](../README.md#quick-start).

## 📁 Legacy Repository Structure

```
chess_gpt_eval/
├── 🆕 STOCKFISH ANALYSIS SYSTEM (Legacy paths)
│   ├── mass_stockfish_processor.py    # Now: src/analysis/mass_stockfish_processor.py
│   ├── stockfish_analysis.py          # Now: src/analysis/stockfish_analysis.py
│   ├── run_mass_stockfish.sh          # Legacy batch script
│   ├── advanced_chess_dashboard.py    # Now: src/visualization/interactive_dashboard.py
│   └── chess_llm_benchmark.py         # Analysis benchmarking
├── scripts/                           # Legacy scripts directory
│   ├── parallel_chess_eval.py         # Main parallel evaluation script
│   └── test_parallel_setup.sh         # Test script for parallel setup
└── logs/                              # Legacy game storage (now: data/games/)
```

## 🔧 CSF3-Specific Configurations

### **Resource Optimization**
- **Memory Efficiency**: Optimized for CSF3 resource limits
- **CPU Parallelization**: 11 workers for game logic, 140 workers for Stockfish analysis
- **GPU Batch Processing**: 16-32 positions simultaneously
- **Array Jobs**: Scale to multiple concurrent evaluations

### **Performance Expectations**
- **Stockfish Analysis**: ~50-100 games/minute (140 workers)
- **80K games**: 15-30 hours (full analysis)  
- **Memory usage**: ~1.4TB RAM, 160 CPUs for large analyses
- **Game Evaluation**: 5-15 games/second per job, 2-5 minutes for 1000 games

## 🛠️ CSF3 Setup

### **Module Loading**
```bash
# CSF3-specific modules (older versions)
module load apps/anaconda3/2022.10

# Install dependencies
pip install -r requirements.txt
```

### **Legacy Stockfish Setup**
```bash
# Install Stockfish (automated - legacy approach)
bash install_stockfish.sh

# Manual setup (see setup_stockfish.md for details)
```

## 🔍 CSF3 Monitoring

### **SLURM Job Monitoring**
```bash
# Monitor batch processing
tail -f mass_stockfish_*.out
squeue -u $USER

# Check analysis results (legacy paths)
ls -lh stockfish_analysis_results/
head stockfish_analysis_results/*_summary_*.csv
```

### **Legacy Analysis Outputs**
```bash
# Game-level statistics (legacy output format)
head stockfish_analysis_results/large-16_summary_*.csv

# Move-by-move data (legacy format)
head stockfish_analysis_results/large-16_moves_*.csv

# Detailed JSON data (legacy format)
jq '.games[0].moves_analysis[0]' stockfish_analysis_results/large-16_detailed_*.json
```

## 📊 CSF3 Analysis Features

### **High-Performance Batch Analysis**
- **140 workers** with 150K nodes per position
- **Move Classifications**: Blunder, mistake, inaccuracy, good, best
- **Position Metrics**: Material imbalance, complexity scoring (0-100)
- **Game Phases**: Opening, middlegame, endgame detection
- **Error Patterns**: Blunder frequency, mistake distribution

### **Available Models (Legacy naming)**
```bash
# Models in logs/ directory (legacy structure)
large-16-600k_iters_pt_vs_stockfish_sweep.csv      # ~1548 Elo
medium-16-600k_iters_pt_vs_stockfish_sweep.csv     # ~1527 Elo
medium-12-600k_iters_pt_vs_stockfish_sweep.csv     # ~1482 Elo
small-16-600k_iters_pt_vs_stockfish_sweep.csv      # ~1469 Elo
small-8-600k_iters_pt_vs_stockfish_sweep.csv       # ~1377 Elo
```

## 🎮 CSF3 Example Workflows

### **Legacy Analysis Pipeline**
For complete examples using current file structure, see [main README workflows](../README.md#example-workflows).

Legacy commands (for older file structure):
```bash
# 1. Generate games (legacy approach)  
python scripts/parallel_chess_eval.py --model large-24-600K_iters.pt --games 10000

# 2. Comprehensive Stockfish analysis (legacy paths)
sbatch run_mass_stockfish.sh

# 3. Create visualizations (legacy paths)
python advanced_chess_dashboard.py --input stockfish_analysis_results/

# Current approach (recommended)
python launch_dashboards.py interactive

# Monitor progress for large-scale analysis
watch -n 30 'ls stockfish_analysis_results/ | wc -l'
```

## 📚 CSF3-Specific Documentation

- **[CSF3 Setup Guide](README_CSF3_Setup.md)** - Detailed CSF3 cluster setup
- **[Resource Limits](CSF3_Resource_Limits.md)** - CSF3 resource constraints and optimization
- **[Dashboard Guide](dashboard_README.md)** - Visualization tools
- **[Stockfish Setup](setup_stockfish.md)** - Engine installation

## 🔄 Migration from Legacy

To migrate from legacy file structure to current organized structure:
1. Use current [main README workflows](../README.md#quick-start) for new projects
2. Reference this guide for maintaining existing CSF3 jobs using legacy paths
3. See main documentation for all current features and capabilities

---

**For complete project documentation and current workflows, see the [main README](../README.md).** 