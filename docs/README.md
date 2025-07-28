# Chess GPT Evaluation on CSF3

A comprehensive system for evaluating chess language models against Stockfish on the University of Manchester's CSF3 cluster, with optimized parallel processing, batch inference, and **comprehensive Stockfish analysis**.

## 🚀 Quick Start

### **Game Evaluation**
```bash
# Test the parallel setup (20 games)
sbatch scripts/test_parallel_setup.sh

# Run evaluation (1000 games)
python scripts/parallel_chess_eval.py --model large-24-600K_iters.pt --games 1000
```

### **🆕 Stockfish Analysis (NEW)**
```bash
# Analyze all games with comprehensive Stockfish annotations
sbatch run_mass_stockfish.sh

# Quick test (2 games)
python mass_stockfish_processor.py --max-games 2 --workers 4

# Custom analysis
python mass_stockfish_processor.py --input-dir logs --workers 32 --nodes 150000
```

## 📁 Repository Structure

```
chess_gpt_eval/
├── 🆕 STOCKFISH ANALYSIS SYSTEM
│   ├── mass_stockfish_processor.py    # High-performance batch Stockfish analyzer
│   ├── stockfish_analysis.py          # Detailed single-game analysis
│   ├── run_mass_stockfish.sh          # SLURM batch processing script
│   ├── advanced_chess_dashboard.py    # Comprehensive visualization dashboard
│   ├── chess_llm_benchmark.py         # Standardized benchmarking suite
│   └── setup_stockfish.md             # Stockfish installation guide
├── docs/                              # Documentation
│   ├── README_CSF3_Setup.md           # CSF3 setup guide
│   ├── PARALLEL_SETUP_SUMMARY.md      # Complete parallel processing guide
│   ├── CSF3_Resource_Limits.md        # Resource limits and optimization
│   └── GPU_Performance_Analysis.md    # Performance optimization guide
├── scripts/                           # Executable scripts
│   ├── parallel_chess_eval.py         # Main parallel evaluation script
│   ├── test_parallel_setup.sh         # Test script for parallel setup
│   └── gpu_performance_test.py        # GPU performance diagnostics
├── nanogpt/                           # NanoGPT model code
│   ├── nanogpt_batch_module.py        # Batch-enabled inference module
│   ├── model.py                       # GPT model implementation
│   └── out/                           # Model checkpoints
├── logs/                              # Game CSV files (input data)
├── stockfish_analysis_results/        # 🆕 Comprehensive analysis outputs
├── stockfish/                         # Stockfish engine installation
├── results/                           # Test outputs and job results
└── requirements.txt                   # Python dependencies
```

## 🎯 Key Features

### **🆕 Comprehensive Stockfish Analysis**
- **Move-by-Move Analysis**: Centipawn loss, blunders, mistakes, inaccuracies
- **Position Evaluation**: Material imbalance, complexity scoring, game phases
- **Best Move Analysis**: Principal variations, alternative lines
- **High Performance**: 140 workers, 150K nodes per position
- **Multiple Outputs**: JSON (detailed), CSV (summary + moves), visualizations

### **Parallel Game Evaluation**
- **Batch GPU Inference**: Process 16-32 chess positions simultaneously
- **CPU Parallelization**: 11 workers handle game logic and Stockfish
- **Queue-based Architecture**: Efficient communication between GPU and CPU workers

### **CSF3 Optimization**
- **Resource Discovery**: Automatically determined optimal configurations
- **Memory Efficiency**: Works within CSF3 resource limits
- **Array Jobs**: Scale to multiple concurrent evaluations

## 🔧 Usage

### **🆕 Stockfish Batch Analysis**

#### **Full Analysis (80K+ games)**
```bash
# Submit to SLURM (recommended for large datasets)
sbatch run_mass_stockfish.sh

# Direct execution with custom settings
python mass_stockfish_processor.py \
    --input-dir logs \
    --output-dir stockfish_analysis_results \
    --workers 140 \
    --nodes 150000 \
    --chunk-size 50
```

#### **Quick Testing**
```bash
# Test with 5 games, 4 workers
python mass_stockfish_processor.py --max-games 5 --workers 4 --nodes 1000

# Analyze specific files
python mass_stockfish_processor.py --files logs/small-8-600k_iters_pt_vs_stockfish_sweep.csv
```

#### **Output Files**
```bash
stockfish_analysis_results/
├── {model}_detailed_{timestamp}.json     # Complete move-by-move data
├── {model}_summary_{timestamp}.csv       # Game-level statistics  
└── {model}_moves_{timestamp}.csv         # Move-by-move CSV data
```

### **Game Evaluation**
```bash
# Small test
python scripts/parallel_chess_eval.py \
    --model small-8-600k_iters.pt \
    --games 100 \
    --workers 4 \
    --batch-size 8

# Full evaluation
python scripts/parallel_chess_eval.py \
    --model large-24-600K_iters.pt \
    --games 1000 \
    --workers 11 \
    --batch-size 16
```

## 📊 Analysis Capabilities

### **🆕 Stockfish Analysis Features**
- ✅ **Move Classifications**: Blunder, mistake, inaccuracy, good, best
- ✅ **Centipawn Analysis**: Position scores, centipawn loss per move
- ✅ **Best Move Analysis**: Stockfish recommendations + principal variations
- ✅ **Position Metrics**: Material imbalance, complexity scoring (0-100)
- ✅ **Game Phases**: Opening, middlegame, endgame detection
- ✅ **Performance Stats**: Nodes, depth, time, NPS per position
- ✅ **Error Patterns**: Blunder frequency, mistake distribution
- ✅ **Comprehensive Statistics**: 25+ metrics per game

### **Available Models for Analysis**
```bash
# Your trained models (in logs/)
large-16-600k_iters_pt_vs_stockfish_sweep.csv      # Best model (~1548 Elo)
medium-16-600k_iters_pt_vs_stockfish_sweep.csv     # ~1527 Elo
medium-12-600k_iters_pt_vs_stockfish_sweep.csv     # ~1482 Elo
small-16-600k_iters_pt_vs_stockfish_sweep.csv      # ~1469 Elo
small-8-600k_iters_pt_vs_stockfish_sweep.csv       # ~1377 Elo
```

## 📈 Performance Expectations

### **🆕 Stockfish Analysis Performance**
- **Throughput**: ~50-100 games/minute (140 workers)
- **80K games**: 15-30 hours (full analysis)
- **Memory usage**: ~1.4TB RAM, 160 CPUs
- **Output size**: ~1-2GB per 10K games

### **Game Evaluation Performance**
- **Throughput**: 5-15 games/second per job
- **1000 games**: 2-5 minutes
- **Memory usage**: 2-5GB GPU, 10-20GB RAM

## 🛠️ Setup and Installation

### **Prerequisites**
```bash
# On CSF3
module load apps/anaconda3/2022.10

# Install dependencies
pip install -r requirements.txt
```

### **🆕 Stockfish Setup**
```bash
# Install Stockfish (automated)
bash install_stockfish.sh

# Manual setup (see setup_stockfish.md for details)
```

### **First Run**
```bash
# Test Stockfish analysis
python mass_stockfish_processor.py --max-games 2 --workers 2

# Test game evaluation
sbatch scripts/test_parallel_setup.sh
```

## 📚 Documentation

- **[Stockfish Setup Guide](setup_stockfish.md)** - Installing and configuring Stockfish
- **[Stockfish Features](stockfish_missing_features.md)** - Detailed analysis capabilities
- **[Dashboard Guide](dashboard_README.md)** - Visualization and dashboard usage
- **[CSF3 Setup Guide](docs/README_CSF3_Setup.md)** - Getting started with CSF3
- **[Resource Limits](docs/CSF3_Resource_Limits.md)** - CSF3 resource constraints

## 🔍 Monitoring and Analysis

### **🆕 Stockfish Analysis Monitoring**
```bash
# Monitor batch processing
tail -f mass_stockfish_*.out
squeue -u $USER

# Check results
ls -lh stockfish_analysis_results/
head stockfish_analysis_results/*_summary_*.csv
```

### **Analysis Outputs**
```bash
# Game-level statistics
head stockfish_analysis_results/large-16_summary_*.csv

# Move-by-move data  
head stockfish_analysis_results/large-16_moves_*.csv

# Detailed JSON data
jq '.games[0].moves_analysis[0]' stockfish_analysis_results/large-16_detailed_*.json
```

## 🎮 Example Workflows

### **🆕 Complete Analysis Pipeline**
```bash
# 1. Generate games (if needed)
python scripts/parallel_chess_eval.py --model large-24-600K_iters.pt --games 10000

# 2. Comprehensive Stockfish analysis
sbatch run_mass_stockfish.sh

# 3. Create visualizations
python advanced_chess_dashboard.py --input stockfish_analysis_results/

# 4. Generate benchmarks
python chess_llm_benchmark.py --input stockfish_analysis_results/
```

### **Quick Model Comparison**
```bash
# Analyze existing game logs
python mass_stockfish_processor.py --max-games 1000 --workers 32

# Compare results
python advanced_chess_dashboard.py --compare-models
```

### **Large-Scale Research Analysis**
```bash
# Full dataset analysis (80K+ games)
sbatch run_mass_stockfish.sh

# Monitor progress
watch -n 30 'ls stockfish_analysis_results/ | wc -l'
```

## 📄 License

This project builds upon the original chess evaluation framework and adds CSF3-optimized parallel processing capabilities.

## 🤝 Contributing

1. Test changes with `scripts/test_parallel_setup.sh`
2. Update documentation in `docs/`
3. Follow the established directory structure
4. Monitor performance impact

---

**Ready to evaluate your chess models at scale on CSF3!** 🚀 