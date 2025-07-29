#!/bin/bash
#SBATCH --job-name=chess_stockfish_analysis
#SBATCH --output=chess_stockfish_analysis_%j.out
#SBATCH --error=chess_stockfish_analysis_%j.err
#SBATCH --partition=multicore
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j74739jt@students.manchester.ac.uk

# Enhanced Stockfish Analysis Workflow
# Usage: sbatch run_stockfish_analysis.sh [input_dir] [workers] [nodes_per_position]
# Example: sbatch run_stockfish_analysis.sh data/game_logs 64 150000

# Load required modules
module load apps/binapps/anaconda3/2023.09
module load tools/gcc/11.2.0

# Default parameters
INPUT_DIR=${1:-"data/games"}
WORKERS=${2:-$SLURM_CPUS_PER_TASK}
NODES_PER_POSITION=${3:-150000}

echo "=========================================="
echo "CHESS STOCKFISH ANALYSIS WORKFLOW"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Workers: $WORKERS"
echo "Nodes per position: $NODES_PER_POSITION"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 512GB"
echo "=========================================="

# Set up environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Change to project directory
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_gpt_eval

# Create output directory
mkdir -p data/analysis

# Count total games to analyze
echo "Counting games to analyze..."
TOTAL_GAMES=0
for csv in $INPUT_DIR/*.csv; do
    if [[ -f "$csv" && $(stat -c%s "$csv") -gt 1000 ]]; then
        GAMES=$(wc -l < "$csv")
        echo "  $(basename "$csv"): $GAMES games"
        TOTAL_GAMES=$((TOTAL_GAMES + GAMES))
    fi
done
echo "Total games to analyze: $TOTAL_GAMES"

# Estimate completion time
ESTIMATED_HOURS=$((TOTAL_GAMES / (WORKERS * 10)))
echo "Estimated completion time: $ESTIMATED_HOURS hours"

echo "=========================================="
echo "STARTING STOCKFISH ANALYSIS..."
echo "=========================================="

# Run the comprehensive analysis
python src/analysis/mass_stockfish_processor.py \
    --input-dir "$INPUT_DIR" \
    --output-dir data/analysis \
    --workers "$WORKERS" \
    --chunk-size 50 \
    --nodes "$NODES_PER_POSITION" \
    --stockfish-path ./src/models/stockfish/stockfish-ubuntu-x86-64-avx2

# Check if successful
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "STOCKFISH ANALYSIS COMPLETED SUCCESSFULLY!"
    echo "Completed at: $(date)"
    echo "=========================================="
    
    # Show output files
    echo "Generated files:"
    ls -lh data/analysis/
    
    # Count analyzed games
    echo "Analysis summary:"
    for file in data/analysis/*_summary_*.csv; do
        if [ -f "$file" ]; then
            echo "  $(basename "$file"): $(wc -l < "$file") games analyzed"
        fi
    done
    
    # Generate analysis summary
    echo "Generating analysis summary..."
    python src/visualization/generate_analysis_summary.py
    
    echo "=========================================="
    echo "NEXT STEPS:"
    echo "1. Launch dashboard: python src/visualization/enhanced_analysis_dashboard.py"
    echo "2. Generate reports: python src/visualization/model_comparative_analysis.py"
    echo "3. View detailed analysis: ls data/analysis/"
    echo "=========================================="
else
    echo "=========================================="
    echo "STOCKFISH ANALYSIS FAILED!"
    echo "Check error logs for details."
    echo "=========================================="
    exit 1
fi 