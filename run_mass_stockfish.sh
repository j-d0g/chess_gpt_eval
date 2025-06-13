#!/bin/bash
#SBATCH --job-name=mass_stockfish
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=1400G
#SBATCH --time=48:00:00
#SBATCH --output=mass_stockfish_%j.out
#SBATCH --error=mass_stockfish_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@manchester.ac.uk

# Load required modules
module load apps/anaconda3/2022.10

# Activate conda environment
source activate base

# Set up environment
export OMP_NUM_THREADS=1  # Important for multiprocessing
export PYTHONUNBUFFERED=1

# Create output directory
mkdir -p stockfish_analysis_results

# Log system information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Available CPUs: $(nproc --all)"
echo "Available memory: $(free -h)"
echo "Stockfish version: $(./stockfish/stockfish-ubuntu-x86-64-modern | head -1)"

# Run the mass processing
echo "Starting mass Stockfish processing..."
python mass_stockfish_processor.py \
    --input-dir logs \
    --output-dir stockfish_analysis_results \
    --workers 140 \
    --chunk-size 50 \
    --nodes 150000 \
    --stockfish-path ./stockfish/stockfish-ubuntu-x86-64-modern

echo "Job completed at: $(date)"

# Show final results summary
echo "=== FINAL RESULTS SUMMARY ==="
ls -lh stockfish_analysis_results/
echo "Total output files: $(ls stockfish_analysis_results/ | wc -l)"
echo "Total disk usage: $(du -sh stockfish_analysis_results/)" 