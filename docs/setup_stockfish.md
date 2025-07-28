# Stockfish Setup for Batch Processing

## Quick Setup

### Option 1: Install Stockfish (Recommended)
```bash
# On CSF3/RHEL systems:
sudo yum install stockfish
# OR if you have conda/mamba:
conda install -c conda-forge stockfish

# Test installation:
stockfish --help
```

### Option 2: Download Stockfish Binary
```bash
# Download and install locally
mkdir -p ~/bin
cd ~/bin
wget https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar
tar -xf stockfish-ubuntu-x86-64-avx2.tar
mv stockfish/stockfish-ubuntu-x86-64-avx2 stockfish
chmod +x stockfish

# Add to PATH
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Test with Small Sample (100 games per model)
```bash
python batch_stockfish_processor.py --sample 100 --workers 4
```

### Process All 80,000 Games
```bash
# Full processing with 8 workers (recommended)
python batch_stockfish_processor.py --workers 8

# Monitor progress in batch_stockfish.log
tail -f batch_stockfish.log
```

### Advanced Usage
```bash
# Process specific number of games per model
python batch_stockfish_processor.py --sample 5000 --workers 6

# Process single model file
python -c "
from batch_stockfish_processor import BatchStockfishProcessor
processor = BatchStockfishProcessor()
processor.process_file('logs/medium-16-600k_iters_pt_vs_stockfish_sweep.csv', 
                      'stockfish_batch/medium-16_stockfish.json', 
                      num_workers=8)
"
```

## Performance Expectations

**Hardware Setup:** 8-core system with Stockfish
- **Per game:** ~2-5 seconds (depth 15, 50 moves analyzed)
- **1,000 games:** ~45-90 minutes with 8 workers
- **10,000 games:** ~7-15 hours with 8 workers  
- **80,000 games:** ~2-5 days with 8 workers

**Optimization Tips:**
- Use `--sample` for testing first
- Monitor CPU/memory usage with `htop`
- Run in `screen` or `tmux` for long jobs
- Consider reducing depth for faster processing

## Output Files

Each model generates:
- `{model}_stockfish.json` - Detailed analysis data
- `{model}_stockfish.csv` - Summary statistics
- `batch_stockfish.log` - Processing log

## Expected Output Structure

```json
{
  "game_id": "1749361809-9677",
  "total_moves": 42,
  "average_centipawn_loss": 145.3,
  "blunder_count": 3,
  "mistake_count": 8,
  "opening_accuracy": 78.5,
  "middlegame_accuracy": 65.2,
  "endgame_accuracy": 45.1
}
```

## Troubleshooting

**"Stockfish not found"**
- Ensure Stockfish is in PATH: `which stockfish`
- Try manual installation above

**"Memory issues"**
- Reduce number of workers: `--workers 4`
- Process in smaller batches: `--sample 1000`

**"Process hanging"**
- Check system resources: `htop`
- Reduce analysis depth in code (line 35: `depth=10`)

## Next Steps After Processing

Once you have the Stockfish annotations:
1. **Analyze patterns:** Which models make more blunders in different game phases?
2. **Correlate with performance:** Do models with lower centipawn loss have higher Elo?
3. **Build visualizations:** Use the processed data for your thesis dashboard
4. **Mechanistic interpretability:** Correlate Stockfish evaluations with model internal states 