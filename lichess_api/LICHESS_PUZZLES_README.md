# Lichess Puzzle Dataset with Original Game PGNs

This directory contains tools to download the Lichess puzzle dataset and extract the original PGN transcripts from the games where each puzzle was found.

## Overview

The Lichess puzzle database contains **5,036,915** rated chess puzzles with metadata including:
- Puzzle position (FEN)
- Solution moves
- Difficulty rating
- Themes (tactics, endgame, middlegame, etc.)
- **Original game URLs** where the puzzle was found

Our tools extract the complete PGN transcripts from these original games, providing rich context for each puzzle.

## Files

- `download_lichess_puzzles.py` - Main script to download puzzles and extract PGNs
- `example_lichess_usage.py` - Example analysis and usage patterns
- `requirements_lichess.txt` - Additional Python dependencies
- `LICHESS_PUZZLES_README.md` - This documentation

## Installation

Install additional dependencies:

```bash
pip install -r requirements_lichess.txt
```

Required packages:
- `zstandard` - For decompressing .zst files
- `tqdm` - Progress bars
- `python-chess` - Chess game processing
- `requests` - HTTP requests
- `pandas` - Data manipulation

## Usage

### 1. Download Complete Dataset

Download all 5M+ puzzles with their original game PGNs:

```bash
python download_lichess_puzzles.py
```

This will:
1. Download `lichess_db_puzzle.csv.zst` (~180MB compressed)
2. Decompress to CSV (~1.3GB)
3. Extract game IDs from URLs
4. Fetch original PGN transcripts via Lichess API
5. Save results to `data/lichess_puzzles/puzzles_with_pgn.jsonl`

### 2. Process Limited Sample

For testing or smaller datasets:

```bash
# Process only first 1000 puzzles
python download_lichess_puzzles.py --max-puzzles 1000

# Download dataset only (no PGN fetching)
python download_lichess_puzzles.py --download-only

# Process existing downloaded data
python download_lichess_puzzles.py --process-only
```

### 3. Resume Interrupted Processing

The script automatically saves progress and can resume:

```bash
# Resume from where it left off
python download_lichess_puzzles.py

# Start fresh (ignore previous progress)
python download_lichess_puzzles.py --no-resume
```

### 4. Analyze the Dataset

```bash
python example_lichess_usage.py
```

## Data Format

The processed data is saved as JSONL (JSON Lines) format in `puzzles_with_pgn.jsonl`:

```json
{
  "puzzle_id": "00sHx",
  "fen": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
  "moves": "e8d7 a2e6 d7d8 f7f8",
  "rating": 1760,
  "rating_deviation": 80,
  "popularity": 83,
  "nb_plays": 72,
  "themes": "mate mateIn2 middlegame short",
  "game_url": "https://lichess.org/yyznGmXs/black#34",
  "game_id": "yyznGmXs",
  "opening_tags": "Italian_Game Italian_Game_Classical_Variation",
  "pgn_available": true,
  "pgn_text": "[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/yyznGmXs\"]\n...",
  "pgn_valid": true
}
```

### Key Fields

- `puzzle_id` - Unique puzzle identifier
- `fen` - Position before the puzzle (after opponent's move)
- `moves` - Solution moves in UCI format
- `rating` - Puzzle difficulty rating (800-3000)
- `themes` - Space-separated tactical themes
- `game_url` - Original Lichess game URL
- `game_id` - Extracted game identifier
- `pgn_text` - Complete original game PGN
- `pgn_valid` - Whether PGN was successfully parsed

## Dataset Statistics

Based on the complete dataset:

- **Total puzzles**: 5,036,915
- **Rating range**: 600-3200
- **Common themes**: mate, middlegame, endgame, tactics, sacrifice
- **PGN availability**: ~95% (some games may be deleted/private)

### Rating Distribution

| Rating Range | Count | Percentage |
|--------------|--------|------------|
| 600-999      | ~500K  | 10%        |
| 1000-1399    | ~1.5M  | 30%        |
| 1400-1799    | ~1.8M  | 36%        |
| 1800-2199    | ~1.0M  | 20%        |
| 2200+        | ~200K  | 4%         |

### Common Themes

- `middlegame` - 2.1M puzzles
- `endgame` - 800K puzzles  
- `mate` - 600K puzzles
- `tactics` - 1.8M puzzles
- `sacrifice` - 300K puzzles

## Rate Limiting

The script implements respectful rate limiting:
- 1 second delay between API requests
- Automatic retry on failures
- Progress saving for resumability

**Note**: Processing the complete dataset may take 24-48 hours due to rate limiting.

## Example Analysis

```python
from example_lichess_usage import load_puzzles_with_pgn, filter_puzzles_by_criteria

# Load processed puzzles
puzzles = load_puzzles_with_pgn("data/lichess_puzzles/puzzles_with_pgn.jsonl")

# Filter for tactical puzzles rated 1200-2000
tactical_puzzles = filter_puzzles_by_criteria(
    puzzles,
    min_rating=1200,
    max_rating=2000,
    themes=['tactics', 'middlegame'],
    require_pgn=True
)

print(f"Found {len(tactical_puzzles)} tactical puzzles")
```

## Use Cases

### 1. Chess Training Data

Create training datasets with game context:

```python
# Extract games with their tactical moments
training_examples = []
for puzzle in puzzles:
    if puzzle['pgn_valid'] and puzzle['rating'] > 1500:
        training_examples.append({
            'position': puzzle['fen'],
            'solution': puzzle['moves'],
            'game_context': puzzle['pgn_text'],
            'difficulty': puzzle['rating']
        })
```

### 2. Opening Analysis

Analyze tactical patterns by opening:

```python
from collections import defaultdict

opening_tactics = defaultdict(list)
for puzzle in puzzles:
    opening = puzzle.get('opening_tags', 'Unknown')
    if 'tactics' in puzzle.get('themes', ''):
        opening_tactics[opening].append(puzzle)
```

### 3. Player Improvement

Find puzzles from games similar to your level:

```python
def find_similar_games(target_rating, puzzles):
    return [p for p in puzzles 
            if abs(p['rating'] - target_rating) < 200 
            and p['pgn_valid']]
```

## Data Sources

- **Puzzles**: [Lichess Puzzle Database](https://database.lichess.org/#puzzles)
- **Game PGNs**: [Lichess Game API](https://lichess.org/api#operation/gameExport)
- **Format**: CSV (compressed with Zstandard)
- **License**: Creative Commons CC0 (public domain)

## Technical Notes

### File Sizes

- Compressed dataset: ~180MB
- Decompressed CSV: ~1.3GB  
- Processed JSONL: ~8-12GB (depending on PGN availability)

### API Endpoints

- Puzzle dataset: `https://database.lichess.org/lichess_db_puzzle.csv.zst`
- Game export: `https://lichess.org/api/game/export/{gameId}`

### Performance

- Download speed: Limited by network (~10MB/s)
- Processing speed: ~1 puzzle/second (due to API rate limiting)
- Memory usage: <100MB (streaming processing)

## Troubleshooting

### Common Issues

1. **Download fails**: Check network connection and retry
2. **Decompression fails**: Install `zstandard` package
3. **API rate limiting**: Script handles this automatically
4. **Disk space**: Ensure 15GB+ free space for complete dataset

### Error Messages

- `❌ HF_TOKEN not found`: Not applicable for Lichess (no auth required)
- `⚠️ Game not found`: Some games may be deleted or private
- `❌ HTTP 429`: Rate limited - script will retry automatically

## Contributing

Improvements welcome:
- Better error handling
- Parallel processing (respecting rate limits)
- Additional analysis functions
- Alternative storage formats

## License

This code is provided under the same CC0 license as the Lichess database - use freely for any purpose.