#!/usr/bin/env python3
"""
Download Lichess puzzle dataset and extract PGN transcripts from original games
"""

import os
import sys
import requests
import pandas as pd
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import zstandard as zstd
from tqdm import tqdm
import re
from urllib.parse import urlparse
import chess.pgn
import io

class LichessPuzzleDownloader:
    """Download and process Lichess puzzle dataset with original game PGNs"""
    
    def __init__(self, data_dir: str = "data/lichess_puzzles"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Lichess API endpoints
        self.base_url = "https://lichess.org"
        self.api_url = "https://lichess.org/api"
        self.download_url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Files
        self.puzzle_file = self.data_dir / "lichess_db_puzzle.csv.zst"
        self.puzzle_csv = self.data_dir / "lichess_db_puzzle.csv"
        self.processed_puzzles = self.data_dir / "puzzles_with_pgn.jsonl"
        self.progress_file = self.data_dir / "download_progress.json"
    
    def download_puzzle_dataset(self) -> bool:
        """Download the Lichess puzzle dataset if not already present"""
        
        if self.puzzle_file.exists():
            print(f"✅ Puzzle dataset already exists: {self.puzzle_file}")
            return True
        
        print(f"⬇️  Downloading Lichess puzzle dataset...")
        print(f"📡 URL: {self.download_url}")
        print(f"💾 Destination: {self.puzzle_file}")
        
        try:
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.puzzle_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded successfully: {self.puzzle_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading puzzle dataset: {e}")
            return False
    
    def decompress_dataset(self) -> bool:
        """Decompress the .zst file to CSV"""
        
        if self.puzzle_csv.exists():
            print(f"✅ Decompressed CSV already exists: {self.puzzle_csv}")
            return True
        
        if not self.puzzle_file.exists():
            print(f"❌ Compressed file not found: {self.puzzle_file}")
            return False
        
        print(f"📦 Decompressing {self.puzzle_file}...")
        
        try:
            with open(self.puzzle_file, 'rb') as compressed_file:
                decompressor = zstd.ZstdDecompressor()
                with open(self.puzzle_csv, 'wb') as output_file:
                    decompressor.copy_stream(compressed_file, output_file)
            
            print(f"✅ Decompressed to: {self.puzzle_csv}")
            print(f"📊 File size: {self.puzzle_csv.stat().st_size / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error decompressing dataset: {e}")
            return False
    
    def load_puzzles(self, max_puzzles: Optional[int] = None) -> pd.DataFrame:
        """Load puzzles from CSV file"""
        
        if not self.puzzle_csv.exists():
            print(f"❌ CSV file not found: {self.puzzle_csv}")
            return pd.DataFrame()
        
        print(f"📖 Loading puzzles from {self.puzzle_csv}...")
        
        try:
            # Read with specified chunk size to handle large files
            if max_puzzles:
                df = pd.read_csv(self.puzzle_csv, nrows=max_puzzles)
                print(f"📊 Loaded {len(df)} puzzles (limited to {max_puzzles})")
            else:
                df = pd.read_csv(self.puzzle_csv)
                print(f"📊 Loaded {len(df)} puzzles")
            
            # Validate required columns
            required_cols = ['PuzzleId', 'GameUrl', 'FEN', 'Moves', 'Rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            print(f"✅ Puzzle data loaded successfully")
            print(f"📋 Columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading puzzles: {e}")
            return pd.DataFrame()
    
    def extract_game_id_from_url(self, game_url: str) -> Optional[str]:
        """Extract game ID from Lichess game URL"""
        # Example URL: https://lichess.org/yyznGmXs/black#34
        try:
            # Parse the URL
            parsed = urlparse(game_url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) >= 1:
                # The game ID is the first part of the path
                game_id = path_parts[0]
                return game_id
            return None
        except:
            return None
    
    def rate_limit(self):
        """Simple rate limiting to avoid overwhelming Lichess API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_game_pgn(self, game_id: str) -> Optional[str]:
        """Fetch PGN for a specific game from Lichess export endpoint"""
        
        self.rate_limit()
        
        try:
            # Use the correct game export endpoint
            url = f"{self.base_url}/game/export/{game_id}"
            headers = {
                'Accept': 'application/x-chess-pgn',
                'User-Agent': 'ChessGPT Research Project'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.text.strip()
            elif response.status_code == 404:
                print(f"⚠️  Game not found: {game_id}")
                return None
            else:
                print(f"⚠️  HTTP {response.status_code} for game {game_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching game {game_id}: {e}")
            return None
    
    def validate_pgn(self, pgn_text: str) -> bool:
        """Validate that PGN text is properly formatted"""
        try:
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            return game is not None
        except:
            return False
    
    def save_progress(self, processed_count: int, total_count: int):
        """Save processing progress"""
        progress = {
            'processed_count': processed_count,
            'total_count': total_count,
            'progress_percent': (processed_count / total_count) * 100 if total_count > 0 else 0,
            'timestamp': time.time()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self) -> Dict:
        """Load processing progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'processed_count': 0, 'total_count': 0}
    
    def process_puzzles_with_pgn(self, df: pd.DataFrame, 
                                max_puzzles: Optional[int] = None,
                                resume: bool = True) -> None:
        """Process puzzles and fetch their original game PGNs"""
        
        if max_puzzles:
            df = df.head(max_puzzles)
        
        total_puzzles = len(df)
        print(f"🎯 Processing {total_puzzles} puzzles to extract PGN data...")
        
        # Load existing progress
        progress = self.load_progress() if resume else {'processed_count': 0}
        start_idx = progress.get('processed_count', 0)
        
        if start_idx > 0:
            print(f"📍 Resuming from puzzle {start_idx}")
        
        # Open output file for appending
        mode = 'a' if resume and start_idx > 0 else 'w'
        
        with open(self.processed_puzzles, mode, encoding='utf-8') as f:
            with tqdm(total=total_puzzles, initial=start_idx, desc="Processing puzzles") as pbar:
                
                for idx, row in df.iloc[start_idx:].iterrows():
                    try:
                        puzzle_id = row['PuzzleId']
                        game_url = row['GameUrl']
                        
                        # Extract game ID
                        game_id = self.extract_game_id_from_url(game_url)
                        if not game_id:
                            print(f"⚠️  Could not extract game ID from: {game_url}")
                            continue
                        
                        # Fetch game PGN
                        pgn_text = self.fetch_game_pgn(game_id)
                        
                        # Create puzzle record
                        puzzle_record = {
                            'puzzle_id': puzzle_id,
                            'fen': row['FEN'],
                            'moves': row['Moves'],
                            'rating': row['Rating'],
                            'rating_deviation': row.get('RatingDeviation', 0),
                            'popularity': row.get('Popularity', 0),
                            'nb_plays': row.get('NbPlays', 0),
                            'themes': row.get('Themes', ''),
                            'game_url': game_url,
                            'game_id': game_id,
                            'opening_tags': row.get('OpeningTags', ''),
                            'pgn_available': pgn_text is not None,
                            'pgn_text': pgn_text if pgn_text else '',
                            'pgn_valid': self.validate_pgn(pgn_text) if pgn_text else False
                        }
                        
                        # Write to JSONL file
                        f.write(json.dumps(puzzle_record, ensure_ascii=False) + '\n')
                        f.flush()  # Ensure data is written
                        
                        # Update progress
                        processed_count = start_idx + (idx - start_idx) + 1
                        if processed_count % 100 == 0:  # Save progress every 100 puzzles
                            self.save_progress(processed_count, total_puzzles)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"❌ Error processing puzzle {puzzle_id}: {e}")
                        continue
        
        print(f"✅ Processing complete! Results saved to: {self.processed_puzzles}")
        
        # Final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistics about the processed data"""
        
        if not self.processed_puzzles.exists():
            print("❌ No processed data found")
            return
        
        print("\n📊 Processing Statistics:")
        print("=" * 50)
        
        total_puzzles = 0
        pgn_available = 0
        pgn_valid = 0
        rating_distribution = {}
        
        with open(self.processed_puzzles, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    total_puzzles += 1
                    
                    if record.get('pgn_available'):
                        pgn_available += 1
                    
                    if record.get('pgn_valid'):
                        pgn_valid += 1
                    
                    # Rating distribution
                    rating = record.get('rating', 0)
                    rating_bucket = (rating // 200) * 200  # 200-point buckets
                    rating_distribution[rating_bucket] = rating_distribution.get(rating_bucket, 0) + 1
                    
                except:
                    continue
        
        print(f"Total puzzles processed: {total_puzzles:,}")
        print(f"PGN data available: {pgn_available:,} ({pgn_available/total_puzzles*100:.1f}%)")
        print(f"Valid PGN data: {pgn_valid:,} ({pgn_valid/total_puzzles*100:.1f}%)")
        
        print(f"\n📈 Rating Distribution:")
        for rating_bucket in sorted(rating_distribution.keys()):
            count = rating_distribution[rating_bucket]
            print(f"  {rating_bucket}-{rating_bucket+199}: {count:,} puzzles")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Lichess puzzles and extract original game PGNs")
    parser.add_argument("--data-dir", default="data/lichess_puzzles", help="Data directory")
    parser.add_argument("--max-puzzles", type=int, help="Maximum number of puzzles to process")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't process")
    parser.add_argument("--process-only", action="store_true", help="Only process existing data")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous progress")
    
    args = parser.parse_args()
    
    downloader = LichessPuzzleDownloader(args.data_dir)
    
    print("🏁 Lichess Puzzle Dataset Downloader")
    print("=" * 50)
    
    if not args.process_only:
        # Download and decompress dataset
        if not downloader.download_puzzle_dataset():
            sys.exit(1)
        
        if not downloader.decompress_dataset():
            sys.exit(1)
    
    if not args.download_only:
        # Load and process puzzles
        puzzles_df = downloader.load_puzzles(args.max_puzzles)
        
        if puzzles_df.empty:
            print("❌ No puzzle data loaded")
            sys.exit(1)
        
        # Process puzzles with PGN extraction
        downloader.process_puzzles_with_pgn(
            puzzles_df, 
            args.max_puzzles,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()