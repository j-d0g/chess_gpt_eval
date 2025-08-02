#!/usr/bin/env python3
"""
Process sample puzzles and fetch PGNs from Lichess API
"""

import pandas as pd
import requests
import json
import time
import re
from pathlib import Path
from typing import Optional
import chess.pgn
import io

class SamplePuzzleProcessor:
    """Process sample puzzle CSV and fetch PGNs"""
    
    def __init__(self, csv_file: str, output_file: str):
        self.csv_file = csv_file
        self.output_file = output_file
        self.request_delay = 1.0  # Lichess rate limit
        self.last_request_time = 0
        
    def extract_game_id(self, game_url: str) -> Optional[str]:
        """Extract game ID from Lichess URL"""
        if not game_url or not isinstance(game_url, str):
            return None
        
        # Parse URLs like https://lichess.org/ABC123/white#45
        match = re.search(r'lichess\.org/([a-zA-Z0-9]+)', game_url)
        if match:
            return match.group(1)
        return None
    
    def fetch_pgn(self, game_id: str) -> Optional[str]:
        """Fetch PGN for a game from Lichess API"""
        if not game_id:
            return None
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        
        try:
            url = f"https://lichess.org/game/export/{game_id}"
            headers = {
                'User-Agent': 'Chess-LLM-Research/1.0 (Educational)',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.text.strip()
            elif response.status_code == 404:
                print(f"⚠️  Game {game_id} not found (404)")
                return None
            else:
                print(f"⚠️  HTTP {response.status_code} for game {game_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching game {game_id}: {e}")
            return None
    
    def validate_pgn(self, pgn_text: str) -> bool:
        """Validate PGN can be parsed"""
        if not pgn_text:
            return False
        
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            return game is not None
        except:
            return False
    
    def process_puzzles(self):
        """Main processing function"""
        print(f"🎯 Processing puzzles from {self.csv_file}")
        print("=" * 60)
        
        # Load puzzle data
        print(f"📖 Loading puzzle data...")
        df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(df):,} puzzles")
        
        # Process each puzzle
        processed_puzzles = []
        successful_fetches = 0
        failed_fetches = 0
        
        print(f"\\n🔄 Fetching PGNs from Lichess API...")
        print(f"⏱️  Rate limit: {self.request_delay}s between requests")
        print(f"📊 Progress updates every 100 puzzles")
        
        for idx, row in df.iterrows():
            if idx > 0 and idx % 100 == 0:
                success_rate = successful_fetches / (successful_fetches + failed_fetches) * 100 if (successful_fetches + failed_fetches) > 0 else 0
                print(f"   Progress: {idx:,}/{len(df):,} ({idx/len(df)*100:.1f}%) | Success rate: {success_rate:.1f}%")
            
            # Extract game ID
            game_id = self.extract_game_id(row['GameUrl'])
            if not game_id:
                failed_fetches += 1
                continue
            
            # Fetch PGN
            pgn_text = self.fetch_pgn(game_id)
            pgn_valid = self.validate_pgn(pgn_text) if pgn_text else False
            
            # Create puzzle record
            puzzle_record = {
                'puzzle_id': row['PuzzleId'],
                'fen': row['FEN'],
                'moves': row['Moves'],
                'rating': row['Rating'],
                'rating_deviation': row['RatingDeviation'],
                'popularity': row['Popularity'],
                'nb_plays': row['NbPlays'],
                'themes': row['Themes'],
                'game_url': row['GameUrl'],
                'game_id': game_id,
                'opening_tags': row.get('OpeningTags', ''),
                'pgn_available': pgn_text is not None,
                'pgn_text': pgn_text,
                'pgn_valid': pgn_valid
            }
            
            processed_puzzles.append(puzzle_record)
            
            if pgn_text:
                successful_fetches += 1
            else:
                failed_fetches += 1
        
        # Final statistics
        total_processed = len(processed_puzzles)
        success_rate = successful_fetches / total_processed * 100 if total_processed > 0 else 0
        
        print(f"\\n📊 Processing complete!")
        print(f"   Total puzzles: {total_processed:,}")
        print(f"   PGNs fetched: {successful_fetches:,} ({success_rate:.1f}%)")
        print(f"   Failed fetches: {failed_fetches:,}")
        
        # Save results
        print(f"\\n💾 Saving results to {self.output_file}...")
        with open(self.output_file, 'w') as f:
            for puzzle in processed_puzzles:
                f.write(json.dumps(puzzle) + '\\n')
        
        print(f"✅ Successfully saved {len(processed_puzzles):,} processed puzzles!")
        
        return processed_puzzles

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sample puzzles and fetch PGNs')
    parser.add_argument('--csv-file', required=True, help='Input CSV file with puzzles')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API requests (seconds)')
    
    args = parser.parse_args()
    
    processor = SamplePuzzleProcessor(args.csv_file, args.output)
    processor.request_delay = args.delay
    
    processed_puzzles = processor.process_puzzles()
    
    print(f"\\n🎉 Processing complete! Output: {args.output}")

if __name__ == "__main__":
    main()