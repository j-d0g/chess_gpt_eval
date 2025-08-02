#!/usr/bin/env python3
"""
Example usage of the Lichess puzzle dataset with PGN transcripts
"""

import json
import pandas as pd
from pathlib import Path
import chess.pgn
import io
from typing import List, Dict, Optional

def load_puzzles_with_pgn(data_file: str) -> List[Dict]:
    """Load processed puzzles with PGN data from JSONL file"""
    
    puzzles = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                puzzle = json.loads(line.strip())
                puzzles.append(puzzle)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    return puzzles

def filter_puzzles_by_criteria(puzzles: List[Dict], 
                             min_rating: int = 1000,
                             max_rating: int = 3000,
                             require_pgn: bool = True,
                             themes: Optional[List[str]] = None) -> List[Dict]:
    """Filter puzzles by various criteria"""
    
    filtered = []
    
    for puzzle in puzzles:
        # Rating filter
        rating = puzzle.get('rating', 0)
        if not (min_rating <= rating <= max_rating):
            continue
        
        # PGN requirement
        if require_pgn and not puzzle.get('pgn_valid', False):
            continue
        
        # Theme filter
        if themes:
            puzzle_themes = puzzle.get('themes', '').lower()
            if not any(theme.lower() in puzzle_themes for theme in themes):
                continue
        
        filtered.append(puzzle)
    
    return filtered

def analyze_puzzle_games(puzzles: List[Dict]) -> Dict:
    """Analyze the original games that puzzles came from"""
    
    analysis = {
        'total_puzzles': len(puzzles),
        'puzzles_with_pgn': 0,
        'unique_games': set(),
        'opening_distribution': {},
        'rating_stats': {
            'min': float('inf'),
            'max': 0,
            'total': 0,
            'count': 0
        },
        'theme_distribution': {},
        'move_count_distribution': {}
    }
    
    for puzzle in puzzles:
        # Count puzzles with PGN
        if puzzle.get('pgn_valid'):
            analysis['puzzles_with_pgn'] += 1
            
            # Track unique games
            game_id = puzzle.get('game_id')
            if game_id:
                analysis['unique_games'].add(game_id)
        
        # Rating statistics
        rating = puzzle.get('rating', 0)
        if rating > 0:
            analysis['rating_stats']['min'] = min(analysis['rating_stats']['min'], rating)
            analysis['rating_stats']['max'] = max(analysis['rating_stats']['max'], rating)
            analysis['rating_stats']['total'] += rating
            analysis['rating_stats']['count'] += 1
        
        # Opening distribution
        opening = puzzle.get('opening_tags', 'Unknown')
        if opening:
            analysis['opening_distribution'][opening] = analysis['opening_distribution'].get(opening, 0) + 1
        
        # Theme distribution
        themes = puzzle.get('themes', '').split()
        for theme in themes:
            if theme:
                analysis['theme_distribution'][theme] = analysis['theme_distribution'].get(theme, 0) + 1
        
        # Analyze move count in original games
        pgn_text = puzzle.get('pgn_text', '')
        if pgn_text:
            try:
                pgn_io = io.StringIO(pgn_text)
                game = chess.pgn.read_game(pgn_io)
                if game:
                    move_count = len(list(game.mainline_moves()))
                    move_bucket = (move_count // 10) * 10  # 10-move buckets
                    analysis['move_count_distribution'][move_bucket] = analysis['move_count_distribution'].get(move_bucket, 0) + 1
            except:
                pass
    
    # Calculate average rating
    if analysis['rating_stats']['count'] > 0:
        analysis['rating_stats']['average'] = analysis['rating_stats']['total'] / analysis['rating_stats']['count']
    
    # Convert unique games set to count
    analysis['unique_games'] = len(analysis['unique_games'])
    
    return analysis

def extract_game_context(puzzle: Dict, moves_before_puzzle: int = 5) -> Optional[str]:
    """Extract the game context leading up to the puzzle position"""
    
    pgn_text = puzzle.get('pgn_text', '')
    if not pgn_text:
        return None
    
    try:
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        if not game:
            return None
        
        # Get all moves
        moves = list(game.mainline_moves())
        move_texts = []
        
        board = game.board()
        for i, move in enumerate(moves):
            # Convert to SAN notation
            san_move = board.san(move)
            
            # Format as PGN with move numbers
            if board.turn == chess.WHITE:  # Before the move
                move_num = (i // 2) + 1
                move_texts.append(f"{move_num}. {san_move}")
            else:
                move_texts.append(san_move)
            
            board.push(move)
        
        # Take the last N moves before puzzle
        context_moves = move_texts[-moves_before_puzzle:] if len(move_texts) > moves_before_puzzle else move_texts
        
        return " ".join(context_moves)
        
    except Exception as e:
        print(f"Error extracting context: {e}")
        return None

def create_training_data_format(puzzles: List[Dict]) -> List[Dict]:
    """Convert puzzles to training data format"""
    
    training_data = []
    
    for puzzle in puzzles:
        if not puzzle.get('pgn_valid'):
            continue
        
        # Extract game context
        context = extract_game_context(puzzle)
        if not context:
            continue
        
        training_example = {
            'puzzle_id': puzzle['puzzle_id'],
            'fen': puzzle['fen'],
            'moves': puzzle['moves'],
            'rating': puzzle['rating'],
            'themes': puzzle.get('themes', ''),
            'game_context': context,
            'solution_moves': puzzle['moves'].split(),
            'full_pgn': puzzle['pgn_text']
        }
        
        training_data.append(training_example)
    
    return training_data

def main():
    """Example usage of the Lichess puzzle dataset"""
    
    # Configuration
    data_file = "data/lichess_puzzles/puzzles_with_pgn.jsonl"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run download_lichess_puzzles.py first to download and process the data")
        return
    
    print("🧩 Lichess Puzzle Dataset Analysis")
    print("=" * 50)
    
    # Load puzzles
    print("📖 Loading puzzles...")
    puzzles = load_puzzles_with_pgn(data_file)
    print(f"✅ Loaded {len(puzzles):,} puzzles")
    
    # Filter puzzles by criteria
    print("\n🔍 Filtering puzzles...")
    filtered_puzzles = filter_puzzles_by_criteria(
        puzzles,
        min_rating=1200,
        max_rating=2000,
        require_pgn=True,
        themes=['middlegame', 'endgame', 'tactics']
    )
    print(f"✅ Filtered to {len(filtered_puzzles):,} puzzles")
    
    # Analyze the dataset
    print("\n📊 Analyzing dataset...")
    analysis = analyze_puzzle_games(filtered_puzzles)
    
    print(f"Total puzzles: {analysis['total_puzzles']:,}")
    print(f"Puzzles with valid PGN: {analysis['puzzles_with_pgn']:,}")
    print(f"Unique games: {analysis['unique_games']:,}")
    
    if analysis['rating_stats']['count'] > 0:
        print(f"Rating range: {analysis['rating_stats']['min']}-{analysis['rating_stats']['max']}")
        print(f"Average rating: {analysis['rating_stats']['average']:.0f}")
    
    # Top themes
    print(f"\n🏷️  Top puzzle themes:")
    top_themes = sorted(analysis['theme_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for theme, count in top_themes:
        print(f"  {theme}: {count:,}")
    
    # Top openings
    print(f"\n♘ Top openings:")
    top_openings = sorted(analysis['opening_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for opening, count in top_openings:
        print(f"  {opening}: {count:,}")
    
    # Create training data sample
    print(f"\n🎯 Creating training data format...")
    training_data = create_training_data_format(filtered_puzzles[:100])  # Sample 100
    print(f"✅ Created {len(training_data)} training examples")
    
    # Show example
    if training_data:
        example = training_data[0]
        print(f"\n📝 Example training data:")
        print(f"Puzzle ID: {example['puzzle_id']}")
        print(f"Rating: {example['rating']}")
        print(f"Themes: {example['themes']}")
        print(f"Game context: {example['game_context']}")
        print(f"FEN: {example['fen']}")
        print(f"Solution: {' '.join(example['solution_moves'])}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()