#!/usr/bin/env python3
"""
Mass Stockfish Processor - High Performance Batch Analysis
Optimized for HPC systems with comprehensive analytics at low depth
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
import io
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import pandas as pd
import chess
import chess.engine
import chess.pgn
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import signal
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mass_stockfish_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockfishProcessor:
    """High-performance Stockfish processor for batch analysis with comprehensive analytics"""
    
    def __init__(self, stockfish_path: str = "./stockfish/stockfish-ubuntu-x86-64-avx2", 
                 nodes_per_position: int = 150000,
                 time_limit: float = 10.0):
        self.stockfish_path = stockfish_path
        self.nodes_per_position = nodes_per_position
        self.time_limit = time_limit
        
        # Verify Stockfish is available
        if not os.path.exists(stockfish_path):
            # Try common locations
            common_paths = [
                "./stockfish/stockfish-ubuntu-x86-64-avx2",  # Your local build (AVX2)
                "./stockfish/stockfish-ubuntu-x86-64-modern",  # Your local build (modern)
                "/usr/bin/stockfish",
                "/usr/local/bin/stockfish", 
                "./stockfish/stockfish",
                "stockfish"
            ]
            for path in common_paths:
                if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                    self.stockfish_path = path
                    break
            else:
                raise FileNotFoundError("Stockfish not found. Please install or specify path.")
        
        logger.info(f"Using Stockfish at: {self.stockfish_path}")
        logger.info(f"Analysis depth: {nodes_per_position} nodes per position")

    def calculate_material_imbalance(self, board: chess.Board) -> float:
        """Calculate material imbalance between sides"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return abs(white_material - black_material)

    def calculate_position_complexity(self, multipv_info: List[Dict], board: chess.Board) -> float:
        """
        Calculate position complexity based on:
        - Evaluation spread of top moves
        - Number of reasonable moves
        - Material imbalance
        """
        if len(multipv_info) < 2:
            return 0
        
        # Evaluation spread
        evals = []
        for info in multipv_info[:5]:
            score = info.get('score')
            if score:
                if hasattr(score, 'relative'):
                    cp_score = score.relative.score(mate_score=10000)
                elif hasattr(score, 'white'):
                    cp_score = score.white().score(mate_score=10000)
                else:
                    cp_score = 0
                evals.append(cp_score)
        
        if len(evals) < 2:
            return 0
            
        top_eval = evals[0]
        eval_spread = np.std(evals)
        
        # Number of reasonable moves (within 50 centipawns of best)
        reasonable_moves = sum(1 for e in evals if abs(e - top_eval) < 50)
        
        # Material imbalance
        material_imbalance = self.calculate_material_imbalance(board)
        
        # Normalize and combine
        complexity = (
            min(eval_spread / 100, 1) * 0.4 +  # Eval spread component
            min(reasonable_moves / 5, 1) * 0.3 +  # Choice richness
            min(material_imbalance / 10, 1) * 0.3  # Material complexity
        ) * 100
        
        return complexity

    def analyze_single_game(self, game_data: Dict) -> Dict:
        """Analyze a single game with Stockfish"""
        try:
            game_id = game_data.get('game_id', 'unknown')
            transcript = game_data.get('transcript', '')
            
            if not transcript:
                return {'game_id': game_id, 'error': 'No transcript'}
            
            # Parse PGN
            try:
                game = chess.pgn.read_game(io.StringIO(transcript))
                if game is None:
                    return {'game_id': game_id, 'error': 'Failed to parse PGN'}
            except Exception as e:
                return {'game_id': game_id, 'error': f'PGN parse error: {str(e)}'}
            
            # Initialize engine
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                # Configure engine for high-depth analysis
                engine.configure({
                    "Hash": 512,  # MB
                    "Threads": 1,  # Per worker thread
                })
                
                board = game.board()
                moves_analysis = []
                move_number = 1
                previous_score = None
                evaluation_curve = []
                complexity_curve = []
                
                # Analyze each position
                for move in game.mainline_moves():
                    try:
                        # Analyze position before move
                        info = engine.analyse(
                            board, 
                            chess.engine.Limit(nodes=self.nodes_per_position),
                            multipv=5  # Get top 5 moves for complexity analysis
                        )
                        
                        # Extract analysis data
                        analysis = self.extract_analysis_data(info, board, move, move_number)
                        
                        # Add centipawn loss and move classification
                        if previous_score is not None and analysis.get('centipawn_score') is not None:
                            current_score = analysis['centipawn_score']
                            # Adjust for turn (black scores are negated)
                            if board.turn == chess.BLACK:
                                centipawn_loss = previous_score - (-current_score)
                            else:
                                centipawn_loss = previous_score - current_score
                            
                            analysis['centipawn_loss'] = max(0, centipawn_loss)  # Only positive losses
                            
                            # Classify move by centipawn loss
                            if centipawn_loss > 300:
                                analysis['move_classification'] = 'blunder'
                            elif centipawn_loss > 100:
                                analysis['move_classification'] = 'mistake'
                            elif centipawn_loss > 50:
                                analysis['move_classification'] = 'inaccuracy'
                            else:
                                analysis['move_classification'] = 'good'
                        else:
                            analysis['centipawn_loss'] = 0
                            analysis['move_classification'] = 'unknown'
                        
                        # Add comprehensive analytics
                        analysis['material_imbalance'] = self.calculate_material_imbalance(board)
                        analysis['position_complexity'] = self.calculate_position_complexity(info, board)
                        
                        # Calculate move quality score (0-100)
                        if analysis.get('move_quality') == 'best':
                            analysis['move_quality_score'] = 100
                        else:
                            cp_loss = analysis.get('centipawn_loss', 0)
                            analysis['move_quality_score'] = max(0, 100 - cp_loss)
                        
                        moves_analysis.append(analysis)
                        previous_score = analysis.get('centipawn_score')
                        
                        # Track curves for game-level analysis
                        evaluation_curve.append(analysis.get('centipawn_score', 0))
                        complexity_curve.append(analysis.get('position_complexity', 0))
                        
                        # Make the move
                        board.push(move)
                        move_number += 1
                        
                    except Exception as e:
                        logger.warning(f"Analysis error for game {game_id}, move {move_number}: {str(e)}")
                        moves_analysis.append({
                            'move_number': move_number,
                            'move': str(move),
                            'error': str(e)
                        })
                        board.push(move)
                        move_number += 1
            
            # Calculate comprehensive game-level statistics
            game_stats = self.calculate_comprehensive_game_statistics(moves_analysis, game_data, evaluation_curve, complexity_curve)
            
            return {
                'game_id': game_id,
                'moves_analysis': moves_analysis,
                'game_statistics': game_stats,
                'total_moves': len(moves_analysis),
                'evaluation_curve': evaluation_curve,
                'complexity_curve': complexity_curve,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Critical error analyzing game {game_data.get('game_id', 'unknown')}: {str(e)}")
            return {
                'game_id': game_data.get('game_id', 'unknown'),
                'error': f'Critical analysis error: {str(e)}'
            }

    def extract_analysis_data(self, info: chess.engine.InfoDict, board: chess.Board, 
                            move: chess.Move, move_number: int) -> Dict:
        """Extract comprehensive analysis data from Stockfish output"""
        try:
            analysis = {
                'move_number': move_number,
                'move': str(move),
                'fen': board.fen(),
                'turn': 'white' if board.turn else 'black'
            }
            
            # Handle different info formats
            if isinstance(info, list):
                # MultiPV results
                best_info = info[0] if info else {}
            else:
                best_info = info
            
            # Extract score
            score = best_info.get('score')
            if score:
                if hasattr(score, 'relative'):
                    cp_score = score.relative.score(mate_score=10000)
                elif hasattr(score, 'white'):
                    cp_score = score.white().score(mate_score=10000)
                else:
                    cp_score = None
                
                analysis['centipawn_score'] = cp_score
                analysis['is_mate'] = score.is_mate() if hasattr(score, 'is_mate') else False
                if analysis['is_mate']:
                    analysis['mate_in'] = score.mate() if hasattr(score, 'mate') else None
            
            # Extract best move and PV
            pv = best_info.get('pv', [])
            if pv:
                analysis['best_move'] = str(pv[0])
                analysis['principal_variation'] = [str(m) for m in pv[:5]]  # First 5 moves
            
            # Extract depth and nodes
            analysis['depth'] = best_info.get('depth', 0)
            analysis['nodes'] = best_info.get('nodes', 0)
            analysis['time'] = best_info.get('time', 0)
            analysis['nps'] = best_info.get('nps', 0)
            
            # Move quality assessment
            if move in [m for m in pv[:3]]:  # Move is in top 3
                if str(move) == str(pv[0]):
                    analysis['move_quality'] = 'best'
                else:
                    analysis['move_quality'] = 'good'
            else:
                analysis['move_quality'] = 'suboptimal'
            
            # Position complexity (number of legal moves)
            analysis['legal_moves_count'] = len(list(board.legal_moves))
            
            # Game phase detection
            piece_count = len([p for p in str(board).replace('\n', '').replace(' ', '') if p != '.'])
            if piece_count >= 28:
                analysis['game_phase'] = 'opening'
            elif piece_count >= 16:
                analysis['game_phase'] = 'middlegame'
            else:
                analysis['game_phase'] = 'endgame'
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error extracting analysis data: {str(e)}")
            return {
                'move_number': move_number,
                'move': str(move),
                'error': str(e)
            }

    def calculate_comprehensive_game_statistics(self, moves_analysis: List[Dict], game_data: Dict, 
                                              evaluation_curve: List[float], complexity_curve: List[float]) -> Dict:
        """Calculate comprehensive game-level statistics with phase analysis"""
        try:
            stats = {
                'total_moves': len(moves_analysis),
                'average_depth': 0,
                'total_nodes': 0,
                'total_time': 0,
                'blunders': 0,
                'mistakes': 0,
                'inaccuracies': 0,
                'best_moves': 0,
                'good_moves': 0,
                'suboptimal_moves': 0
            }
            
            valid_moves = [m for m in moves_analysis if 'error' not in m]
            if not valid_moves:
                return stats
            
            # Calculate averages
            depths = [m.get('depth', 0) for m in valid_moves]
            nodes = [m.get('nodes', 0) for m in valid_moves]
            times = [m.get('time', 0) for m in valid_moves]
            
            stats['average_depth'] = np.mean(depths) if depths else 0
            stats['total_nodes'] = sum(nodes)
            stats['total_time'] = sum(times)
            
            # Count move qualities and classifications
            for move in valid_moves:
                # Count by move quality (best/good/suboptimal)
                quality = move.get('move_quality', 'unknown')
                if quality == 'best':
                    stats['best_moves'] += 1
                elif quality == 'good':
                    stats['good_moves'] += 1
                elif quality == 'suboptimal':
                    stats['suboptimal_moves'] += 1
                
                # Count by move classification (blunder/mistake/inaccuracy)
                classification = move.get('move_classification', 'unknown')
                if classification == 'blunder':
                    stats['blunders'] += 1
                elif classification == 'mistake':
                    stats['mistakes'] += 1
                elif classification == 'inaccuracy':
                    stats['inaccuracies'] += 1
            
            # Calculate average centipawn loss
            cp_losses = [m.get('centipawn_loss', 0) for m in valid_moves if m.get('centipawn_loss') is not None]
            stats['average_centipawn_loss'] = np.mean(cp_losses) if cp_losses else 0
            
            # Game phase analysis
            opening_moves = [m for m in valid_moves if m.get('game_phase') == 'opening']
            middlegame_moves = [m for m in valid_moves if m.get('game_phase') == 'middlegame']
            endgame_moves = [m for m in valid_moves if m.get('game_phase') == 'endgame']
            
            # Opening analysis
            if opening_moves:
                stats['opening_accuracy'] = np.mean([m.get('move_quality_score', 0) for m in opening_moves])
                stats['opening_blunders'] = sum(1 for m in opening_moves if m.get('move_classification') == 'blunder')
                stats['opening_avg_complexity'] = np.mean([m.get('position_complexity', 0) for m in opening_moves])
            else:
                stats['opening_accuracy'] = 0
                stats['opening_blunders'] = 0
                stats['opening_avg_complexity'] = 0
            
            # Middlegame analysis
            if middlegame_moves:
                stats['middlegame_accuracy'] = np.mean([m.get('move_quality_score', 0) for m in middlegame_moves])
                stats['middlegame_blunders'] = sum(1 for m in middlegame_moves if m.get('move_classification') == 'blunder')
                stats['middlegame_avg_complexity'] = np.mean([m.get('position_complexity', 0) for m in middlegame_moves])
            else:
                stats['middlegame_accuracy'] = 0
                stats['middlegame_blunders'] = 0
                stats['middlegame_avg_complexity'] = 0
            
            # Endgame analysis
            if endgame_moves:
                stats['endgame_accuracy'] = np.mean([m.get('move_quality_score', 0) for m in endgame_moves])
                stats['endgame_blunders'] = sum(1 for m in endgame_moves if m.get('move_classification') == 'blunder')
                stats['endgame_avg_complexity'] = np.mean([m.get('position_complexity', 0) for m in endgame_moves])
            else:
                stats['endgame_accuracy'] = 0
                stats['endgame_blunders'] = 0
                stats['endgame_avg_complexity'] = 0
            
            # Critical moments detection
            critical_moments = []
            for i in range(1, len(valid_moves)):
                # Large evaluation swings
                if i < len(evaluation_curve) and (i-1) < len(evaluation_curve):
                    eval_swing = abs(evaluation_curve[i] - evaluation_curve[i-1])
                    if eval_swing > 200:
                        critical_moments.append(i)
                
                # Blunders
                if valid_moves[i].get('move_classification') == 'blunder':
                    critical_moments.append(i)
                
                # High complexity positions
                if valid_moves[i].get('position_complexity', 0) > 80:
                    critical_moments.append(i)
            
            stats['critical_moments'] = len(set(critical_moments))
            
            # Overall complexity and evaluation metrics
            stats['average_position_complexity'] = np.mean(complexity_curve) if complexity_curve else 0
            stats['max_position_complexity'] = max(complexity_curve) if complexity_curve else 0
            stats['evaluation_volatility'] = np.std(evaluation_curve) if len(evaluation_curve) > 1 else 0
            
            # Add original game data
            stats.update({
                'result': game_data.get('result', 'unknown'),
                'illegal_moves': game_data.get('illegal_moves', 0),
                'original_move_count': game_data.get('number_of_moves', 0)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating game statistics: {str(e)}")
            return {'error': str(e)}

def process_game_chunk(args):
    """Process a chunk of games - designed for multiprocessing"""
    chunk_data, chunk_id, stockfish_path, nodes_per_position, progress_queue = args
    
    processor = StockfishProcessor(stockfish_path, nodes_per_position)
    results = []
    
    for i, game_data in enumerate(chunk_data):
        try:
            result = processor.analyze_single_game(game_data)
            results.append(result)
            
            # Update progress
            if progress_queue:
                progress_queue.put(1)
                
        except Exception as e:
            logger.error(f"Error processing game in chunk {chunk_id}: {str(e)}")
            results.append({
                'game_id': game_data.get('game_id', f'chunk_{chunk_id}_game_{i}'),
                'error': str(e)
            })
    
    return results

def load_csv_data(csv_file: str) -> List[Dict]:
    """Load game data from CSV file"""
    logger.info(f"Loading data from {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} games from CSV")
        
        # Convert to list of dictionaries
        games = []
        for idx, row in df.iterrows():
            game_data = {
                'game_id': f"{Path(csv_file).stem}_{idx}",
                'transcript': row.get('transcript', ''),
                'result': row.get('result', 'unknown'),
                'illegal_moves': row.get('illegal_moves', 0),
                'number_of_moves': row.get('number_of_moves', 0)
            }
            
            # Add any additional columns
            for col in df.columns:
                if col not in ['transcript', 'result', 'illegal_moves', 'number_of_moves']:
                    game_data[col] = row.get(col)
            
            games.append(game_data)
        
        return games
        
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_file}: {str(e)}")
        return []

def save_results(results: List[Dict], output_dir: str, filename_prefix: str):
    """Save analysis results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed JSON results
    json_file = output_path / f"{filename_prefix}_detailed_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {json_file}")
    
    # Save move-by-move CSV
    moves_csv_file = output_path / f"{filename_prefix}_moves_{timestamp}.csv"
    moves_data = []
    
    for result in results:
        if 'error' not in result and 'moves_analysis' in result:
            game_id = result['game_id']
            for move_data in result['moves_analysis']:
                if 'error' not in move_data:
                    moves_data.append({
                        'game_id': game_id,
                        'move_number': move_data.get('move_number', 0),
                        'move': move_data.get('move', ''),
                        'fen': move_data.get('fen', ''),
                        'turn': move_data.get('turn', ''),
                        'centipawn_score': move_data.get('centipawn_score', 0),
                        'centipawn_loss': move_data.get('centipawn_loss', 0),
                        'best_move': move_data.get('best_move', ''),
                        'principal_variation': '|'.join(move_data.get('principal_variation', [])),
                        'move_quality': move_data.get('move_quality', ''),
                        'move_classification': move_data.get('move_classification', ''),
                        'depth': move_data.get('depth', 0),
                        'nodes': move_data.get('nodes', 0),
                        'time': move_data.get('time', 0),
                        'legal_moves_count': move_data.get('legal_moves_count', 0),
                        'game_phase': move_data.get('game_phase', ''),
                        'is_mate': move_data.get('is_mate', False),
                        'mate_in': move_data.get('mate_in', ''),
                        'material_imbalance': move_data.get('material_imbalance', 0),
                        'position_complexity': move_data.get('position_complexity', 0),
                        'move_quality_score': move_data.get('move_quality_score', 0)
                    })
    
    if moves_data:
        moves_df = pd.DataFrame(moves_data)
        moves_df.to_csv(moves_csv_file, index=False)
        logger.info(f"Saved move-by-move analysis to {moves_csv_file}")
    
    # Save summary CSV
    csv_file = output_path / f"{filename_prefix}_summary_{timestamp}.csv"
    summary_data = []
    
    for result in results:
        if 'error' in result:
            summary_data.append({
                'game_id': result['game_id'],
                'error': result['error']
            })
        else:
            stats = result.get('game_statistics', {})
            summary_row = {
                'game_id': result['game_id'],
                'total_moves': result.get('total_moves', 0),
                'analysis_timestamp': result.get('analysis_timestamp', ''),
                **stats
            }
            summary_data.append(summary_row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(csv_file, index=False)
    logger.info(f"Saved summary to {csv_file}")
    
    return json_file, csv_file

def main():
    parser = argparse.ArgumentParser(description='Mass Stockfish Processor')
    parser.add_argument('--input-dir', default='logs', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='stockfish_analysis_results', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--chunk-size', type=int, default=100, help='Games per chunk')
    parser.add_argument('--nodes', type=int, default=150000, help='Stockfish nodes per position')
    parser.add_argument('--stockfish-path', default='/usr/bin/stockfish', help='Path to Stockfish binary')
    parser.add_argument('--files', nargs='+', help='Specific CSV files to process')
    parser.add_argument('--max-games', type=int, help='Maximum games to process per file')
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.workers is None:
        # Use 80% of available CPUs, leaving some for system
        args.workers = max(1, int(cpu_count() * 0.8))
    
    logger.info(f"Starting mass Stockfish processing with {args.workers} workers")
    logger.info(f"Analysis depth: {args.nodes} nodes per position")
    logger.info(f"Chunk size: {args.chunk_size} games")
    
    # Find CSV files to process
    if args.files:
        csv_files = args.files
    else:
        input_path = Path(args.input_dir)
        csv_files = list(input_path.glob("*.csv"))
        csv_files = [str(f) for f in csv_files if f.stat().st_size > 1000]  # Skip tiny files
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    total_start_time = time.time()
    
    for csv_file in csv_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {csv_file}")
        logger.info(f"{'='*60}")
        
        file_start_time = time.time()
        
        # Load game data
        games = load_csv_data(csv_file)
        if not games:
            logger.warning(f"No games loaded from {csv_file}, skipping")
            continue
        
        # Limit games if specified
        if args.max_games and len(games) > args.max_games:
            games = games[:args.max_games]
            logger.info(f"Limited to {args.max_games} games")
        
        # Split into chunks for multiprocessing
        chunks = []
        for i in range(0, len(games), args.chunk_size):
            chunk = games[i:i + args.chunk_size]
            chunks.append(chunk)
        
        logger.info(f"Split {len(games)} games into {len(chunks)} chunks")
        
        # Set up progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        
        # Prepare arguments for multiprocessing
        chunk_args = [
            (chunk, i, args.stockfish_path, args.nodes, progress_queue)
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        results = []
        with Pool(processes=args.workers) as pool:
            # Start progress tracking
            total_games = len(games)
            pbar = tqdm(total=total_games, desc=f"Analyzing {Path(csv_file).name}")
            
            # Process chunks
            chunk_results = pool.map_async(process_game_chunk, chunk_args)
            
            # Update progress bar
            completed = 0
            while not chunk_results.ready():
                try:
                    while not progress_queue.empty():
                        progress_queue.get()
                        completed += 1
                        pbar.update(1)
                except:
                    pass
                time.sleep(0.1)
            
            # Get final results
            all_chunk_results = chunk_results.get()
            pbar.close()
            
            # Flatten results
            for chunk_result in all_chunk_results:
                results.extend(chunk_result)
        
        # Save results
        filename_prefix = Path(csv_file).stem
        json_file, csv_file_out = save_results(results, args.output_dir, filename_prefix)
        
        file_time = time.time() - file_start_time
        logger.info(f"Completed {csv_file} in {file_time:.2f} seconds")
        logger.info(f"Processed {len(results)} games")
        logger.info(f"Results saved to {json_file} and {csv_file_out}")
    
    total_time = time.time() - total_start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"MASS PROCESSING COMPLETE")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Processed {len(csv_files)} files")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()