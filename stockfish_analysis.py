#!/usr/bin/env python3
"""
Stockfish Analysis Module for Chess GPT Evaluation
Provides deep analysis of games using Stockfish engine
"""

import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
from collections import defaultdict
import re
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


@dataclass
class MoveAnalysis:
    """Analysis data for a single move"""
    move: str
    move_uci: str
    fen_before: str
    fen_after: str
    evaluation_before: float
    evaluation_after: float
    best_move: str
    best_move_eval: float
    centipawn_loss: float
    is_blunder: bool
    is_mistake: bool
    is_inaccuracy: bool
    is_best_move: bool
    move_quality_score: float
    complexity_score: float
    time_pressure: bool
    alternatives: List[Tuple[str, float]]


@dataclass
class GameAnalysis:
    """Complete analysis for a game"""
    game_id: str
    moves: List[MoveAnalysis]
    average_centipawn_loss: float
    blunder_count: int
    mistake_count: int
    inaccuracy_count: int
    perfect_move_count: int
    game_phases: Dict[str, Dict]
    critical_moments: List[int]
    opening_accuracy: float
    middlegame_accuracy: float
    endgame_accuracy: float
    complexity_curve: List[float]
    evaluation_curve: List[float]


class StockfishAnalyzer:
    """Advanced Stockfish analysis for chess games"""
    
    def __init__(self, stockfish_path: Optional[str] = None, 
                 depth: int = 20, 
                 threads: int = 4,
                 hash_mb: int = 1024):
        """
        Initialize Stockfish analyzer
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth
            threads: Number of threads for Stockfish
            hash_mb: Hash table size in MB
        """
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        
        # Thresholds for move classification
        self.blunder_threshold = 200  # centipawns
        self.mistake_threshold = 100
        self.inaccuracy_threshold = 50
        
    def _find_stockfish(self) -> str:
        """Find Stockfish executable"""
        import platform
        
        if platform.system() == "Linux":
            paths = [
                "./stockfish/stockfish-ubuntu-x86-64-modern",
                "/usr/games/stockfish",
                "/usr/local/bin/stockfish"
            ]
        elif platform.system() == "Darwin":
            paths = [
                "/usr/local/bin/stockfish",
                "/opt/homebrew/bin/stockfish",
                "stockfish"
            ]
        elif platform.system() == "Windows":
            paths = [
                "stockfish.exe",
                r"C:\Program Files\Stockfish\stockfish.exe"
            ]
        else:
            paths = ["stockfish"]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Stockfish not found. Please install or specify path.")
    
    def analyze_game(self, pgn_string: str, game_id: str = "") -> GameAnalysis:
        """
        Analyze a complete game
        
        Args:
            pgn_string: PGN format game string
            game_id: Unique game identifier
            
        Returns:
            GameAnalysis object with complete analysis
        """
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            # Configure engine
            engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb
            })
            
            # Parse game
            board = chess.Board()
            moves = self._parse_pgn_moves(pgn_string)
            
            move_analyses = []
            evaluation_curve = []
            complexity_curve = []
            
            # Analyze each position
            for move_num, (move_san, move_uci) in enumerate(moves):
                # Get position before move
                fen_before = board.fen()
                
                # Analyze position
                info_before = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                eval_before = self._score_to_centipawns(info_before['score'])
                
                # Get best move and alternatives
                multipv_info = engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=5)
                best_move = board.san(multipv_info[0]['pv'][0])
                best_eval = self._score_to_centipawns(multipv_info[0]['score'])
                
                alternatives = []
                for i in range(1, min(5, len(multipv_info))):
                    alt_move = board.san(multipv_info[i]['pv'][0])
                    alt_eval = self._score_to_centipawns(multipv_info[i]['score'])
                    alternatives.append((alt_move, alt_eval))
                
                # Make the actual move
                try:
                    board.push(move_uci)
                except:
                    # Handle illegal moves
                    break
                
                fen_after = board.fen()
                
                # Analyze position after move
                info_after = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                eval_after = self._score_to_centipawns(info_after['score'])
                
                # Calculate centipawn loss
                if board.turn == chess.BLACK:  # Just moved was white
                    centipawn_loss = eval_before - eval_after
                else:  # Just moved was black
                    centipawn_loss = eval_after - eval_before
                
                # Classify move quality
                is_best = move_san == best_move
                is_blunder = centipawn_loss > self.blunder_threshold
                is_mistake = self.mistake_threshold < centipawn_loss <= self.blunder_threshold
                is_inaccuracy = self.inaccuracy_threshold < centipawn_loss <= self.mistake_threshold
                
                # Calculate move quality score (0-100)
                if is_best:
                    quality_score = 100
                else:
                    quality_score = max(0, 100 - centipawn_loss)
                
                # Calculate position complexity
                complexity = self._calculate_complexity(multipv_info, board)
                
                # Check time pressure (arbitrary: after move 30)
                time_pressure = move_num > 60
                
                move_analysis = MoveAnalysis(
                    move=move_san,
                    move_uci=str(move_uci),
                    fen_before=fen_before,
                    fen_after=fen_after,
                    evaluation_before=eval_before,
                    evaluation_after=eval_after,
                    best_move=best_move,
                    best_move_eval=best_eval,
                    centipawn_loss=centipawn_loss,
                    is_blunder=is_blunder,
                    is_mistake=is_mistake,
                    is_inaccuracy=is_inaccuracy,
                    is_best_move=is_best,
                    move_quality_score=quality_score,
                    complexity_score=complexity,
                    time_pressure=time_pressure,
                    alternatives=alternatives
                )
                
                move_analyses.append(move_analysis)
                evaluation_curve.append(eval_after)
                complexity_curve.append(complexity)
            
            # Calculate game-level statistics
            if move_analyses:
                avg_cpl = np.mean([m.centipawn_loss for m in move_analyses])
                blunders = sum(m.is_blunder for m in move_analyses)
                mistakes = sum(m.is_mistake for m in move_analyses)
                inaccuracies = sum(m.is_inaccuracy for m in move_analyses)
                perfect_moves = sum(m.is_best_move for m in move_analyses)
                
                # Analyze by game phase
                game_phases = self._analyze_game_phases(move_analyses)
                
                # Find critical moments
                critical_moments = self._find_critical_moments(move_analyses)
                
                # Calculate phase accuracies
                opening_moves = move_analyses[:20]
                middlegame_moves = move_analyses[20:40]
                endgame_moves = move_analyses[40:]
                
                opening_acc = np.mean([m.move_quality_score for m in opening_moves]) if opening_moves else 0
                middlegame_acc = np.mean([m.move_quality_score for m in middlegame_moves]) if middlegame_moves else 0
                endgame_acc = np.mean([m.move_quality_score for m in endgame_moves]) if endgame_moves else 0
            else:
                avg_cpl = 0
                blunders = mistakes = inaccuracies = perfect_moves = 0
                game_phases = {}
                critical_moments = []
                opening_acc = middlegame_acc = endgame_acc = 0
            
            return GameAnalysis(
                game_id=game_id,
                moves=move_analyses,
                average_centipawn_loss=avg_cpl,
                blunder_count=blunders,
                mistake_count=mistakes,
                inaccuracy_count=inaccuracies,
                perfect_move_count=perfect_moves,
                game_phases=game_phases,
                critical_moments=critical_moments,
                opening_accuracy=opening_acc,
                middlegame_accuracy=middlegame_acc,
                endgame_accuracy=endgame_acc,
                complexity_curve=complexity_curve,
                evaluation_curve=evaluation_curve
            )
    
    def _score_to_centipawns(self, score) -> float:
        """Convert engine score to centipawns"""
        if score.is_mate():
            # Convert mate scores to large centipawn values
            mate_in = score.mate()
            if mate_in > 0:
                return 10000 - mate_in * 100  # Positive for white advantage
            else:
                return -10000 - mate_in * 100  # Negative for black advantage
        else:
            return score.score()
    
    def _parse_pgn_moves(self, pgn_string: str) -> List[Tuple[str, chess.Move]]:
        """Parse PGN string and return list of (SAN, UCI) moves"""
        moves = []
        board = chess.Board()
        
        # Clean PGN string
        pgn_string = pgn_string.strip()
        if pgn_string.startswith(';'):
            pgn_string = pgn_string[1:]
        
        # Parse moves
        move_pattern = re.compile(r'\d+\.\s*(\S+)(?:\s+(\S+))?')
        matches = move_pattern.findall(pgn_string)
        
        for white_move, black_move in matches:
            try:
                # White move
                move = board.parse_san(white_move)
                moves.append((white_move, move))
                board.push(move)
                
                # Black move (if exists)
                if black_move and black_move not in ['1-0', '0-1', '1/2-1/2', '*']:
                    move = board.parse_san(black_move)
                    moves.append((black_move, move))
                    board.push(move)
            except:
                break
        
        return moves
    
    def _calculate_complexity(self, multipv_info: List[Dict], board: chess.Board) -> float:
        """
        Calculate position complexity based on:
        - Evaluation spread of top moves
        - Number of reasonable moves
        - Material imbalance
        - King safety
        """
        if len(multipv_info) < 2:
            return 0
        
        # Evaluation spread
        top_eval = self._score_to_centipawns(multipv_info[0]['score'])
        evals = [self._score_to_centipawns(info['score']) for info in multipv_info[:5]]
        eval_spread = np.std(evals) if len(evals) > 1 else 0
        
        # Number of reasonable moves (within 50 centipawns of best)
        reasonable_moves = sum(1 for e in evals if abs(e - top_eval) < 50)
        
        # Material imbalance
        material_imbalance = self._calculate_material_imbalance(board)
        
        # Normalize and combine
        complexity = (
            min(eval_spread / 100, 1) * 0.4 +  # Eval spread component
            min(reasonable_moves / 5, 1) * 0.3 +  # Choice richness
            min(material_imbalance / 10, 1) * 0.3  # Material complexity
        ) * 100
        
        return complexity
    
    def _calculate_material_imbalance(self, board: chess.Board) -> float:
        """Calculate material imbalance"""
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
    
    def _analyze_game_phases(self, moves: List[MoveAnalysis]) -> Dict[str, Dict]:
        """Analyze performance in different game phases"""
        phases = {
            'opening': {'moves': moves[:20], 'phase': 'Opening'},
            'middlegame': {'moves': moves[20:40], 'phase': 'Middlegame'},
            'endgame': {'moves': moves[40:], 'phase': 'Endgame'}
        }
        
        results = {}
        for phase_name, phase_data in phases.items():
            phase_moves = phase_data['moves']
            if phase_moves:
                results[phase_name] = {
                    'average_quality': np.mean([m.move_quality_score for m in phase_moves]),
                    'average_cpl': np.mean([m.centipawn_loss for m in phase_moves]),
                    'blunders': sum(m.is_blunder for m in phase_moves),
                    'mistakes': sum(m.is_mistake for m in phase_moves),
                    'perfect_moves': sum(m.is_best_move for m in phase_moves),
                    'complexity': np.mean([m.complexity_score for m in phase_moves])
                }
            else:
                results[phase_name] = {
                    'average_quality': 0,
                    'average_cpl': 0,
                    'blunders': 0,
                    'mistakes': 0,
                    'perfect_moves': 0,
                    'complexity': 0
                }
        
        return results
    
    def _find_critical_moments(self, moves: List[MoveAnalysis]) -> List[int]:
        """Find critical moments in the game"""
        critical_moments = []
        
        for i in range(1, len(moves)):
            # Large evaluation swings
            eval_swing = abs(moves[i].evaluation_after - moves[i-1].evaluation_after)
            if eval_swing > 200:
                critical_moments.append(i)
            
            # Blunders
            if moves[i].is_blunder:
                critical_moments.append(i)
            
            # High complexity positions
            if moves[i].complexity_score > 80:
                critical_moments.append(i)
        
        return sorted(list(set(critical_moments)))
    
    def batch_analyze_games(self, games_df: pd.DataFrame, 
                          output_file: str,
                          num_processes: int = 4,
                          sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze multiple games in parallel
        
        Args:
            games_df: DataFrame with 'game_id' and 'transcript' columns
            output_file: Path to save analysis results
            num_processes: Number of parallel processes
            sample_size: Optional sample size for testing
            
        Returns:
            DataFrame with analysis results
        """
        # Sample games if requested
        if sample_size and len(games_df) > sample_size:
            games_df = games_df.sample(n=sample_size, random_state=42)
        
        print(f"Analyzing {len(games_df)} games with {num_processes} processes...")
        
        # Prepare game data
        game_data = [(row['game_id'], row['transcript']) 
                     for _, row in games_df.iterrows()]
        
        # Analyze games in parallel
        with mp.Pool(num_processes) as pool:
            analyses = list(tqdm(
                pool.starmap(self._analyze_game_wrapper, game_data),
                total=len(game_data),
                desc="Analyzing games"
            ))
        
        # Convert to DataFrame
        analysis_records = []
        for analysis in analyses:
            if analysis:
                record = {
                    'game_id': analysis.game_id,
                    'average_centipawn_loss': analysis.average_centipawn_loss,
                    'blunder_count': analysis.blunder_count,
                    'mistake_count': analysis.mistake_count,
                    'inaccuracy_count': analysis.inaccuracy_count,
                    'perfect_move_count': analysis.perfect_move_count,
                    'opening_accuracy': analysis.opening_accuracy,
                    'middlegame_accuracy': analysis.middlegame_accuracy,
                    'endgame_accuracy': analysis.endgame_accuracy,
                    'critical_moments': len(analysis.critical_moments),
                    'move_count': len(analysis.moves)
                }
                
                # Add phase-specific data
                for phase, stats in analysis.game_phases.items():
                    for stat, value in stats.items():
                        record[f'{phase}_{stat}'] = value
                
                analysis_records.append(record)
        
        analysis_df = pd.DataFrame(analysis_records)
        
        # Save results
        analysis_df.to_csv(output_file, index=False)
        print(f"Analysis saved to {output_file}")
        
        # Also save detailed analysis as JSON
        detailed_file = output_file.replace('.csv', '_detailed.json')
        detailed_data = []
        for analysis in analyses:
            if analysis:
                detailed_data.append({
                    'game_id': analysis.game_id,
                    'moves': [
                        {
                            'move': m.move,
                            'centipawn_loss': m.centipawn_loss,
                            'is_best': m.is_best_move,
                            'quality_score': m.move_quality_score,
                            'complexity': m.complexity_score
                        }
                        for m in analysis.moves
                    ],
                    'evaluation_curve': analysis.evaluation_curve,
                    'complexity_curve': analysis.complexity_curve,
                    'critical_moments': analysis.critical_moments
                })
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        return analysis_df
    
    def _analyze_game_wrapper(self, game_id: str, transcript: str) -> Optional[GameAnalysis]:
        """Wrapper for parallel processing"""
        try:
            return self.analyze_game(transcript, game_id)
        except Exception as e:
            print(f"Error analyzing game {game_id}: {e}")
            return None
    
    def create_analysis_visualizations(self, analysis_df: pd.DataFrame, 
                                     output_dir: str = "stockfish_analysis"):
        """Create visualizations from Stockfish analysis"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Centipawn Loss Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(analysis_df['average_centipawn_loss'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Average Centipawn Loss')
        plt.ylabel('Number of Games')
        plt.title('Distribution of Average Centipawn Loss')
        plt.axvline(analysis_df['average_centipawn_loss'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {analysis_df["average_centipawn_loss"].mean():.1f}')
        plt.legend()
        plt.savefig(f'{output_dir}/centipawn_loss_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error Type Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        error_types = ['blunder_count', 'mistake_count', 'inaccuracy_count']
        error_means = [analysis_df[col].mean() for col in error_types]
        
        ax1.bar(['Blunders', 'Mistakes', 'Inaccuracies'], error_means, 
                color=['red', 'orange', 'yellow'], alpha=0.7)
        ax1.set_ylabel('Average per Game')
        ax1.set_title('Average Errors per Game')
        
        # Perfect moves vs errors
        ax2.scatter(analysis_df['perfect_move_count'], 
                   analysis_df['blunder_count'] + analysis_df['mistake_count'],
                   alpha=0.5)
        ax2.set_xlabel('Perfect Moves')
        ax2.set_ylabel('Blunders + Mistakes')
        ax2.set_title('Perfect Moves vs Errors')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Phase Performance
        phase_cols = ['opening_accuracy', 'middlegame_accuracy', 'endgame_accuracy']
        phase_data = analysis_df[phase_cols].mean()
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Opening', 'Middlegame', 'Endgame'], phase_data, 
                color=['green', 'blue', 'purple'], alpha=0.7)
        plt.ylabel('Average Accuracy')
        plt.title('Performance by Game Phase')
        plt.ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(phase_data):
            plt.text(i, v + 1, f'{v:.1f}', ha='center')
        
        plt.savefig(f'{output_dir}/phase_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Complexity Analysis
        complexity_cols = [col for col in analysis_df.columns if 'complexity' in col]
        if complexity_cols:
            plt.figure(figsize=(12, 6))
            
            # Average complexity by phase
            phase_complexity = []
            for phase in ['opening', 'middlegame', 'endgame']:
                col = f'{phase}_complexity'
                if col in analysis_df.columns:
                    phase_complexity.append(analysis_df[col].mean())
            
            if phase_complexity:
                plt.subplot(1, 2, 1)
                plt.bar(['Opening', 'Middlegame', 'Endgame'][:len(phase_complexity)], 
                       phase_complexity, color=['green', 'blue', 'purple'], alpha=0.7)
                plt.ylabel('Average Complexity')
                plt.title('Position Complexity by Phase')
                
            # Complexity vs Performance
            plt.subplot(1, 2, 2)
            if 'opening_complexity' in analysis_df.columns:
                plt.scatter(analysis_df['opening_complexity'], 
                           analysis_df['opening_accuracy'],
                           alpha=0.5, label='Opening')
            if 'middlegame_complexity' in analysis_df.columns:
                plt.scatter(analysis_df['middlegame_complexity'], 
                           analysis_df['middlegame_accuracy'],
                           alpha=0.5, label='Middlegame')
            if 'endgame_complexity' in analysis_df.columns:
                plt.scatter(analysis_df['endgame_complexity'], 
                           analysis_df['endgame_accuracy'],
                           alpha=0.5, label='Endgame')
            
            plt.xlabel('Complexity')
            plt.ylabel('Accuracy')
            plt.title('Complexity vs Performance')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/complexity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
        
        # Generate summary statistics
        summary = {
            'Total Games Analyzed': len(analysis_df),
            'Average Centipawn Loss': f"{analysis_df['average_centipawn_loss'].mean():.1f}",
            'Average Blunders per Game': f"{analysis_df['blunder_count'].mean():.2f}",
            'Average Mistakes per Game': f"{analysis_df['mistake_count'].mean():.2f}",
            'Average Perfect Moves per Game': f"{analysis_df['perfect_move_count'].mean():.2f}",
            'Best Game (Lowest CPL)': analysis_df.loc[analysis_df['average_centipawn_loss'].idxmin(), 'game_id'],
            'Worst Game (Highest CPL)': analysis_df.loc[analysis_df['average_centipawn_loss'].idxmax(), 'game_id']
        }
        
        with open(f'{output_dir}/summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Example usage of StockfishAnalyzer"""
    # Initialize analyzer
    analyzer = StockfishAnalyzer(depth=15, threads=4)
    
    # Load games
    games_df = pd.read_csv('logs/medium-12-600k_iters_pt_vs_stockfish_sweep.csv')
    
    # Sample analysis
    sample_game = games_df.iloc[0]
    analysis = analyzer.analyze_game(sample_game['transcript'], sample_game['game_id'])
    
    print(f"Game {analysis.game_id} analysis:")
    print(f"Average centipawn loss: {analysis.average_centipawn_loss:.1f}")
    print(f"Blunders: {analysis.blunder_count}")
    print(f"Mistakes: {analysis.mistake_count}")
    print(f"Perfect moves: {analysis.perfect_move_count}")
    print(f"Critical moments: {analysis.critical_moments}")
    
    # Batch analysis (small sample for demo)
    # analysis_df = analyzer.batch_analyze_games(games_df, 
    #                                          'stockfish_analysis_results.csv',
    #                                          num_processes=4,
    #                                          sample_size=100)
    
    # Create visualizations
    # analyzer.create_analysis_visualizations(analysis_df)


if __name__ == "__main__":
    main() 