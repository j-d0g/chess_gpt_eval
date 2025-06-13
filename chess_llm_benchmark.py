#!/usr/bin/env python3
"""
Comprehensive Chess LLM Benchmark Suite
A standardized benchmark for evaluating chess-playing language models
"""

import pandas as pd
import numpy as np
import chess
import chess.pgn
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re


@dataclass
class BenchmarkTask:
    """Individual benchmark task"""
    task_id: str
    category: str
    difficulty: str  # easy, medium, hard, expert
    position_fen: str
    correct_moves: List[str]
    description: str
    evaluation_criteria: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_name: str
    overall_score: float
    category_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    detailed_results: List[Dict]
    elo_estimate: int
    strengths: List[str]
    weaknesses: List[str]


class ChessLLMBenchmark:
    """Comprehensive benchmark suite for chess language models"""
    
    def __init__(self):
        self.tasks = []
        self.results = {}
        self._initialize_benchmark_tasks()
    
    def _initialize_benchmark_tasks(self):
        """Initialize comprehensive benchmark tasks"""
        
        # 1. Tactical Puzzles
        self._add_tactical_puzzles()
        
        # 2. Positional Understanding
        self._add_positional_tasks()
        
        # 3. Endgame Knowledge
        self._add_endgame_tasks()
        
        # 4. Opening Theory
        self._add_opening_tasks()
        
        # 5. Game Continuation
        self._add_game_continuation_tasks()
        
        # 6. Illegal Move Detection
        self._add_illegal_move_tasks()
        
        # 7. Strategic Planning
        self._add_strategic_tasks()
        
        # 8. Time Pressure Simulation
        self._add_time_pressure_tasks()
        
        # 9. Complex Positions
        self._add_complex_position_tasks()
        
        # 10. Historical Games
        self._add_historical_game_tasks()
    
    def _add_tactical_puzzles(self):
        """Add tactical puzzle tasks"""
        tactical_puzzles = [
            {
                'task_id': 'tactic_001',
                'category': 'tactics',
                'difficulty': 'easy',
                'position_fen': 'r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
                'correct_moves': ['Bxc6'],
                'description': 'Simple piece capture',
                'evaluation_criteria': {
                    'move_found': 1.0,
                    'time_limit': 30,
                    'explanation_quality': 0.0
                }
            },
            {
                'task_id': 'tactic_002',
                'category': 'tactics',
                'difficulty': 'medium',
                'position_fen': '2kr1b1r/pp1npppp/2p1bn2/q7/3PN3/1PN1B3/P1P1BPPP/R2QK2R w KQ - 0 1',
                'correct_moves': ['Nxf6+', 'Bxf6'],
                'description': 'Fork pattern recognition',
                'evaluation_criteria': {
                    'move_found': 0.7,
                    'sequence_completion': 0.3,
                    'time_limit': 60
                }
            },
            {
                'task_id': 'tactic_003',
                'category': 'tactics',
                'difficulty': 'hard',
                'position_fen': 'r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 1',
                'correct_moves': ['Nxc6', 'bxc6', 'Bxc5'],
                'description': 'Complex tactical sequence',
                'evaluation_criteria': {
                    'move_found': 0.5,
                    'sequence_completion': 0.4,
                    'calculation_depth': 0.1,
                    'time_limit': 120
                }
            },
            {
                'task_id': 'tactic_004',
                'category': 'tactics',
                'difficulty': 'expert',
                'position_fen': '3r1rk1/1p3pnp/p1p2qp1/2P1p3/1P2P1b1/P2P1NBP/4QPP1/2R2RK1 b - - 0 1',
                'correct_moves': ['Rxd3', 'Qxd3', 'Qxf3'],
                'description': 'Sacrifice combination',
                'evaluation_criteria': {
                    'move_found': 0.4,
                    'sequence_completion': 0.4,
                    'evaluation_accuracy': 0.2,
                    'time_limit': 180
                }
            }
        ]
        
        for puzzle in tactical_puzzles:
            self.tasks.append(BenchmarkTask(**puzzle))
    
    def _add_positional_tasks(self):
        """Add positional understanding tasks"""
        positional_tasks = [
            {
                'task_id': 'position_001',
                'category': 'positional',
                'difficulty': 'medium',
                'position_fen': 'r1bqk2r/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 1',
                'correct_moves': ['cxd5', 'e4'],
                'description': 'Central pawn break',
                'evaluation_criteria': {
                    'strategic_understanding': 0.6,
                    'move_found': 0.4,
                    'plan_quality': 0.0
                }
            },
            {
                'task_id': 'position_002',
                'category': 'positional',
                'difficulty': 'hard',
                'position_fen': 'r2q1rk1/1b1nbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2QB1PP/R4RK1 w - - 0 1',
                'correct_moves': ['Nd5', 'f4'],
                'description': 'Knight outpost exploitation',
                'evaluation_criteria': {
                    'strategic_understanding': 0.5,
                    'long_term_planning': 0.3,
                    'move_found': 0.2
                }
            }
        ]
        
        for task in positional_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_endgame_tasks(self):
        """Add endgame knowledge tasks"""
        endgame_tasks = [
            {
                'task_id': 'endgame_001',
                'category': 'endgame',
                'difficulty': 'easy',
                'position_fen': '8/8/8/8/4K3/8/5k2/7R w - - 0 1',
                'correct_moves': ['Rh2+', 'Rf1'],
                'description': 'Basic rook endgame - cutting off king',
                'evaluation_criteria': {
                    'technique_knowledge': 0.7,
                    'move_found': 0.3
                }
            },
            {
                'task_id': 'endgame_002',
                'category': 'endgame',
                'difficulty': 'medium',
                'position_fen': '8/5kpp/8/5P2/5K2/8/6PP/8 w - - 0 1',
                'correct_moves': ['g4', 'h4'],
                'description': 'King and pawn endgame - breakthrough',
                'evaluation_criteria': {
                    'calculation_accuracy': 0.5,
                    'endgame_knowledge': 0.5
                }
            },
            {
                'task_id': 'endgame_003',
                'category': 'endgame',
                'difficulty': 'expert',
                'position_fen': '8/8/1p6/p1p5/P1P5/1P6/8/5K1k w - - 0 1',
                'correct_moves': ['Kf2'],
                'description': 'Complex pawn endgame - triangulation',
                'evaluation_criteria': {
                    'endgame_knowledge': 0.6,
                    'calculation_depth': 0.4
                }
            }
        ]
        
        for task in endgame_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_opening_tasks(self):
        """Add opening theory tasks"""
        opening_tasks = [
            {
                'task_id': 'opening_001',
                'category': 'opening',
                'difficulty': 'easy',
                'position_fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                'correct_moves': ['e5', 'c5', 'e6'],
                'description': 'Basic opening response to 1.e4',
                'evaluation_criteria': {
                    'opening_knowledge': 0.8,
                    'move_found': 0.2
                }
            },
            {
                'task_id': 'opening_002',
                'category': 'opening',
                'difficulty': 'medium',
                'position_fen': 'rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2',
                'correct_moves': ['Nxe4', 'e6', 'd5'],
                'description': 'Alekhine Defense position',
                'evaluation_criteria': {
                    'opening_knowledge': 0.6,
                    'theoretical_accuracy': 0.4
                }
            }
        ]
        
        for task in opening_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_game_continuation_tasks(self):
        """Add game continuation tasks"""
        continuation_tasks = [
            {
                'task_id': 'continuation_001',
                'category': 'game_continuation',
                'difficulty': 'medium',
                'position_fen': 'r1bqkb1r/pp2pppp/2n2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 1',
                'correct_moves': ['cxd5', 'Bg5', 'e3'],
                'description': 'Continue from standard position',
                'evaluation_criteria': {
                    'move_quality': 0.5,
                    'consistency': 0.3,
                    'plan_coherence': 0.2
                }
            }
        ]
        
        for task in continuation_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_illegal_move_tasks(self):
        """Add illegal move detection tasks"""
        illegal_tasks = [
            {
                'task_id': 'illegal_001',
                'category': 'illegal_detection',
                'difficulty': 'easy',
                'position_fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'correct_moves': ['NOT Nf6', 'NOT Be5'],  # These are illegal
                'description': 'Detect illegal moves from starting position',
                'evaluation_criteria': {
                    'illegal_detection': 1.0
                }
            }
        ]
        
        for task in illegal_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_strategic_tasks(self):
        """Add strategic planning tasks"""
        strategic_tasks = [
            {
                'task_id': 'strategy_001',
                'category': 'strategy',
                'difficulty': 'hard',
                'position_fen': 'r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1',
                'correct_moves': ['b4', 'Rc1', 'a3'],
                'description': 'Minority attack planning',
                'evaluation_criteria': {
                    'plan_quality': 0.6,
                    'strategic_understanding': 0.4
                }
            }
        ]
        
        for task in strategic_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_time_pressure_tasks(self):
        """Add time pressure simulation tasks"""
        time_tasks = [
            {
                'task_id': 'time_001',
                'category': 'time_pressure',
                'difficulty': 'medium',
                'position_fen': 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
                'correct_moves': ['Bxf6', 'Nxf7', 'O-O-O'],
                'description': 'Quick decision under time pressure',
                'evaluation_criteria': {
                    'move_found': 0.7,
                    'time_taken': 0.3,
                    'time_limit': 10  # seconds
                }
            }
        ]
        
        for task in time_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_complex_position_tasks(self):
        """Add complex position evaluation tasks"""
        complex_tasks = [
            {
                'task_id': 'complex_001',
                'category': 'complex_position',
                'difficulty': 'expert',
                'position_fen': 'r2q1r1k/1b1nbpp1/pp1ppn1p/8/2PNP2B/1PN1BP2/P2Q2PP/2R2RK1 w - - 0 1',
                'correct_moves': ['Nd5', 'f4', 'Bg3'],
                'description': 'Complex middlegame with multiple plans',
                'evaluation_criteria': {
                    'evaluation_accuracy': 0.3,
                    'plan_quality': 0.4,
                    'move_found': 0.3
                }
            }
        ]
        
        for task in complex_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def _add_historical_game_tasks(self):
        """Add historical game position tasks"""
        historical_tasks = [
            {
                'task_id': 'historical_001',
                'category': 'historical',
                'difficulty': 'hard',
                'position_fen': 'r1bqkb1r/pp2pppp/2n2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 1',
                'correct_moves': ['cxd5'],  # From Kasparov-Karpov
                'description': 'Position from famous historical game',
                'evaluation_criteria': {
                    'historical_knowledge': 0.3,
                    'move_quality': 0.7
                }
            }
        ]
        
        for task in historical_tasks:
            self.tasks.append(BenchmarkTask(**task))
    
    def evaluate_model(self, model_name: str, model_responses: Dict[str, Dict]) -> ModelPerformance:
        """
        Evaluate a model's performance on the benchmark
        
        Args:
            model_name: Name of the model
            model_responses: Dictionary mapping task_id to response dict with:
                - move: The move suggested by the model
                - time_taken: Time taken to respond
                - explanation: Optional explanation
                
        Returns:
            ModelPerformance object with detailed results
        """
        detailed_results = []
        category_scores = defaultdict(list)
        difficulty_scores = defaultdict(list)
        
        for task in self.tasks:
            if task.task_id not in model_responses:
                continue
            
            response = model_responses[task.task_id]
            score = self._evaluate_task(task, response)
            
            result = {
                'task_id': task.task_id,
                'category': task.category,
                'difficulty': task.difficulty,
                'score': score,
                'move': response.get('move'),
                'correct_moves': task.correct_moves,
                'time_taken': response.get('time_taken', 0)
            }
            
            detailed_results.append(result)
            category_scores[task.category].append(score)
            difficulty_scores[task.difficulty].append(score)
        
        # Calculate aggregate scores
        overall_score = np.mean([r['score'] for r in detailed_results])
        
        category_avg = {cat: np.mean(scores) for cat, scores in category_scores.items()}
        difficulty_avg = {diff: np.mean(scores) for diff, scores in difficulty_scores.items()}
        
        # Estimate Elo based on performance
        elo_estimate = self._estimate_elo(difficulty_avg)
        
        # Identify strengths and weaknesses
        strengths = [cat for cat, score in category_avg.items() if score > 0.7]
        weaknesses = [cat for cat, score in category_avg.items() if score < 0.4]
        
        return ModelPerformance(
            model_name=model_name,
            overall_score=overall_score,
            category_scores=category_avg,
            difficulty_scores=difficulty_avg,
            detailed_results=detailed_results,
            elo_estimate=elo_estimate,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _evaluate_task(self, task: BenchmarkTask, response: Dict) -> float:
        """Evaluate a single task response"""
        score = 0.0
        criteria = task.evaluation_criteria
        
        # Check if move is correct
        if 'move_found' in criteria:
            if response.get('move') in task.correct_moves:
                score += criteria['move_found']
        
        # Check time constraints
        if 'time_limit' in criteria and 'time_taken' in criteria:
            time_taken = response.get('time_taken', float('inf'))
            if time_taken <= criteria['time_limit']:
                score += criteria['time_taken']
        
        # Additional criteria can be added here
        # For now, we'll use simplified scoring
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _estimate_elo(self, difficulty_scores: Dict[str, float]) -> int:
        """Estimate Elo rating based on difficulty scores"""
        # Simple mapping - can be refined with actual data
        elo_map = {
            'easy': (1200, 1400),
            'medium': (1400, 1700),
            'hard': (1700, 2000),
            'expert': (2000, 2400)
        }
        
        estimated_elo = 1200  # Base rating
        
        for difficulty, score in difficulty_scores.items():
            if difficulty in elo_map:
                min_elo, max_elo = elo_map[difficulty]
                # Linear interpolation based on score
                difficulty_elo = min_elo + score * (max_elo - min_elo)
                estimated_elo = max(estimated_elo, difficulty_elo)
        
        return int(estimated_elo)
    
    def create_benchmark_report(self, performances: List[ModelPerformance], 
                              output_dir: str = "benchmark_results"):
        """Create comprehensive benchmark report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Overall Comparison
        self._create_overall_comparison(performances, output_dir, timestamp)
        
        # 2. Category Analysis
        self._create_category_analysis(performances, output_dir, timestamp)
        
        # 3. Difficulty Progression
        self._create_difficulty_analysis(performances, output_dir, timestamp)
        
        # 4. Detailed Report
        self._create_detailed_report(performances, output_dir, timestamp)
        
        # 5. Leaderboard
        self._create_leaderboard(performances, output_dir, timestamp)
        
        print(f"Benchmark report created in {output_dir}/")
    
    def _create_overall_comparison(self, performances: List[ModelPerformance], 
                                 output_dir: str, timestamp: str):
        """Create overall comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sort by overall score
        performances.sort(key=lambda x: x.overall_score, reverse=True)
        
        # 1. Overall scores
        ax = axes[0, 0]
        models = [p.model_name for p in performances]
        scores = [p.overall_score for p in performances]
        
        bars = ax.bar(models, scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax.set_ylabel('Overall Score')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Elo estimates
        ax = axes[0, 1]
        elos = [p.elo_estimate for p in performances]
        
        ax.barh(models, elos, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
        ax.set_xlabel('Estimated Elo Rating')
        ax.set_title('Elo Rating Estimates')
        
        # 3. Category radar chart
        ax = axes[1, 0]
        categories = list(performances[0].category_scores.keys())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 3, projection='polar')
        
        for i, perf in enumerate(performances[:5]):  # Top 5 models
            values = [perf.category_scores.get(cat, 0) for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=perf.model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Performance by Category')
        
        # 4. Strengths and weaknesses
        ax = axes[1, 1]
        ax.axis('off')
        
        y_pos = 0.9
        for perf in performances[:5]:
            ax.text(0.1, y_pos, f"{perf.model_name}:", fontweight='bold')
            ax.text(0.1, y_pos - 0.05, f"Strengths: {', '.join(perf.strengths) or 'None'}", 
                   color='green', fontsize=9)
            ax.text(0.1, y_pos - 0.1, f"Weaknesses: {', '.join(perf.weaknesses) or 'None'}", 
                   color='red', fontsize=9)
            y_pos -= 0.2
        
        ax.set_title('Model Strengths and Weaknesses')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_category_analysis(self, performances: List[ModelPerformance], 
                                output_dir: str, timestamp: str):
        """Create category-specific analysis"""
        categories = list(performances[0].category_scores.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(categories))
        width = 0.8 / len(performances)
        
        for i, perf in enumerate(performances):
            scores = [perf.category_scores.get(cat, 0) for cat in categories]
            offset = (i - len(performances)/2) * width + width/2
            ax.bar(x + offset, scores, width, label=perf.model_name, alpha=0.8)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Average Score')
        ax.set_title('Performance by Task Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_difficulty_analysis(self, performances: List[ModelPerformance], 
                                  output_dir: str, timestamp: str):
        """Create difficulty progression analysis"""
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Line plot showing progression
        for perf in performances:
            scores = [perf.difficulty_scores.get(diff, 0) for diff in difficulties]
            ax1.plot(difficulties, scores, marker='o', linewidth=2, 
                    label=perf.model_name, markersize=8)
        
        ax1.set_xlabel('Difficulty Level')
        ax1.set_ylabel('Average Score')
        ax1.set_title('Performance vs Task Difficulty')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Heatmap
        data = []
        model_names = []
        for perf in performances:
            model_names.append(perf.model_name)
            scores = [perf.difficulty_scores.get(diff, 0) for diff in difficulties]
            data.append(scores)
        
        sns.heatmap(data, xticklabels=difficulties, yticklabels=model_names,
                   annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2,
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        ax2.set_title('Difficulty Score Heatmap')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/difficulty_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_report(self, performances: List[ModelPerformance], 
                              output_dir: str, timestamp: str):
        """Create detailed text report"""
        report = []
        report.append("# Chess LLM Benchmark Report")
        report.append(f"Generated: {timestamp}\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"- Total Tasks: {len(self.tasks)}")
        report.append(f"- Categories: {', '.join(set(t.category for t in self.tasks))}")
        report.append(f"- Difficulty Levels: {', '.join(set(t.difficulty for t in self.tasks))}")
        report.append(f"- Models Evaluated: {len(performances)}\n")
        
        # Leaderboard
        report.append("## Leaderboard\n")
        report.append("| Rank | Model | Overall Score | Estimated Elo |")
        report.append("|------|-------|---------------|---------------|")
        
        for i, perf in enumerate(performances):
            report.append(f"| {i+1} | {perf.model_name} | {perf.overall_score:.3f} | {perf.elo_estimate} |")
        
        # Detailed results per model
        report.append("\n## Detailed Results\n")
        
        for perf in performances:
            report.append(f"### {perf.model_name}")
            report.append(f"- Overall Score: {perf.overall_score:.3f}")
            report.append(f"- Estimated Elo: {perf.elo_estimate}")
            report.append(f"- Strengths: {', '.join(perf.strengths) or 'None identified'}")
            report.append(f"- Weaknesses: {', '.join(perf.weaknesses) or 'None identified'}")
            
            # Category breakdown
            report.append("\n#### Category Scores:")
            for cat, score in sorted(perf.category_scores.items()):
                report.append(f"- {cat}: {score:.3f}")
            
            # Difficulty breakdown
            report.append("\n#### Difficulty Scores:")
            for diff, score in perf.difficulty_scores.items():
                report.append(f"- {diff}: {score:.3f}")
            
            report.append("\n---\n")
        
        # Task analysis
        report.append("## Task Analysis\n")
        
        # Find hardest and easiest tasks
        task_performances = defaultdict(list)
        for perf in performances:
            for result in perf.detailed_results:
                task_performances[result['task_id']].append(result['score'])
        
        avg_task_scores = {task_id: np.mean(scores) 
                          for task_id, scores in task_performances.items()}
        
        sorted_tasks = sorted(avg_task_scores.items(), key=lambda x: x[1])
        
        report.append("### Hardest Tasks")
        for task_id, avg_score in sorted_tasks[:5]:
            task = next(t for t in self.tasks if t.task_id == task_id)
            report.append(f"- {task_id} ({task.category}, {task.difficulty}): "
                         f"Avg Score = {avg_score:.3f}")
            report.append(f"  Description: {task.description}")
        
        report.append("\n### Easiest Tasks")
        for task_id, avg_score in sorted_tasks[-5:]:
            task = next(t for t in self.tasks if t.task_id == task_id)
            report.append(f"- {task_id} ({task.category}, {task.difficulty}): "
                         f"Avg Score = {avg_score:.3f}")
            report.append(f"  Description: {task.description}")
        
        # Save report
        with open(f"{output_dir}/detailed_report_{timestamp}.md", 'w') as f:
            f.write('\n'.join(report))
        
        # Also save raw data as JSON
        raw_data = {
            'timestamp': timestamp,
            'tasks': [asdict(t) for t in self.tasks],
            'performances': [asdict(p) for p in performances]
        }
        
        with open(f"{output_dir}/raw_results_{timestamp}.json", 'w') as f:
            json.dump(raw_data, f, indent=2)
    
    def _create_leaderboard(self, performances: List[ModelPerformance], 
                          output_dir: str, timestamp: str):
        """Create visual leaderboard"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by overall score
        performances.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Create horizontal bar chart
        models = [p.model_name for p in performances]
        scores = [p.overall_score for p in performances]
        elos = [p.elo_estimate for p in performances]
        
        y_pos = np.arange(len(models))
        
        bars = ax.barh(y_pos, scores, color=plt.cm.RdYlGn(scores))
        
        # Add Elo ratings as text
        for i, (bar, elo) in enumerate(zip(bars, elos)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'Elo: {elo}', ha='left', va='center')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Overall Score')
        ax.set_title('Chess LLM Benchmark Leaderboard')
        ax.set_xlim(0, 1.1)
        
        # Add rank numbers
        for i, bar in enumerate(bars):
            ax.text(0.01, bar.get_y() + bar.get_height()/2,
                   f'#{i+1}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaderboard_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_benchmark_tasks(self, output_file: str = "benchmark_tasks.json"):
        """Export benchmark tasks for external use"""
        tasks_data = []
        for task in self.tasks:
            task_dict = asdict(task)
            # Add board visualization
            board = chess.Board(task.position_fen)
            task_dict['board_ascii'] = str(board)
            tasks_data.append(task_dict)
        
        with open(output_file, 'w') as f:
            json.dump(tasks_data, f, indent=2)
        
        print(f"Exported {len(tasks_data)} benchmark tasks to {output_file}")
    
    def run_benchmark_on_csv(self, csv_file: str, model_name: str) -> ModelPerformance:
        """
        Run benchmark on a model using game data from CSV
        This is a simplified version - in practice, you'd need to:
        1. Set up positions from benchmark tasks
        2. Get model to play from those positions
        3. Evaluate the moves
        """
        # Placeholder implementation
        # In reality, this would interface with the chess playing system
        print(f"Running benchmark on {model_name} using data from {csv_file}")
        
        # For now, return mock results
        mock_responses = {}
        for task in self.tasks:
            mock_responses[task.task_id] = {
                'move': task.correct_moves[0] if task.correct_moves else 'e4',
                'time_taken': np.random.uniform(1, 10)
            }
        
        return self.evaluate_model(model_name, mock_responses)


def main():
    """Example usage of the benchmark suite"""
    # Initialize benchmark
    benchmark = ChessLLMBenchmark()
    
    # Export tasks for reference
    benchmark.export_benchmark_tasks()
    
    # Example: Run benchmark on multiple models
    # In practice, you would get actual model responses
    performances = []
    
    # Mock data for demonstration
    model_names = ['small-8-layers', 'medium-12-layers', 'large-16-layers']
    
    for model_name in model_names:
        # Generate mock responses (in practice, get from actual model)
        mock_responses = {}
        for task in benchmark.tasks:
            # Simulate varying performance
            if 'small' in model_name:
                success_rate = 0.6
            elif 'medium' in model_name:
                success_rate = 0.75
            else:
                success_rate = 0.85
            
            # Mock response
            is_correct = np.random.random() < success_rate
            mock_responses[task.task_id] = {
                'move': task.correct_moves[0] if (task.correct_moves and is_correct) else 'e4',
                'time_taken': np.random.uniform(1, 30)
            }
        
        performance = benchmark.evaluate_model(model_name, mock_responses)
        performances.append(performance)
    
    # Create comprehensive report
    benchmark.create_benchmark_report(performances)
    
    print("\nBenchmark complete! Check benchmark_results/ for detailed reports.")


if __name__ == "__main__":
    main() 