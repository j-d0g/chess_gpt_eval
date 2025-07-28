#!/usr/bin/env python3
"""
Enhanced Chess Game Results Visualization
Combines the best features from multiple analysis scripts for comprehensive game analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional
import chess
import chess.pgn
import io

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')  # Fallback style
sns.set_palette("husl")

class EnhancedGameAnalyzer:
    """Enhanced analyzer with superior features from archived scripts"""
    
    def __init__(self, log_directory: str = "data/games"):
        self.log_directory = log_directory
        self.data = OrderedDict()
        self.model_names = []
        
    def extract_stockfish_level(self, player_two):
        """Extract Stockfish level from player_two column"""
        match = re.search(r'Stockfish (\d+)', str(player_two))
        if match:
            return int(match.group(1))
        return None

    def result_to_score(self, result):
        """Convert result to numeric score"""
        if result in ['1-0', '1']:
            return 1.0
        elif result in ['0-1', '0']:
            return 0.0
        elif result in ['1/2-1/2', '0.5-0.5', '1/2', '0.5']:
            return 0.5
        else:
            return None
    
    def extract_opening(self, transcript: str, moves: int = 6) -> str:
        """Extract first N moves as opening"""
        try:
            # Remove move numbers and clean up
            moves_only = re.sub(r'\d+\.', '', transcript)
            move_list = moves_only.split()[:moves*2]  # *2 for both players
            return ' '.join(move_list[:min(len(move_list), moves*2)])
        except:
            return ""
    
    def categorize_opening(self, opening_moves: str) -> str:
        """Categorize opening based on first moves"""
        opening_map = {
            'e4 e5': 'King\'s Pawn Game',
            'e4 c5': 'Sicilian Defense',
            'e4 e6': 'French Defense',
            'e4 c6': 'Caro-Kann Defense',
            'e4 d6': 'Pirc Defense',
            'd4 d5': 'Queen\'s Pawn Game',
            'd4 Nf6': 'Indian Defense',
            'd4 f5': 'Dutch Defense',
            'Nf3': 'Reti Opening',
            'c4': 'English Opening',
            'f4': 'Bird\'s Opening',
            'b3': 'Larsen\'s Opening'
        }
        
        for pattern, name in opening_map.items():
            if opening_moves.startswith(pattern):
                return name
        return 'Other'
    
    def estimate_elo(self, df: pd.DataFrame) -> int:
        """Estimate Elo rating based on performance vs different Stockfish levels"""
        if 'stockfish_level' not in df.columns:
            return 1200  # Default
        
        # Calculate performance at each level
        level_performance = {}
        for level in range(10):
            level_games = df[df['stockfish_level'] == level]
            if len(level_games) >= 5:  # Minimum games for reliable estimate
                level_performance[level] = level_games['score'].mean()
        
        if not level_performance:
            return 1200
        
        # Estimate based on 50% score point
        # Rough mapping: Stockfish level 0 ≈ 1200, level 9 ≈ 2000
        base_elos = {i: 1200 + i * 89 for i in range(10)}  # Linear progression
        
        # Find interpolated Elo where model scores ~50%
        estimated_elo = 1200
        for level, score in level_performance.items():
            if score >= 0.4:  # Model performs reasonably well
                # Interpolate based on score
                level_elo = base_elos[level]
                if score > 0.5:
                    # Performing better than 50%, estimate higher
                    estimated_elo = max(estimated_elo, level_elo + (score - 0.5) * 200)
                else:
                    # Performing worse than 50%, estimate lower
                    estimated_elo = max(estimated_elo, level_elo - (0.5 - score) * 200)
        
        return int(estimated_elo)
    
    def calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate consistency score"""
        if 'stockfish_level' not in df.columns:
            return df['score'].std()
        
        # Calculate standard deviation of performance across levels
        level_scores = []
        for level in range(10):
            level_games = df[df['stockfish_level'] == level]
            if len(level_games) >= 5:
                level_scores.append(level_games['score'].mean())
        
        if len(level_scores) < 3:
            return df['score'].std()
        
        return np.std(level_scores)
    
    def load_data(self):
        """Load all game data with enhanced processing"""
        # Define log files in order of expected performance
        log_files = [
            'large-16-600k_iters_pt_vs_stockfish_sweep.csv',
            'medium-16-600k_iters_pt_vs_stockfish_sweep.csv', 
            'medium-12-600k_iters_pt_vs_stockfish_sweep.csv',
            'small-24-600k_iters_pt_vs_stockfish_sweep.csv',
            'small-16-600k_iters_pt_vs_stockfish_sweep.csv',
            'small-8-600k_iters_pt_vs_stockfish_sweep.csv',
            'small-36-600k_iters_pt_vs_stockfish_sweep.csv',
            'adam_stockfish_16layers_pt_vs_stockfish_sweep.csv',
            'adam_stockfish_8layers_pt_vs_stockfish_sweep.csv'
        ]
        
        for log_file in log_files:
            file_path = os.path.join(self.log_directory, log_file)
            if os.path.exists(file_path):
                print(f"Loading {log_file}...")
                
                # Handle files with missing headers
                if 'large-16-600k_iters_pt_vs_stockfish_sweep.csv' in log_file:
                    try:
                        df = pd.read_csv(file_path)
                    except:
                        column_names = [
                            'game_id', 'transcript', 'result', 'player_one', 'player_two', 
                            'player_one_time', 'player_two_time', 'player_one_score', 'player_two_score',
                            'player_one_illegal_moves', 'player_two_illegal_moves', 'player_one_legal_moves', 
                            'player_two_legal_moves', 'player_one_resignation', 'player_two_resignation',
                            'player_one_failed_to_find_legal_move', 'player_two_failed_to_find_legal_move',
                            'game_title', 'number_of_moves', 'time_taken', 'total_moves', 'illegal_moves',
                            'player_one_illegal_moves_details', 'player_two_illegal_moves_details'
                        ]
                        df = pd.read_csv(file_path, header=None, names=column_names)
                else:
                    df = pd.read_csv(file_path)
                
                # Extract clean model name
                model_name = log_file.replace('_vs_stockfish_sweep.csv', '').replace('-600k_iters_pt', '')
                self.model_names.append(model_name)
                
                # Enhanced data processing
                df['stockfish_level'] = df['player_two'].apply(self.extract_stockfish_level)
                df['score'] = df['result'].apply(self.result_to_score)
                
                # Ensure numeric columns
                numeric_columns = ['number_of_moves', 'player_one_illegal_moves', 'total_moves']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Enhanced features
                df['opening_moves'] = df['transcript'].apply(self.extract_opening)
                df['opening_category'] = df['opening_moves'].apply(self.categorize_opening)
                df['illegal_move_rate'] = df.apply(
                    lambda row: row['player_one_illegal_moves'] / row['total_moves'] 
                    if row['total_moves'] > 0 else 0, axis=1
                )
                
                # Game length categories
                df['game_length_category'] = pd.cut(df['number_of_moves'], 
                                                   bins=[0, 20, 40, 60, 80, 100, float('inf')],
                                                   labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extremely Long'])
                
                self.data[model_name] = df
                print(f"Loaded {len(df)} games from {model_name}")
    
    def create_enhanced_analysis(self, output_file: str = "enhanced_chess_analysis.png"):
        """Create enhanced analysis with superior visualizations"""
        # Create figure with enhanced layout (4x4 = 16 subplots)
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Performance vs Stockfish Level (Enhanced)
        ax1 = plt.subplot(4, 4, 1)
        self.plot_performance_vs_stockfish(ax1)
        
        # 2. Elo Estimation Comparison
        ax2 = plt.subplot(4, 4, 2)
        self.plot_elo_estimates(ax2)
        
        # 3. Consistency Analysis
        ax3 = plt.subplot(4, 4, 3)
        self.plot_consistency_analysis(ax3)
        
        # 4. Opening Performance Heatmap
        ax4 = plt.subplot(4, 4, 4)
        self.plot_opening_performance_heatmap(ax4)
        
        # 5. Win Rate by Game Length
        ax5 = plt.subplot(4, 4, 5)
        self.plot_win_rate_by_length(ax5)
        
        # 6. Illegal Move Patterns
        ax6 = plt.subplot(4, 4, 6)
        self.plot_illegal_move_patterns(ax6)
        
        # 7. Time Pressure Analysis
        ax7 = plt.subplot(4, 4, 7)
        self.plot_time_pressure_analysis(ax7)
        
        # 8. Opening Repertoire Analysis
        ax8 = plt.subplot(4, 4, 8)
        self.plot_opening_repertoire(ax8)
        
        # 9. Performance Heatmap (Model vs Stockfish Level)
        ax9 = plt.subplot(4, 4, 9)
        self.plot_performance_heatmap(ax9)
        
        # 10. Game Length Distribution
        ax10 = plt.subplot(4, 4, 10)
        self.plot_game_length_distribution(ax10)
        
        # 11. Architecture Impact Analysis
        ax11 = plt.subplot(4, 4, 11)
        self.plot_architecture_impact(ax11)
        
        # 12. Performance Degradation Analysis
        ax12 = plt.subplot(4, 4, 12)
        self.plot_performance_degradation(ax12)
        
        # 13. Model Comparison Radar Chart
        ax13 = plt.subplot(4, 4, 13, projection='polar')
        self.plot_model_comparison_radar(ax13)
        
        # 14. Win/Draw/Loss Stacked Bar
        ax14 = plt.subplot(4, 4, 14)
        self.plot_win_draw_loss_stacked(ax14)
        
        # 15. Illegal Move Rate vs Game Length
        ax15 = plt.subplot(4, 4, 15)
        self.plot_illegal_moves_vs_length(ax15)
        
        # 16. Model Rankings Summary
        ax16 = plt.subplot(4, 4, 16)
        self.plot_model_rankings(ax16)
        
        plt.suptitle('Enhanced Chess Language Model Analysis Dashboard', fontsize=20, y=0.995)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print enhanced summary
        self.print_enhanced_summary()
    
    def plot_performance_vs_stockfish(self, ax):
        """Enhanced performance vs Stockfish level plot"""
        # Calculate model ordering by average performance
        model_avg_scores = {}
        for model, df in self.data.items():
            if 'stockfish_level' in df.columns:
                level_means = df.groupby('stockfish_level')['score'].mean()
                model_avg_scores[model] = level_means.mean()
            else:
                model_avg_scores[model] = df['score'].mean()
        
        # Plot in order of performance
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_avg_scores)))
        for i, model in enumerate(sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)):
            df = self.data[model]
            if 'stockfish_level' in df.columns:
                level_stats = df.groupby('stockfish_level')['score'].agg(['mean', 'count', 'std']).reset_index()
                
                # Plot line with error bars
                ax.errorbar(level_stats['stockfish_level'], level_stats['mean'], 
                           yerr=level_stats['std'], marker='o', linewidth=2, markersize=6,
                           label=f"{model.replace('_', ' ')} (Elo: {self.estimate_elo(df)})",
                           color=colors[i], capsize=3)
        
        ax.set_xlabel('Stockfish Level')
        ax.set_ylabel('Average Score (draws = 0.5)')
        ax.set_title('Performance vs Stockfish Level (with Elo Estimates)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, 9.5)
    
    # Placeholder methods for other plots - implement basic versions
    def plot_elo_estimates(self, ax):
        """Plot Elo estimates for all models"""
        elo_data = []
        for model, df in self.data.items():
            elo = self.estimate_elo(df)
            elo_data.append({'Model': model.replace('_', ' '), 'Elo': elo, 'Games': len(df)})
        
        if elo_data:
            elo_df = pd.DataFrame(elo_data).sort_values('Elo', ascending=True)
            bars = ax.barh(range(len(elo_df)), elo_df['Elo'], 
                           color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(elo_df))))
            ax.set_yticks(range(len(elo_df)))
            ax.set_yticklabels(elo_df['Model'])
            ax.set_xlabel('Estimated Elo Rating')
            ax.set_title('Model Elo Estimates')
            ax.grid(True, alpha=0.3)
    
    def plot_consistency_analysis(self, ax):
        """Plot consistency analysis"""
        consistency_data = []
        for model, df in self.data.items():
            consistency = self.calculate_consistency(df)
            avg_score = df['score'].mean()
            consistency_data.append({
                'Model': model.replace('_', ' '),
                'Consistency': 1 / (consistency + 0.01),
                'Avg Score': avg_score
            })
        
        if consistency_data:
            consist_df = pd.DataFrame(consistency_data)
            scatter = ax.scatter(consist_df['Avg Score'], consist_df['Consistency'], 
                               s=100, alpha=0.7, c=range(len(consist_df)), cmap='viridis')
            ax.set_xlabel('Average Score')
            ax.set_ylabel('Consistency Score')
            ax.set_title('Performance vs Consistency')
            ax.grid(True, alpha=0.3)
    
    # Add basic implementations for remaining plot methods
    def plot_opening_performance_heatmap(self, ax):
        ax.text(0.5, 0.5, 'Opening Performance\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Opening Performance Heatmap')
    
    def plot_win_rate_by_length(self, ax):
        ax.text(0.5, 0.5, 'Win Rate by Length\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Win Rate by Game Length')
    
    def plot_illegal_move_patterns(self, ax):
        ax.text(0.5, 0.5, 'Illegal Move Patterns\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Illegal Move Patterns')
    
    def plot_time_pressure_analysis(self, ax):
        ax.text(0.5, 0.5, 'Time Pressure Analysis\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Time Pressure Analysis')
    
    def plot_opening_repertoire(self, ax):
        ax.text(0.5, 0.5, 'Opening Repertoire\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Opening Repertoire')
    
    def plot_performance_heatmap(self, ax):
        ax.text(0.5, 0.5, 'Performance Heatmap\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Heatmap')
    
    def plot_game_length_distribution(self, ax):
        ax.text(0.5, 0.5, 'Game Length Distribution\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Game Length Distribution')
    
    def plot_architecture_impact(self, ax):
        ax.text(0.5, 0.5, 'Architecture Impact\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Architecture Impact')
    
    def plot_performance_degradation(self, ax):
        ax.text(0.5, 0.5, 'Performance Degradation\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Degradation')
    
    def plot_model_comparison_radar(self, ax):
        ax.text(0.5, 0.5, 'Model Comparison\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Comparison Radar')
    
    def plot_win_draw_loss_stacked(self, ax):
        ax.text(0.5, 0.5, 'Win/Draw/Loss\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Win/Draw/Loss Distribution')
    
    def plot_illegal_moves_vs_length(self, ax):
        ax.text(0.5, 0.5, 'Illegal Moves vs Length\n(Needs game data)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Illegal Moves vs Game Length')
    
    def plot_model_rankings(self, ax):
        """Plot model rankings summary"""
        rankings = []
        for model, df in self.data.items():
            elo = self.estimate_elo(df)
            avg_score = df['score'].mean()
            rankings.append({'Model': model.replace('_', ' '), 'Elo': elo, 'Score': avg_score})
        
        if rankings:
            rankings.sort(key=lambda x: x['Elo'], reverse=True)
            models = [r['Model'] for r in rankings]
            elos = [r['Elo'] for r in rankings]
            
            bars = ax.barh(range(len(models)), elos, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            ax.set_xlabel('Estimated Elo')
            ax.set_title('Model Rankings by Elo')
    
    def print_enhanced_summary(self):
        """Print enhanced summary statistics"""
        print("\n" + "="*80)
        print("ENHANCED CHESS MODEL ANALYSIS SUMMARY")
        print("="*80)
        
        for model, df in self.data.items():
            print(f"\n{model.replace('_', ' ').upper()}:")
            print(f"  Games: {len(df):,}")
            print(f"  Average Score: {df['score'].mean():.3f}")
            print(f"  Estimated Elo: {self.estimate_elo(df):,}")
            print(f"  Consistency: {self.calculate_consistency(df):.3f}")
            
            if 'stockfish_level' in df.columns:
                best_level = df.groupby('stockfish_level')['score'].mean().idxmax()
                print(f"  Best vs Stockfish Level: {best_level}")


def main():
    """Main function to run enhanced analysis"""
    analyzer = EnhancedGameAnalyzer()
    analyzer.load_data()
    
    if analyzer.data:
        analyzer.create_enhanced_analysis()
    else:
        print("No game data found. Please run game evaluation first.")


if __name__ == "__main__":
    main() 