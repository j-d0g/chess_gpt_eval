#!/usr/bin/env python3
"""
Advanced Chess Analysis Dashboard for NanoGPT Models
Comprehensive benchmarking and visualization suite for chess language models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

# Set style for better-looking plots
try:
    # Try different seaborn styles in order of preference
    for style in ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'seaborn', 'ggplot', 'default']:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    sns.set_palette("husl")
except ImportError:
    # If seaborn is not available, use matplotlib defaults
    plt.style.use('ggplot')

class ChessAnalysisDashboard:
    """Comprehensive analysis dashboard for chess GPT models"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        self.data = {}
        self.model_names = []
        self.stockfish_annotations = {}
        
    def load_data(self, specific_files: Optional[List[str]] = None):
        """Load chess game data from CSV files"""
        if specific_files:
            log_files = specific_files
        else:
            log_files = [
                'logs/small-8-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/small-16-600k_iters_pt_vs_stockfish_sweep.csv', 
                'logs/small-24-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/small-36-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/medium-12-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/medium-16-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/large-16-600k_iters_pt_vs_stockfish_sweep.csv',
                'logs/adam_stockfish_8layers_pt_vs_stockfish_sweep.csv',
                'logs/adam_stockfish_16layers_pt_vs_stockfish_sweep.csv'
            ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"Loading {log_file}...")
                
                # Special handling for large-16 file which might be missing header
                if 'large-16-600k_iters_pt_vs_stockfish_sweep.csv' in log_file:
                    try:
                        df = pd.read_csv(log_file)
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
                        df = pd.read_csv(log_file, header=None, names=column_names)
                else:
                    df = pd.read_csv(log_file)
                
                model_name = self._extract_model_name(log_file)
                self.model_names.append(model_name)
                
                # Process and enrich data
                df = self._process_dataframe(df)
                
                self.data[model_name] = df
                print(f"Loaded {len(df)} games from {model_name}")
    
    def _extract_model_name(self, filepath: str) -> str:
        """Extract clean model name from filepath"""
        filename = filepath.split('/')[-1]
        model_name = filename.replace('_vs_stockfish_sweep.csv', '').replace('-600k_iters_pt', '')
        return model_name
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enrich dataframe with additional features"""
        # Extract Stockfish level
        df['stockfish_level'] = df['player_two'].apply(self._extract_stockfish_level)
        
        # Convert result to numeric score
        df['score'] = df['result'].apply(self._result_to_score)
        
        # Ensure numeric columns
        numeric_columns = ['number_of_moves', 'player_one_illegal_moves', 
                          'player_two_illegal_moves', 'total_moves']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract opening information
        df['opening_moves'] = df['transcript'].apply(self._extract_opening)
        df['opening_category'] = df['opening_moves'].apply(self._categorize_opening)
        
        # Game phase analysis
        df['game_length_category'] = pd.cut(df['number_of_moves'], 
                                           bins=[0, 20, 40, 60, 80, 100, float('inf')],
                                           labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extremely Long'])
        
        # Illegal move rate
        df['illegal_move_rate'] = df.apply(
            lambda row: row['player_one_illegal_moves'] / row['total_moves'] 
            if row['total_moves'] > 0 else 0, axis=1
        )
        
        # Parse illegal move details if available
        if 'player_one_illegal_moves_details' in df.columns:
            df['parsed_illegal_moves'] = df['player_one_illegal_moves_details'].apply(self._parse_illegal_moves)
        
        return df
    
    def _extract_stockfish_level(self, player_two: str) -> Optional[int]:
        """Extract Stockfish level from player_two column"""
        match = re.search(r'Stockfish (\d+)', str(player_two))
        if match:
            return int(match.group(1))
        return None
    
    def _result_to_score(self, result: str) -> Optional[float]:
        """Convert chess result to numeric score"""
        if result in ['1-0', '1']:
            return 1.0
        elif result in ['0-1', '0']:
            return 0.0
        elif result in ['1/2-1/2', '0.5-0.5', '1/2', '0.5']:
            return 0.5
        return None
    
    def _extract_opening(self, transcript: str, moves: int = 10) -> str:
        """Extract first N moves as opening"""
        try:
            # Remove move numbers and clean up
            moves_only = re.sub(r'\d+\.', '', transcript)
            move_list = moves_only.split()[:moves*2]  # *2 for both players
            return ' '.join(move_list[:min(len(move_list), moves*2)])
        except:
            return ""
    
    def _categorize_opening(self, opening_moves: str) -> str:
        """Categorize opening based on first moves"""
        opening_map = {
            'e4 e5': 'King\'s Pawn',
            'e4 c5': 'Sicilian Defense',
            'e4 e6': 'French Defense',
            'e4 c6': 'Caro-Kann Defense',
            'd4 d5': 'Queen\'s Pawn',
            'd4 Nf6': 'Indian Defense',
            'Nf3': 'Reti Opening',
            'c4': 'English Opening'
        }
        
        for pattern, name in opening_map.items():
            if opening_moves.startswith(pattern):
                return name
        return 'Other'
    
    def _parse_illegal_moves(self, illegal_moves_str: str) -> List[Dict]:
        """Parse illegal moves details from string representation"""
        try:
            if pd.isna(illegal_moves_str) or illegal_moves_str == '[]':
                return []
            # Handle string representation of list
            if isinstance(illegal_moves_str, str):
                return eval(illegal_moves_str)
            return illegal_moves_str
        except:
            return []
    
    def create_comprehensive_dashboard(self, output_dir: str = "dashboard_output"):
        """Create comprehensive dashboard with all visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance Overview Dashboard
        self._create_performance_overview(output_dir, timestamp)
        
        # 2. Detailed Game Analysis
        self._create_game_analysis(output_dir, timestamp)
        
        # 3. Error Pattern Analysis
        self._create_error_analysis(output_dir, timestamp)
        
        # 4. Opening Analysis
        self._create_opening_analysis(output_dir, timestamp)
        
        # 5. Interactive Plotly Dashboard
        self._create_interactive_dashboard(output_dir, timestamp)
        
        # 6. Model Comparison Report
        self._create_comparison_report(output_dir, timestamp)
        
        print(f"\nDashboard created successfully in {output_dir}/")
        print(f"Timestamp: {timestamp}")
    
    def _create_performance_overview(self, output_dir: str, timestamp: str):
        """Create performance overview visualizations"""
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Main Performance vs Stockfish Level
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_vs_stockfish(ax1)
        
        # 2. Elo Estimation
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_elo_estimation(ax2)
        
        # 3. Win Rate Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_win_rate_heatmap(ax3)
        
        # 4. Game Length Analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_game_length_distribution(ax4)
        
        # 5. Illegal Move Patterns
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_illegal_move_patterns(ax5)
        
        # 6. Performance by Game Phase
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_performance_by_phase(ax6)
        
        # 7. Model Architecture Comparison
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_architecture_comparison(ax7)
        
        plt.suptitle('Chess GPT Performance Overview Dashboard', fontsize=20, y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_overview_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_vs_stockfish(self, ax):
        """Plot performance against different Stockfish levels"""
        # Calculate average scores per level for each model
        model_performances = {}
        
        for model, df in self.data.items():
            if 'stockfish_level' in df.columns:
                level_stats = df.groupby('stockfish_level').agg({
                    'score': ['mean', 'std', 'count']
                }).reset_index()
                
                # Only plot if we have sufficient data
                level_stats = level_stats[level_stats[('score', 'count')] >= 20]
                
                if len(level_stats) > 0:
                    model_performances[model] = level_stats
        
        # Sort models by average performance
        model_avg_scores = {}
        for model, stats in model_performances.items():
            model_avg_scores[model] = stats[('score', 'mean')].mean()
        
        sorted_models = sorted(model_avg_scores.keys(), 
                             key=lambda x: model_avg_scores[x], reverse=True)
        
        # Plot with error bars
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
        
        for i, model in enumerate(sorted_models):
            stats = model_performances[model]
            ax.errorbar(stats['stockfish_level'], 
                       stats[('score', 'mean')],
                       yerr=stats[('score', 'std')],
                       marker='o', linewidth=2, markersize=8,
                       label=f"{model} (avg: {model_avg_scores[model]:.3f})",
                       color=colors[i], capsize=5, capthick=2)
        
        ax.set_xlabel('Stockfish Level', fontsize=12)
        ax.set_ylabel('Average Score (Win=1, Draw=0.5, Loss=0)', fontsize=12)
        ax.set_title('Model Performance vs Stockfish Levels', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(10))
    
    def _plot_elo_estimation(self, ax):
        """Estimate and plot Elo ratings for models"""
        # Simple Elo estimation based on win rates against known Stockfish levels
        # Stockfish level 0 ≈ 1300 Elo (based on the blog post)
        stockfish_elo_map = {
            0: 1300, 1: 1400, 2: 1500, 3: 1600, 4: 1700,
            5: 1800, 6: 1900, 7: 2000, 8: 2100, 9: 2200
        }
        
        model_elos = []
        
        for model, df in self.data.items():
            if 'stockfish_level' in df.columns:
                # Calculate expected Elo based on performance
                total_score = 0
                total_weight = 0
                
                for level, elo in stockfish_elo_map.items():
                    level_games = df[df['stockfish_level'] == level]
                    if len(level_games) >= 10:  # Minimum games for reliability
                        win_rate = level_games['score'].mean()
                        # Elo difference formula
                        if 0 < win_rate < 1:
                            elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
                            estimated_elo = elo + elo_diff
                            weight = len(level_games)
                            total_score += estimated_elo * weight
                            total_weight += weight
                
                if total_weight > 0:
                    avg_elo = total_score / total_weight
                    model_elos.append({
                        'Model': model,
                        'Estimated Elo': avg_elo,
                        'Games': len(df)
                    })
        
        if model_elos:
            elo_df = pd.DataFrame(model_elos).sort_values('Estimated Elo', ascending=True)
            
            bars = ax.barh(range(len(elo_df)), elo_df['Estimated Elo'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(elo_df))))
            
            ax.set_yticks(range(len(elo_df)))
            ax.set_yticklabels(elo_df['Model'], fontsize=10)
            ax.set_xlabel('Estimated Elo Rating', fontsize=12)
            ax.set_title('Model Elo Estimation', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (elo, games) in enumerate(zip(elo_df['Estimated Elo'], elo_df['Games'])):
                ax.text(elo + 10, i, f'{int(elo)}', va='center', fontsize=9)
            
            ax.set_xlim(1000, max(elo_df['Estimated Elo']) + 200)
    
    def _plot_win_rate_heatmap(self, ax):
        """Create detailed win rate heatmap"""
        heatmap_data = []
        
        for model, df in self.data.items():
            if 'stockfish_level' in df.columns:
                for level in range(10):
                    level_games = df[df['stockfish_level'] == level]
                    if len(level_games) > 0:
                        wins = (level_games['score'] == 1.0).sum()
                        draws = (level_games['score'] == 0.5).sum()
                        losses = (level_games['score'] == 0.0).sum()
                        total = len(level_games)
                        
                        heatmap_data.append({
                            'Model': model,
                            'Stockfish Level': level,
                            'Win Rate': wins / total if total > 0 else 0,
                            'Draw Rate': draws / total if total > 0 else 0,
                            'Loss Rate': losses / total if total > 0 else 0,
                            'Score Rate': level_games['score'].mean(),
                            'Games': total
                        })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_df = heatmap_df.pivot(index='Model', columns='Stockfish Level', values='Score Rate')
            
            # Sort by average performance
            model_order = pivot_df.mean(axis=1).sort_values(ascending=False).index
            pivot_df = pivot_df.reindex(model_order)
            
            # Create custom colormap
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                       ax=ax, vmin=0, vmax=1,
                       cbar_kws={'label': 'Score Rate (W=1, D=0.5, L=0)'},
                       annot_kws={'fontsize': 9})
            
            ax.set_title('Score Rate Heatmap: Models vs Stockfish Levels', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Stockfish Level', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
    
    def _plot_game_length_distribution(self, ax):
        """Plot game length distributions with win rate overlay"""
        # Prepare data for violin plot
        length_data = []
        
        for model, df in self.data.items():
            if 'number_of_moves' in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row['number_of_moves']):
                        length_data.append({
                            'Model': model,
                            'Moves': row['number_of_moves'],
                            'Result': row['score']
                        })
        
        if length_data:
            length_df = pd.DataFrame(length_data)
            
            # Create violin plot for all games
            model_order = sorted(self.data.keys(), 
                               key=lambda x: self.data[x]['score'].mean(), 
                               reverse=True)
            
            sns.violinplot(data=length_df, x='Model', y='Moves', 
                          order=model_order, ax=ax, inner='box')
            
            # Overlay win rate by length
            ax2 = ax.twinx()
            
            for model in model_order:
                model_data = length_df[length_df['Model'] == model]
                if len(model_data) > 0:
                    # Calculate win rate in bins
                    bins = [0, 30, 50, 70, 100, 200]
                    win_rates = []
                    positions = []
                    
                    for i in range(len(bins)-1):
                        bin_data = model_data[
                            (model_data['Moves'] >= bins[i]) & 
                            (model_data['Moves'] < bins[i+1])
                        ]
                        if len(bin_data) > 10:
                            win_rate = (bin_data['Result'] == 1.0).mean()
                            win_rates.append(win_rate)
                            positions.append((bins[i] + bins[i+1]) / 2)
                    
                    if win_rates:
                        model_idx = model_order.index(model)
                        ax2.scatter([model_idx] * len(positions), positions, 
                                  s=100, c=win_rates, cmap='RdYlGn', 
                                  vmin=0, vmax=1, alpha=0.7, 
                                  edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Game Length (moves)', fontsize=12)
            ax2.set_ylabel('Win Rate by Length (color)', fontsize=12)
            ax.set_title('Game Length Distribution & Win Rates', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_illegal_move_patterns(self, ax):
        """Analyze illegal move patterns"""
        illegal_data = []
        
        for model, df in self.data.items():
            if 'player_one_illegal_moves' in df.columns:
                # Overall illegal move rate
                total_illegal = df['player_one_illegal_moves'].sum()
                total_moves = df['total_moves'].sum()
                games_with_illegal = (df['player_one_illegal_moves'] > 0).sum()
                
                illegal_data.append({
                    'Model': model,
                    'Illegal Move Rate': total_illegal / total_moves if total_moves > 0 else 0,
                    'Games with Illegal Moves': games_with_illegal / len(df) if len(df) > 0 else 0,
                    'Avg Illegal per Game': total_illegal / len(df) if len(df) > 0 else 0
                })
        
        if illegal_data:
            illegal_df = pd.DataFrame(illegal_data).sort_values('Illegal Move Rate')
            
            # Create grouped bar chart
            x = np.arange(len(illegal_df))
            width = 0.25
            
            ax.bar(x - width, illegal_df['Illegal Move Rate'] * 100, width, 
                   label='Illegal Move Rate (%)', color='red', alpha=0.7)
            ax.bar(x, illegal_df['Games with Illegal Moves'] * 100, width,
                   label='Games with Illegal (%)', color='orange', alpha=0.7)
            ax.bar(x + width, illegal_df['Avg Illegal per Game'], width,
                   label='Avg Illegal/Game', color='yellow', alpha=0.7)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Rate / Percentage', fontsize=12)
            ax.set_title('Illegal Move Analysis', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(illegal_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_performance_by_phase(self, ax):
        """Analyze performance by game phase"""
        phase_data = []
        
        for model, df in self.data.items():
            if 'number_of_moves' in df.columns:
                # Define game phases
                opening = df[df['number_of_moves'] <= 15]
                middlegame = df[(df['number_of_moves'] > 15) & (df['number_of_moves'] <= 40)]
                endgame = df[df['number_of_moves'] > 40]
                
                for phase_name, phase_df in [('Opening', opening), 
                                            ('Middlegame', middlegame), 
                                            ('Endgame', endgame)]:
                    if len(phase_df) > 0:
                        phase_data.append({
                            'Model': model,
                            'Phase': phase_name,
                            'Win Rate': (phase_df['score'] == 1.0).mean(),
                            'Score Rate': phase_df['score'].mean(),
                            'Games': len(phase_df)
                        })
        
        if phase_data:
            phase_df = pd.DataFrame(phase_data)
            
            # Create grouped bar chart
            models = phase_df['Model'].unique()
            phases = ['Opening', 'Middlegame', 'Endgame']
            
            # Sort models by overall performance
            model_order = sorted(models, 
                               key=lambda x: self.data[x]['score'].mean(), 
                               reverse=True)
            
            x = np.arange(len(model_order))
            width = 0.25
            
            for i, phase in enumerate(phases):
                phase_scores = []
                for model in model_order:
                    model_phase = phase_df[(phase_df['Model'] == model) & 
                                         (phase_df['Phase'] == phase)]
                    if len(model_phase) > 0:
                        phase_scores.append(model_phase['Score Rate'].values[0])
                    else:
                        phase_scores.append(0)
                
                ax.bar(x + i*width - width, phase_scores, width, 
                       label=phase, alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Score Rate', fontsize=12)
            ax.set_title('Performance by Game Phase', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    def _plot_architecture_comparison(self, ax):
        """Compare model architectures and their performance"""
        # Extract architecture info from model names
        arch_data = []
        
        for model, df in self.data.items():
            # Parse model architecture
            if 'small' in model:
                size = 'Small'
                params = 25  # Approximate
            elif 'medium' in model:
                size = 'Medium'
                params = 50  # Approximate
            elif 'large' in model:
                size = 'Large'
                params = 100  # Approximate
            else:
                size = 'Custom'
                params = 50
            
            # Extract layers
            layers_match = re.search(r'(\d+)layer', model)
            if not layers_match:
                layers_match = re.search(r'-(\d+)-', model)
            
            layers = int(layers_match.group(1)) if layers_match else 8
            
            # Calculate performance metrics
            avg_score = df['score'].mean()
            illegal_rate = df['player_one_illegal_moves'].sum() / df['total_moves'].sum() if df['total_moves'].sum() > 0 else 0
            
            arch_data.append({
                'Model': model,
                'Size': size,
                'Layers': layers,
                'Parameters (M)': params,
                'Avg Score': avg_score,
                'Illegal Rate': illegal_rate,
                'Games': len(df)
            })
        
        if arch_data:
            arch_df = pd.DataFrame(arch_data)
            
            # Create scatter plot
            scatter = ax.scatter(arch_df['Layers'], arch_df['Avg Score'],
                               s=arch_df['Parameters (M)'] * 5,
                               c=arch_df['Illegal Rate'],
                               cmap='RdYlGn_r', alpha=0.7,
                               edgecolors='black', linewidth=2)
            
            # Add labels
            for _, row in arch_df.iterrows():
                ax.annotate(row['Model'], 
                          (row['Layers'], row['Avg Score']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Average Score', fontsize=12)
            ax.set_title('Architecture Impact on Performance', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Illegal Move Rate', fontsize=10)
            
            # Add size legend
            sizes = [25, 50, 100]
            labels = ['25M', '50M', '100M']
            legend_elements = [plt.scatter([], [], s=s*5, c='gray', alpha=0.6, 
                                         edgecolors='black', linewidth=2, 
                                         label=l) for s, l in zip(sizes, labels)]
            ax.legend(handles=legend_elements, title='Parameters', 
                     loc='lower right', fontsize=8)
            
            ax.grid(True, alpha=0.3)
    
    def _create_game_analysis(self, output_dir: str, timestamp: str):
        """Create detailed game analysis visualizations"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Move Quality Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_move_quality_over_time(ax1)
        
        # 2. Opening Performance
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_opening_performance(ax2)
        
        # 3. Time Pressure Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_time_pressure_analysis(ax3)
        
        # 4. Resignation Patterns
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_resignation_patterns(ax4)
        
        # 5. Draw Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_draw_analysis(ax5)
        
        # 6. Critical Position Analysis
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_critical_positions(ax6)
        
        plt.suptitle('Detailed Game Analysis', fontsize=20, y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/game_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_move_quality_over_time(self, ax):
        """Analyze how move quality changes throughout the game"""
        move_quality_data = []
        
        for model, df in self.data.items():
            # Analyze illegal move timing
            if 'parsed_illegal_moves' in df.columns:
                for _, game in df.iterrows():
                    if isinstance(game['parsed_illegal_moves'], list):
                        for illegal_move in game['parsed_illegal_moves']:
                            if isinstance(illegal_move, dict) and 'move_number' in illegal_move:
                                move_quality_data.append({
                                    'Model': model,
                                    'Move Number': illegal_move['move_number'],
                                    'Type': 'Illegal',
                                    'Stockfish Level': game['stockfish_level']
                                })
        
        if move_quality_data:
            quality_df = pd.DataFrame(move_quality_data)
            
            # Plot illegal move frequency by move number
            move_bins = list(range(0, 101, 10))
            
            for model in self.data.keys():
                model_data = quality_df[quality_df['Model'] == model]
                if len(model_data) > 0:
                    hist, bins = np.histogram(model_data['Move Number'], bins=move_bins)
                    # Normalize by total games
                    hist = hist / len(self.data[model])
                    
                    ax.plot(bins[:-1] + 5, hist, marker='o', label=model, linewidth=2)
            
            ax.set_xlabel('Move Number', fontsize=12)
            ax.set_ylabel('Illegal Moves per Game', fontsize=12)
            ax.set_title('Illegal Move Frequency by Game Progress', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_opening_performance(self, ax):
        """Analyze performance by opening type"""
        opening_stats = []
        
        for model, df in self.data.items():
            if 'opening_category' in df.columns:
                for opening in df['opening_category'].unique():
                    opening_games = df[df['opening_category'] == opening]
                    if len(opening_games) >= 5:  # Minimum games
                        opening_stats.append({
                            'Model': model,
                            'Opening': opening,
                            'Score Rate': opening_games['score'].mean(),
                            'Games': len(opening_games)
                        })
        
        if opening_stats:
            opening_df = pd.DataFrame(opening_stats)
            
            # Create heatmap of openings vs models
            pivot_df = opening_df.pivot_table(
                index='Opening', 
                columns='Model', 
                values='Score Rate',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                       ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Score Rate'})
            
            ax.set_title('Performance by Opening Type', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Opening', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_time_pressure_analysis(self, ax):
        """Analyze performance under time pressure (late game)"""
        time_pressure_data = []
        
        for model, df in self.data.items():
            # Define time pressure as games going beyond move 60
            long_games = df[df['number_of_moves'] > 60]
            
            if len(long_games) > 0:
                # Calculate performance in late game
                late_game_score = long_games['score'].mean()
                late_game_illegal = long_games['player_one_illegal_moves'].mean()
                
                time_pressure_data.append({
                    'Model': model,
                    'Late Game Score': late_game_score,
                    'Late Game Illegal Moves': late_game_illegal,
                    'Long Games': len(long_games),
                    'Long Game %': len(long_games) / len(df) * 100
                })
        
        if time_pressure_data:
            pressure_df = pd.DataFrame(time_pressure_data).sort_values('Late Game Score')
            
            # Create dual-axis plot
            ax2 = ax.twinx()
            
            x = range(len(pressure_df))
            ax.bar(x, pressure_df['Late Game Score'], alpha=0.7, color='green', label='Score Rate')
            ax2.bar(x, pressure_df['Late Game Illegal Moves'], alpha=0.5, color='red', 
                   label='Illegal Moves', width=0.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(pressure_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('Late Game Score Rate', fontsize=12, color='green')
            ax2.set_ylabel('Avg Illegal Moves', fontsize=12, color='red')
            ax.set_title('Performance Under Time Pressure (>60 moves)', fontsize=14, fontweight='bold')
            
            # Add game count annotations
            for i, (games, pct) in enumerate(zip(pressure_df['Long Games'], pressure_df['Long Game %'])):
                ax.text(i, 0.02, f'{games}\n({pct:.1f}%)', ha='center', fontsize=8)
    
    def _plot_resignation_patterns(self, ax):
        """Analyze resignation patterns"""
        resignation_data = []
        
        for model, df in self.data.items():
            if 'player_one_resignation' in df.columns:
                resignations = df[df['player_one_resignation'] == True]
                
                resignation_data.append({
                    'Model': model,
                    'Resignation Rate': len(resignations) / len(df) * 100,
                    'Avg Resignation Move': resignations['number_of_moves'].mean() if len(resignations) > 0 else 0,
                    'Total Resignations': len(resignations)
                })
        
        if resignation_data:
            resign_df = pd.DataFrame(resignation_data).sort_values('Resignation Rate')
            
            # Create bar plot with annotations
            bars = ax.bar(range(len(resign_df)), resign_df['Resignation Rate'],
                          color=plt.cm.Reds(np.linspace(0.3, 0.9, len(resign_df))))
            
            ax.set_xticks(range(len(resign_df)))
            ax.set_xticklabels(resign_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('Resignation Rate (%)', fontsize=12)
            ax.set_title('Resignation Patterns', fontsize=14, fontweight='bold')
            
            # Add annotations
            for i, (rate, move, total) in enumerate(zip(resign_df['Resignation Rate'],
                                                       resign_df['Avg Resignation Move'],
                                                       resign_df['Total Resignations'])):
                if total > 0:
                    ax.text(i, rate + 0.5, f'{rate:.1f}%\nMove {move:.0f}', 
                           ha='center', fontsize=8)
    
    def _plot_draw_analysis(self, ax):
        """Analyze draw patterns"""
        draw_data = []
        
        for model, df in self.data.items():
            draws = df[df['score'] == 0.5]
            
            if len(draws) > 0:
                # Categorize draws by game length
                short_draws = draws[draws['number_of_moves'] < 30]
                medium_draws = draws[(draws['number_of_moves'] >= 30) & 
                                   (draws['number_of_moves'] < 60)]
                long_draws = draws[draws['number_of_moves'] >= 60]
                
                draw_data.append({
                    'Model': model,
                    'Draw Rate': len(draws) / len(df) * 100,
                    'Short Draws': len(short_draws),
                    'Medium Draws': len(medium_draws),
                    'Long Draws': len(long_draws),
                    'Avg Draw Length': draws['number_of_moves'].mean()
                })
        
        if draw_data:
            draw_df = pd.DataFrame(draw_data).sort_values('Draw Rate')
            
            # Create stacked bar chart
            x = range(len(draw_df))
            
            ax.bar(x, draw_df['Short Draws'], label='Short (<30)', color='lightblue')
            ax.bar(x, draw_df['Medium Draws'], bottom=draw_df['Short Draws'],
                   label='Medium (30-60)', color='blue')
            ax.bar(x, draw_df['Long Draws'], 
                   bottom=draw_df['Short Draws'] + draw_df['Medium Draws'],
                   label='Long (>60)', color='darkblue')
            
            ax.set_xticks(x)
            ax.set_xticklabels(draw_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('Number of Draws', fontsize=12)
            ax.set_title('Draw Analysis by Game Length', fontsize=14, fontweight='bold')
            ax.legend()
            
            # Add draw rate annotations
            for i, rate in enumerate(draw_df['Draw Rate']):
                total_draws = (draw_df.iloc[i]['Short Draws'] + 
                             draw_df.iloc[i]['Medium Draws'] + 
                             draw_df.iloc[i]['Long Draws'])
                ax.text(i, total_draws + 1, f'{rate:.1f}%', ha='center', fontsize=8)
    
    def _plot_critical_positions(self, ax):
        """Analyze performance in critical positions"""
        # This would require Stockfish analysis - placeholder for now
        critical_data = []
        
        for model, df in self.data.items():
            # Analyze games with many illegal moves as proxy for difficult positions
            difficult_games = df[df['player_one_illegal_moves'] > 2]
            
            if len(difficult_games) > 0:
                recovery_rate = (difficult_games['score'] == 1.0).mean()
                
                critical_data.append({
                    'Model': model,
                    'Difficult Games': len(difficult_games),
                    'Recovery Rate': recovery_rate,
                    'Avg Illegal in Difficult': difficult_games['player_one_illegal_moves'].mean()
                })
        
        if critical_data:
            critical_df = pd.DataFrame(critical_data).sort_values('Recovery Rate')
            
            # Create scatter plot
            ax.scatter(critical_df['Difficult Games'], 
                      critical_df['Recovery Rate'] * 100,
                      s=critical_df['Avg Illegal in Difficult'] * 50,
                      alpha=0.6, edgecolors='black', linewidth=2)
            
            # Add labels
            for _, row in critical_df.iterrows():
                ax.annotate(row['Model'], 
                          (row['Difficult Games'], row['Recovery Rate'] * 100),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Number of Difficult Games (>2 illegal moves)', fontsize=12)
            ax.set_ylabel('Win Rate in Difficult Games (%)', fontsize=12)
            ax.set_title('Performance in Critical Positions', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _create_error_analysis(self, output_dir: str, timestamp: str):
        """Create detailed error analysis"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Error Type Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_error_types(ax1)
        
        # 2. Error Patterns by Move
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_error_patterns(ax2)
        
        # 3. Recovery from Errors
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_error_recovery(ax3)
        
        # 4. Consecutive Errors
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_consecutive_errors(ax4)
        
        # 5. Error Correlation
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_error_correlation(ax5)
        
        plt.suptitle('Error Pattern Analysis', fontsize=20, y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_types(self, ax):
        """Analyze types of errors"""
        error_types = defaultdict(lambda: defaultdict(int))
        
        for model, df in self.data.items():
            if 'parsed_illegal_moves' in df.columns:
                for _, game in df.iterrows():
                    if isinstance(game['parsed_illegal_moves'], list):
                        for error in game['parsed_illegal_moves']:
                            if isinstance(error, dict) and 'error_type' in error:
                                error_types[model][error['error_type']] += 1
        
        if error_types:
            # Create stacked bar chart
            models = list(error_types.keys())
            error_type_names = set()
            for model_errors in error_types.values():
                error_type_names.update(model_errors.keys())
            error_type_names = sorted(list(error_type_names))
            
            bottom = np.zeros(len(models))
            colors = plt.cm.Set3(np.linspace(0, 1, len(error_type_names)))
            
            for i, error_type in enumerate(error_type_names):
                values = [error_types[model].get(error_type, 0) for model in models]
                ax.bar(range(len(models)), values, bottom=bottom, 
                       label=error_type, color=colors[i])
                bottom += values
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Number of Errors', fontsize=12)
            ax.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_error_patterns(self, ax):
        """Analyze when errors occur in games"""
        error_timing = defaultdict(list)
        
        for model, df in self.data.items():
            if 'parsed_illegal_moves' in df.columns:
                for _, game in df.iterrows():
                    if isinstance(game['parsed_illegal_moves'], list):
                        for error in game['parsed_illegal_moves']:
                            if isinstance(error, dict) and 'move_number' in error:
                                # Normalize by game length
                                if game['number_of_moves'] > 0:
                                    relative_position = error['move_number'] / game['number_of_moves']
                                    error_timing[model].append(relative_position)
        
        if error_timing:
            # Create violin plot
            data_for_plot = []
            for model, positions in error_timing.items():
                for pos in positions:
                    data_for_plot.append({'Model': model, 'Relative Position': pos})
            
            timing_df = pd.DataFrame(data_for_plot)
            
            model_order = sorted(error_timing.keys(), 
                               key=lambda x: len(error_timing[x]), reverse=True)
            
            sns.violinplot(data=timing_df, x='Model', y='Relative Position',
                          order=model_order, ax=ax)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Relative Position in Game (0=start, 1=end)', fontsize=12)
            ax.set_title('Error Timing Patterns', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Add horizontal lines for game phases
            ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Opening End')
            ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Endgame Start')
            ax.legend()
    
    def _plot_error_recovery(self, ax):
        """Analyze recovery from errors"""
        recovery_data = []
        
        for model, df in self.data.items():
            # Games with errors
            error_games = df[df['player_one_illegal_moves'] > 0]
            
            if len(error_games) > 0:
                # Group by number of errors
                for n_errors in range(1, 6):
                    games_with_n_errors = error_games[error_games['player_one_illegal_moves'] == n_errors]
                    if len(games_with_n_errors) > 0:
                        win_rate = (games_with_n_errors['score'] == 1.0).mean()
                        recovery_data.append({
                            'Model': model,
                            'Errors': n_errors,
                            'Win Rate': win_rate,
                            'Games': len(games_with_n_errors)
                        })
        
        if recovery_data:
            recovery_df = pd.DataFrame(recovery_data)
            
            # Create line plot
            for model in recovery_df['Model'].unique():
                model_data = recovery_df[recovery_df['Model'] == model]
                ax.plot(model_data['Errors'], model_data['Win Rate'] * 100,
                       marker='o', label=model, linewidth=2)
            
            ax.set_xlabel('Number of Illegal Moves', fontsize=12)
            ax.set_ylabel('Win Rate (%)', fontsize=12)
            ax.set_title('Recovery Rate After Errors', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 6))
    
    def _plot_consecutive_errors(self, ax):
        """Analyze consecutive error patterns"""
        consecutive_data = []
        
        for model, df in self.data.items():
            if 'parsed_illegal_moves' in df.columns:
                max_consecutive = []
                
                for _, game in df.iterrows():
                    if isinstance(game['parsed_illegal_moves'], list) and len(game['parsed_illegal_moves']) > 0:
                        # Find maximum consecutive errors
                        errors = sorted(game['parsed_illegal_moves'], 
                                      key=lambda x: x.get('move_number', 0) if isinstance(x, dict) else 0)
                        
                        max_consec = 1
                        current_consec = 1
                        
                        for i in range(1, len(errors)):
                            if (isinstance(errors[i], dict) and isinstance(errors[i-1], dict) and
                                'move_number' in errors[i] and 'move_number' in errors[i-1]):
                                if errors[i]['move_number'] == errors[i-1]['move_number']:
                                    current_consec += 1
                                    max_consec = max(max_consec, current_consec)
                                else:
                                    current_consec = 1
                        
                        max_consecutive.append(max_consec)
                
                if max_consecutive:
                    consecutive_data.append({
                        'Model': model,
                        'Avg Max Consecutive': np.mean(max_consecutive),
                        'Max Ever': max(max_consecutive),
                        'Games with Consecutive': len(max_consecutive)
                    })
        
        if consecutive_data:
            consec_df = pd.DataFrame(consecutive_data).sort_values('Avg Max Consecutive')
            
            # Create bar plot
            x = range(len(consec_df))
            ax.bar(x, consec_df['Avg Max Consecutive'], alpha=0.7, label='Average')
            ax.scatter(x, consec_df['Max Ever'], color='red', s=100, 
                      zorder=5, label='Maximum')
            
            ax.set_xticks(x)
            ax.set_xticklabels(consec_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('Consecutive Errors', fontsize=12)
            ax.set_title('Consecutive Error Analysis', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_error_correlation(self, ax):
        """Analyze correlation between errors and game outcomes"""
        correlation_data = []
        
        for model, df in self.data.items():
            if len(df) > 10:  # Need sufficient data
                # Calculate correlations
                corr_illegal_score = df['player_one_illegal_moves'].corr(df['score'])
                corr_illegal_length = df['player_one_illegal_moves'].corr(df['number_of_moves'])
                corr_length_score = df['number_of_moves'].corr(df['score'])
                
                correlation_data.append({
                    'Model': model,
                    'Illegal-Score': corr_illegal_score,
                    'Illegal-Length': corr_illegal_length,
                    'Length-Score': corr_length_score
                })
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            
            # Create heatmap
            corr_matrix = corr_df.set_index('Model').T
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, vmin=-1, vmax=1, ax=ax,
                       cbar_kws={'label': 'Correlation'})
            
            ax.set_title('Feature Correlations', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
    
    def _create_opening_analysis(self, output_dir: str, timestamp: str):
        """Create opening-specific analysis"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Opening Frequency
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_opening_frequency(ax1)
        
        # 2. Opening Success Rates
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_opening_success(ax2)
        
        # 3. Opening Depth Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_opening_depth(ax3)
        
        # 4. Opening Transitions
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_opening_transitions(ax4)
        
        plt.suptitle('Opening Analysis', fontsize=20, y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/opening_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_opening_frequency(self, ax):
        """Plot frequency of different openings"""
        opening_counts = defaultdict(lambda: defaultdict(int))
        
        for model, df in self.data.items():
            if 'opening_category' in df.columns:
                for opening in df['opening_category'].value_counts().index[:10]:
                    count = (df['opening_category'] == opening).sum()
                    opening_counts[opening][model] = count
        
        if opening_counts:
            # Create grouped bar chart
            openings = list(opening_counts.keys())
            models = list(self.data.keys())
            
            x = np.arange(len(openings))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                values = [opening_counts[opening].get(model, 0) for opening in openings]
                ax.bar(x + i * width - 0.4 + width/2, values, width, 
                       label=model, alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(openings, rotation=45, ha='right')
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Opening Frequency by Model', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_opening_success(self, ax):
        """Plot success rates for different openings"""
        opening_success = []
        
        for model, df in self.data.items():
            if 'opening_category' in df.columns:
                for opening in df['opening_category'].unique():
                    opening_games = df[df['opening_category'] == opening]
                    if len(opening_games) >= 10:  # Minimum games
                        opening_success.append({
                            'Model': model,
                            'Opening': opening,
                            'Win Rate': (opening_games['score'] == 1.0).mean(),
                            'Score Rate': opening_games['score'].mean(),
                            'Games': len(opening_games)
                        })
        
        if opening_success:
            success_df = pd.DataFrame(opening_success)
            
            # Create box plot
            openings = success_df['Opening'].unique()
            
            data_for_plot = []
            for opening in openings:
                opening_data = success_df[success_df['Opening'] == opening]
                if len(opening_data) >= 3:  # Need multiple models
                    data_for_plot.extend([
                        {'Opening': opening, 'Score Rate': score}
                        for score in opening_data['Score Rate']
                    ])
            
            if data_for_plot:
                plot_df = pd.DataFrame(data_for_plot)
                sns.boxplot(data=plot_df, x='Opening', y='Score Rate', ax=ax)
                
                ax.set_xlabel('Opening', fontsize=12)
                ax.set_ylabel('Score Rate Distribution', fontsize=12)
                ax.set_title('Opening Success Rate Distribution', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
    
    def _plot_opening_depth(self, ax):
        """Analyze how deep into openings models go"""
        depth_data = []
        
        for model, df in self.data.items():
            if 'transcript' in df.columns:
                opening_lengths = []
                
                for _, game in df.iterrows():
                    # Count moves in first 10 moves that match known patterns
                    transcript = game['transcript']
                    moves = re.findall(r'\d+\.\s*(\S+)(?:\s+(\S+))?', transcript)[:10]
                    opening_lengths.append(len(moves))
                
                if opening_lengths:
                    depth_data.append({
                        'Model': model,
                        'Avg Opening Depth': np.mean(opening_lengths),
                        'Std Opening Depth': np.std(opening_lengths)
                    })
        
        if depth_data:
            depth_df = pd.DataFrame(depth_data).sort_values('Avg Opening Depth')
            
            # Create bar plot with error bars
            x = range(len(depth_df))
            ax.bar(x, depth_df['Avg Opening Depth'], 
                   yerr=depth_df['Std Opening Depth'],
                   capsize=5, alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(depth_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('Average Opening Depth (moves)', fontsize=12)
            ax.set_title('Opening Knowledge Depth', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_opening_transitions(self, ax):
        """Analyze transitions from opening to middlegame"""
        transition_data = []
        
        for model, df in self.data.items():
            # Analyze performance drop after opening
            early_games = df[df['number_of_moves'] <= 15]
            mid_games = df[(df['number_of_moves'] > 15) & (df['number_of_moves'] <= 30)]
            
            if len(early_games) > 0 and len(mid_games) > 0:
                early_score = early_games['score'].mean()
                mid_score = mid_games['score'].mean()
                
                transition_data.append({
                    'Model': model,
                    'Opening Score': early_score,
                    'Early Middlegame Score': mid_score,
                    'Transition Drop': early_score - mid_score
                })
        
        if transition_data:
            trans_df = pd.DataFrame(transition_data).sort_values('Transition Drop')
            
            # Create slope plot
            for _, row in trans_df.iterrows():
                ax.plot([0, 1], [row['Opening Score'], row['Early Middlegame Score']],
                       'o-', label=row['Model'], linewidth=2, markersize=8)
                
                # Add drop annotation
                mid_point = (row['Opening Score'] + row['Early Middlegame Score']) / 2
                ax.text(0.5, mid_point, f"{row['Transition Drop']:.3f}", 
                       ha='center', fontsize=8)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Opening\n(≤15 moves)', 'Early Middlegame\n(16-30 moves)'])
            ax.set_ylabel('Score Rate', fontsize=12)
            ax.set_title('Opening to Middlegame Transition', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _create_interactive_dashboard(self, output_dir: str, timestamp: str):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Performance vs Stockfish Level',
                'Win/Draw/Loss Distribution',
                'Game Length Distribution',
                'Illegal Move Patterns',
                'Opening Performance',
                'Model Comparison Radar',
                'Performance Heatmap',
                'Error Timeline',
                'Elo Progression'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'violin'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatterpolar'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Performance vs Stockfish Level
        for model, df in self.data.items():
            if 'stockfish_level' in df.columns:
                level_stats = df.groupby('stockfish_level')['score'].agg(['mean', 'count']).reset_index()
                level_stats = level_stats[level_stats['count'] >= 20]
                
                if len(level_stats) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=level_stats['stockfish_level'],
                            y=level_stats['mean'],
                            mode='lines+markers',
                            name=model,
                            line=dict(width=2),
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )
        
        # 2. Win/Draw/Loss Distribution
        wdl_data = []
        for model, df in self.data.items():
            wins = (df['score'] == 1.0).sum()
            draws = (df['score'] == 0.5).sum()
            losses = (df['score'] == 0.0).sum()
            wdl_data.append({
                'Model': model,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses
            })
        
        if wdl_data:
            wdl_df = pd.DataFrame(wdl_data)
            
            fig.add_trace(
                go.Bar(name='Wins', x=wdl_df['Model'], y=wdl_df['Wins'],
                      marker_color='green'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='Draws', x=wdl_df['Model'], y=wdl_df['Draws'],
                      marker_color='gray'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='Losses', x=wdl_df['Model'], y=wdl_df['Losses'],
                      marker_color='red'),
                row=1, col=2
            )
        
        # 3. Game Length Distribution (Violin plot)
        for model, df in self.data.items():
            if 'number_of_moves' in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df['number_of_moves'].dropna(),
                        name=model,
                        box_visible=True,
                        meanline_visible=True
                    ),
                    row=1, col=3
                )
        
        # 4. Illegal Move Patterns (Scatter)
        for model, df in self.data.items():
            if 'number_of_moves' in df.columns and 'player_one_illegal_moves' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['number_of_moves'],
                        y=df['player_one_illegal_moves'],
                        mode='markers',
                        name=model,
                        marker=dict(size=5, opacity=0.5)
                    ),
                    row=2, col=1
                )
        
        # 5. Opening Performance (Bar)
        opening_perf = []
        for model, df in self.data.items():
            if 'opening_category' in df.columns:
                for opening in ['King\'s Pawn', 'Queen\'s Pawn', 'Sicilian Defense']:
                    opening_games = df[df['opening_category'] == opening]
                    if len(opening_games) >= 5:
                        opening_perf.append({
                            'Model': model,
                            'Opening': opening,
                            'Score': opening_games['score'].mean()
                        })
        
        if opening_perf:
            opening_df = pd.DataFrame(opening_perf)
            for opening in opening_df['Opening'].unique():
                opening_data = opening_df[opening_df['Opening'] == opening]
                fig.add_trace(
                    go.Bar(
                        name=opening,
                        x=opening_data['Model'],
                        y=opening_data['Score']
                    ),
                    row=2, col=2
                )
        
        # 6. Model Comparison Radar
        radar_data = []
        for model, df in self.data.items():
            avg_score = df['score'].mean()
            illegal_rate = 1 - (df['player_one_illegal_moves'].sum() / df['total_moves'].sum() if df['total_moves'].sum() > 0 else 0)
            avg_length = df['number_of_moves'].mean() / 100  # Normalize
            consistency = 1 - df['score'].std()
            
            radar_data.append({
                'Model': model,
                'Performance': avg_score,
                'Legal Move Rate': illegal_rate,
                'Game Length': avg_length,
                'Consistency': consistency
            })
        
        if radar_data:
            for i, data in enumerate(radar_data[:5]):  # Top 5 models
                fig.add_trace(
                    go.Scatterpolar(
                        r=[data['Performance'], data['Legal Move Rate'], 
                           data['Game Length'], data['Consistency']],
                        theta=['Performance', 'Legal Moves', 'Game Length', 'Consistency'],
                        fill='toself',
                        name=data['Model']
                    ),
                    row=2, col=3
                )
        
        # 7. Performance Heatmap
        heatmap_data = []
        models = list(self.data.keys())
        stockfish_levels = list(range(10))
        
        z_data = []
        for model in models:
            row_data = []
            df = self.data[model]
            for level in stockfish_levels:
                level_games = df[df['stockfish_level'] == level] if 'stockfish_level' in df.columns else pd.DataFrame()
                score = level_games['score'].mean() if len(level_games) > 0 else 0
                row_data.append(score)
            z_data.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=[f'Level {i}' for i in stockfish_levels],
                y=models,
                colorscale='RdYlGn',
                zmid=0.5
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Chess GPT Interactive Dashboard - {timestamp}",
            showlegend=True,
            height=1500,
            width=1800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Stockfish Level", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Number of Games", row=1, col=2)
        
        fig.update_yaxes(title_text="Number of Moves", row=1, col=3)
        
        fig.update_xaxes(title_text="Game Length", row=2, col=1)
        fig.update_yaxes(title_text="Illegal Moves", row=2, col=1)
        
        # Save interactive dashboard
        fig.write_html(f"{output_dir}/interactive_dashboard_{timestamp}.html")
        print(f"Interactive dashboard saved to {output_dir}/interactive_dashboard_{timestamp}.html")
    
    def _create_comparison_report(self, output_dir: str, timestamp: str):
        """Create comprehensive comparison report"""
        report = []
        report.append("# Chess GPT Model Comparison Report")
        report.append(f"Generated: {timestamp}\n")
        
        # Model Summary Table
        report.append("## Model Performance Summary\n")
        summary_data = []
        
        for model, df in self.data.items():
            summary = {
                'Model': model,
                'Total Games': len(df),
                'Avg Score': f"{df['score'].mean():.3f}",
                'Win Rate': f"{(df['score'] == 1.0).mean():.1%}",
                'Draw Rate': f"{(df['score'] == 0.5).mean():.1%}",
                'Loss Rate': f"{(df['score'] == 0.0).mean():.1%}",
                'Avg Game Length': f"{df['number_of_moves'].mean():.1f}",
                'Illegal Move Rate': f"{(df['player_one_illegal_moves'].sum() / df['total_moves'].sum() * 100):.2f}%" if df['total_moves'].sum() > 0 else "0%",
                'Estimated Elo': self._estimate_elo(df)
            }
            summary_data.append(summary)
        
        # Sort by average score
        summary_data.sort(key=lambda x: float(x['Avg Score']), reverse=True)
        
        # Create markdown table
        headers = list(summary_data[0].keys())
        report.append("| " + " | ".join(headers) + " |")
        report.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in summary_data:
            report.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        
        report.append("\n## Key Findings\n")
        
        # Best performing model
        best_model = summary_data[0]
        report.append(f"### Best Performing Model: {best_model['Model']}")
        report.append(f"- Average Score: {best_model['Avg Score']}")
        report.append(f"- Estimated Elo: {best_model['Estimated Elo']}")
        report.append(f"- Win Rate: {best_model['Win Rate']}\n")
        
        # Model comparisons
        report.append("### Model Architecture Insights")
        
        # Analyze by size
        size_performance = defaultdict(list)
        for model, df in self.data.items():
            if 'small' in model:
                size = 'Small'
            elif 'medium' in model:
                size = 'Medium'
            elif 'large' in model:
                size = 'Large'
            else:
                size = 'Custom'
            
            size_performance[size].append(df['score'].mean())
        
        for size, scores in size_performance.items():
            if scores:
                report.append(f"- {size} models: Avg score = {np.mean(scores):.3f}")
        
        # Save report
        report_path = f"{output_dir}/comparison_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Comparison report saved to {report_path}")
    
    def _estimate_elo(self, df: pd.DataFrame) -> str:
        """Estimate Elo rating based on performance"""
        if 'stockfish_level' not in df.columns:
            return "N/A"
        
        # Stockfish level to Elo mapping (approximate)
        stockfish_elo = {
            0: 1300, 1: 1400, 2: 1500, 3: 1600, 4: 1700,
            5: 1800, 6: 1900, 7: 2000, 8: 2100, 9: 2200
        }
        
        total_elo = 0
        total_weight = 0
        
        for level, elo in stockfish_elo.items():
            level_games = df[df['stockfish_level'] == level]
            if len(level_games) >= 10:
                win_rate = level_games['score'].mean()
                if 0 < win_rate < 1:
                    elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
                    estimated_elo = elo + elo_diff
                    weight = len(level_games)
                    total_elo += estimated_elo * weight
                    total_weight += weight
        
        if total_weight > 0:
            return f"{int(total_elo / total_weight)}"
        return "N/A"
    
    def add_stockfish_annotations(self, game_ids: List[str], stockfish_path: str = None):
        """Add Stockfish analysis to specific games"""
        # This would integrate with chess.engine to analyze games
        # Placeholder for future implementation
        pass
    
    def export_for_mechanistic_interpretability(self, output_dir: str):
        """Export data in format suitable for mechanistic interpretability analysis"""
        # Export game states at each move for probe training
        mi_data = []
        
        for model, df in self.data.items():
            for _, game in df.iterrows():
                if 'transcript' in game:
                    # Parse game moves
                    board = chess.Board()
                    moves = self._parse_pgn_moves(game['transcript'])
                    
                    for move_num, (move, board_state) in enumerate(moves):
                        mi_data.append({
                            'model': model,
                            'game_id': game.get('game_id', ''),
                            'move_number': move_num,
                            'move': move,
                            'board_fen': board_state,
                            'result': game['result'],
                            'score': game['score']
                        })
        
        # Save as JSON for easy loading
        mi_df = pd.DataFrame(mi_data)
        mi_df.to_json(f"{output_dir}/mechanistic_interpretability_data.json", 
                      orient='records', indent=2)
        
        print(f"Exported {len(mi_data)} board states for mechanistic interpretability")
    
    def _parse_pgn_moves(self, pgn_string: str) -> List[Tuple[str, str]]:
        """Parse PGN string and return moves with board states"""
        moves_with_states = []
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
                board.push(move)
                moves_with_states.append((white_move, board.fen()))
                
                # Black move (if exists)
                if black_move:
                    move = board.parse_san(black_move)
                    board.push(move)
                    moves_with_states.append((black_move, board.fen()))
            except:
                break
        
        return moves_with_states


def main():
    """Main function to run the dashboard"""
    dashboard = ChessAnalysisDashboard()
    
    # Load data
    print("Loading chess game data...")
    dashboard.load_data()
    
    # Create comprehensive dashboard
    print("\nCreating comprehensive dashboard...")
    dashboard.create_comprehensive_dashboard()
    
    # Export data for mechanistic interpretability
    print("\nExporting data for mechanistic interpretability...")
    dashboard.export_for_mechanistic_interpretability("dashboard_output")
    
    print("\nDashboard generation complete!")


if __name__ == "__main__":
    main() 