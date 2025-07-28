#!/usr/bin/env python3
"""
Detailed Breakdown Analysis by Stockfish Level
Addresses confounding variables in game length and illegal move analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict, OrderedDict
import re
from matplotlib.backends.backend_pdf import PdfPages

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')  # Fallback style
sns.set_palette("husl")

# Function to extract Stockfish level from player_two column
def extract_stockfish_level(player_two):
    match = re.search(r'Stockfish (\d+)', str(player_two))
    if match:
        return int(match.group(1))
    return None

# Function to convert result to numeric score
def result_to_score(result):
    if result in ['1-0', '1']:
        return 1.0
    elif result in ['0-1', '0']:
        return 0.0
    elif result in ['1/2-1/2', '0.5-0.5', '1/2', '0.5']:
        return 0.5
    else:
        return None

# Read all CSV files
log_files = [
    'data/games/small-8-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/small-16-600k_iters_pt_vs_stockfish_sweep.csv', 
    'data/games/small-24-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/small-36-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/medium-12-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/medium-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/large-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/adam_stockfish_8layers_pt_vs_stockfish_sweep.csv',
    'data/games/adam_stockfish_16layers_pt_vs_stockfish_sweep.csv'
]

# Dictionary to store dataframes
all_data = OrderedDict()
model_names = []

for log_file in log_files:
    if os.path.exists(log_file):
        # Skip lichess models due to insufficient data points
        if 'lichess' in log_file:
            print(f"Skipping {log_file} - insufficient data points")
            continue
            
        df = pd.read_csv(log_file)
        model_name = log_file.split('/')[-1].replace('_vs_stockfish_sweep.csv', '').replace('-600k_iters_pt', '')
        model_names.append(model_name)
        
        # Add stockfish level and numeric score
        df['stockfish_level'] = df['player_two'].apply(extract_stockfish_level)
        df['score'] = df['result'].apply(result_to_score)
        
        # Ensure numeric columns are properly converted
        if 'number_of_moves' in df.columns:
            df['number_of_moves'] = pd.to_numeric(df['number_of_moves'], errors='coerce')
        if 'player_one_illegal_moves' in df.columns:
            df['player_one_illegal_moves'] = pd.to_numeric(df['player_one_illegal_moves'], errors='coerce')
        if 'total_moves' in df.columns:
            df['total_moves'] = pd.to_numeric(df['total_moves'], errors='coerce')
        
        all_data[model_name] = df
        print(f"Loaded {len(df)} games from {model_name}")

# Calculate model ordering for consistency
model_avg_scores = {}
for model, df in all_data.items():
    if 'stockfish_level' in df.columns:
        level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
        model_avg_scores[model] = level_means.mean()
    else:
        model_avg_scores[model] = df['score'].mean()

# Sort models by performance (best to worst)
raw_model_order = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)

# Create PDF with detailed breakdowns
def create_detailed_pdf():
    with PdfPages('chess_detailed_breakdown_by_stockfish.pdf') as pdf:
        
        # Page 1: Game Length Distribution by Stockfish Level
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))
        fig.suptitle('Game Length Distribution by Stockfish Level\n(Addressing Game Length vs Opponent Strength Confounding)', fontsize=16)
        
        stockfish_levels = range(10)
        colors = plt.cm.tab10(np.linspace(0, 1, len(raw_model_order)))
        
        for i, sf_level in enumerate(stockfish_levels):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            for j, model in enumerate(raw_model_order[:6]):  # Top 6 models
                if model in all_data:
                    df = all_data[model]
                    level_games = df[df['stockfish_level'] == sf_level]
                    
                    if len(level_games) > 20:
                        game_lengths = level_games['number_of_moves'].dropna()
                        if len(game_lengths) > 0:
                            ax.hist(game_lengths, bins=20, alpha=0.6, 
                                   label=model.replace('_pt', '').replace('_', ' '), 
                                   color=colors[j], density=True)
            
            ax.set_title(f'Stockfish Level {sf_level}')
            ax.set_xlabel('Game Length (moves)')
            ax.set_ylabel('Density')
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Illegal Move Rate by Stockfish Level and Game Length
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))
        fig.suptitle('Illegal Moves per Game by Stockfish Level\n(Isolating Illegal Move Patterns from Opponent Strength)', fontsize=16)
        
        for i, sf_level in enumerate(stockfish_levels):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            for j, model in enumerate(raw_model_order[:6]):  # Top 6 models
                if model in all_data:
                    df = all_data[model]
                    level_games = df[df['stockfish_level'] == sf_level]
                    
                    if len(level_games) > 50:
                        # Create game length bins
                        level_games_clean = level_games.dropna(subset=['number_of_moves', 'player_one_illegal_moves'])
                        
                        if len(level_games_clean) > 0:
                            # Bin by game length
                            bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
                            bin_centers = [25, 35, 45, 55, 65, 75, 85, 95]
                            
                            illegal_rates = []
                            valid_centers = []
                            
                            for k in range(len(bins)-1):
                                bin_games = level_games_clean[
                                    (level_games_clean['number_of_moves'] >= bins[k]) & 
                                    (level_games_clean['number_of_moves'] < bins[k+1])
                                ]
                                
                                if len(bin_games) > 10:
                                    avg_illegal = bin_games['player_one_illegal_moves'].mean()
                                    illegal_rates.append(avg_illegal)
                                    valid_centers.append(bin_centers[k])
                            
                            if len(illegal_rates) > 2:
                                ax.plot(valid_centers, illegal_rates, marker='o', 
                                       label=model.replace('_pt', '').replace('_', ' '), 
                                       color=colors[j], linewidth=2)
            
            ax.set_title(f'Stockfish Level {sf_level}')
            ax.set_xlabel('Game Length (moves)')
            ax.set_ylabel('Avg Illegal Moves per Game')
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Win Rate by Game Length for Each Stockfish Level
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))
        fig.suptitle('Win Rate by Game Length for Each Stockfish Level\n(Controlling for Opponent Strength)', fontsize=16)
        
        for i, sf_level in enumerate(stockfish_levels):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            for j, model in enumerate(raw_model_order[:6]):  # Top 6 models
                if model in all_data:
                    df = all_data[model]
                    level_games = df[df['stockfish_level'] == sf_level]
                    
                    if len(level_games) > 50:
                        level_games_clean = level_games.dropna(subset=['number_of_moves', 'score'])
                        
                        if len(level_games_clean) > 0:
                            # Create length bins
                            bins = [0, 40, 50, 60, 70, 80, 90, 100]
                            bin_centers = [35, 45, 55, 65, 75, 85, 95]
                            
                            win_rates = []
                            valid_centers = []
                            
                            for k in range(len(bins)-1):
                                bin_games = level_games_clean[
                                    (level_games_clean['number_of_moves'] >= bins[k]) & 
                                    (level_games_clean['number_of_moves'] < bins[k+1])
                                ]
                                
                                if len(bin_games) > 10:
                                    win_rate = bin_games['score'].mean()
                                    win_rates.append(win_rate)
                                    valid_centers.append(bin_centers[k])
                            
                            if len(win_rates) > 2:
                                ax.plot(valid_centers, win_rates, marker='o', 
                                       label=model.replace('_pt', '').replace('_', ' '), 
                                       color=colors[j], linewidth=2)
            
            ax.set_title(f'Stockfish Level {sf_level}')
            ax.set_xlabel('Game Length (moves)')
            ax.set_ylabel('Win Rate')
            ax.set_ylim(0, 1)
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Early Game Termination Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Early Game Termination Analysis\n(Understanding High Illegal Move Rates in Short Games)', fontsize=16)
        
        # Top left: Proportion of games ending early due to illegal moves
        ax1 = axes[0, 0]
        early_termination_data = []
        
        for model in raw_model_order[:6]:
            if model in all_data:
                df = all_data[model]
                df_clean = df.dropna(subset=['number_of_moves', 'player_one_illegal_moves'])
                
                # Games ending very early (< 20 moves)
                early_games = df_clean[df_clean['number_of_moves'] < 20]
                
                if len(early_games) > 0:
                    # High illegal move games (>3 illegal moves)
                    high_illegal = early_games[early_games['player_one_illegal_moves'] > 3]
                    proportion = len(high_illegal) / len(early_games)
                    
                    early_termination_data.append({
                        'Model': model.replace('_pt', '').replace('_', ' '),
                        'Proportion': proportion
                    })
        
        if early_termination_data:
            term_df = pd.DataFrame(early_termination_data)
            bars = ax1.bar(range(len(term_df)), term_df['Proportion'])
            ax1.set_xticks(range(len(term_df)))
            ax1.set_xticklabels(term_df['Model'], rotation=45, ha='right')
            ax1.set_ylabel('Proportion of Early Games\nwith High Illegal Moves')
            ax1.set_title('Early Termination Due to Illegal Moves')
            ax1.grid(True, alpha=0.3)
        
        # Top right: Average illegal moves by game outcome
        ax2 = axes[0, 1]
        outcome_illegal_data = []
        
        for model in raw_model_order[:6]:
            if model in all_data:
                df = all_data[model]
                df_clean = df.dropna(subset=['score', 'player_one_illegal_moves'])
                
                wins = df_clean[df_clean['score'] == 1.0]['player_one_illegal_moves'].mean()
                draws = df_clean[df_clean['score'] == 0.5]['player_one_illegal_moves'].mean()
                losses = df_clean[df_clean['score'] == 0.0]['player_one_illegal_moves'].mean()
                
                outcome_illegal_data.extend([
                    {'Model': model.replace('_pt', '').replace('_', ' '), 'Outcome': 'Win', 'Avg Illegal': wins},
                    {'Model': model.replace('_pt', '').replace('_', ' '), 'Outcome': 'Draw', 'Avg Illegal': draws},
                    {'Model': model.replace('_pt', '').replace('_', ' '), 'Outcome': 'Loss', 'Avg Illegal': losses}
                ])
        
        if outcome_illegal_data:
            outcome_df = pd.DataFrame(outcome_illegal_data)
            sns.barplot(data=outcome_df, x='Model', y='Avg Illegal', hue='Outcome', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_title('Illegal Moves by Game Outcome')
            ax2.grid(True, alpha=0.3)
        
        # Bottom left: Game length vs illegal moves scatter (sample)
        ax3 = axes[1, 0]
        
        # Sample data from best model for illustration
        best_model = raw_model_order[0]
        if best_model in all_data:
            df = all_data[best_model]
            df_sample = df.dropna(subset=['number_of_moves', 'player_one_illegal_moves']).sample(min(1000, len(df)))
            
            scatter = ax3.scatter(df_sample['number_of_moves'], df_sample['player_one_illegal_moves'], 
                                 alpha=0.6, c=df_sample['score'], cmap='RdYlGn')
            ax3.set_xlabel('Game Length (moves)')
            ax3.set_ylabel('Illegal Moves')
            ax3.set_title(f'Game Length vs Illegal Moves\n({best_model.replace("_", " ")} - Sample)')
            plt.colorbar(scatter, ax=ax3, label='Score')
            ax3.grid(True, alpha=0.3)
        
        # Bottom right: Stockfish level distribution in early vs late games
        ax4 = axes[1, 1]
        
        sf_distribution_data = []
        for model in raw_model_order[:3]:  # Top 3 models
            if model in all_data:
                df = all_data[model]
                df_clean = df.dropna(subset=['number_of_moves', 'stockfish_level'])
                
                early_games = df_clean[df_clean['number_of_moves'] < 30]
                late_games = df_clean[df_clean['number_of_moves'] > 60]
                
                if len(early_games) > 0 and len(late_games) > 0:
                    early_avg_sf = early_games['stockfish_level'].mean()
                    late_avg_sf = late_games['stockfish_level'].mean()
                    
                    sf_distribution_data.extend([
                        {'Model': model.replace('_pt', '').replace('_', ' '), 'Game Type': 'Early (<30)', 'Avg SF Level': early_avg_sf},
                        {'Model': model.replace('_pt', '').replace('_', ' '), 'Game Type': 'Late (>60)', 'Avg SF Level': late_avg_sf}
                    ])
        
        if sf_distribution_data:
            sf_dist_df = pd.DataFrame(sf_distribution_data)
            sns.barplot(data=sf_dist_df, x='Model', y='Avg SF Level', hue='Game Type', ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
            ax4.set_title('Stockfish Level Distribution\nEarly vs Late Games')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("Creating detailed breakdown analysis by Stockfish level...")
    create_detailed_pdf()
    print("Detailed analysis saved to: chess_detailed_breakdown_by_stockfish.pdf") 