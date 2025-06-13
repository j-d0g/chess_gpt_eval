#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict, OrderedDict
import re

# Set style for better-looking plots
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

# Read all CSV files in the desired order
log_files = [
    'logs/small-8-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/small-16-600k_iters_pt_vs_stockfish_sweep.csv', 
    'logs/small-24-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/small-36-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/medium-12-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/medium-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/large-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'logs/adam_lichess_8layers_pt_vs_stockfish_sweep.csv',
    'logs/adam_stockfish_8layers_pt_vs_stockfish_sweep.csv',
    'logs/adam_lichess_16layers_pt_vs_stockfish_sweep.csv',
    'logs/adam_stockfish_16layers_pt_vs_stockfish_sweep.csv'
]

# Dictionary to store dataframes (using OrderedDict to preserve order)
all_data = OrderedDict()
model_names = []

for log_file in log_files:
    if os.path.exists(log_file):
        # Skip lichess models due to insufficient data points
        if 'lichess' in log_file:
            print(f"Skipping {log_file} - insufficient data points")
            continue
            
        # Special handling for large-16 file which is missing header
        if 'large-16-600k_iters_pt_vs_stockfish_sweep.csv' in log_file:
            # Define the expected column names based on other files
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

# Create figure with multiple subplots (4x4 layout for comprehensive analysis)
fig = plt.figure(figsize=(20, 24))

# 1. Performance vs Stockfish level - THE KEY METRIC
ax1 = plt.subplot(4, 4, 1)

# Calculate average of per-level averages for fair ordering
model_avg_scores = {}
for model, df in all_data.items():
    if 'stockfish_level' in df.columns:
        level_means = df.groupby('stockfish_level')['score'].mean()
        model_avg_scores[model] = level_means.mean()
    else:
        model_avg_scores[model] = df['score'].mean()

# Plot in order of average score (best to worst)
for model in sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True):
    df = all_data[model]
    if 'stockfish_level' in df.columns:
        level_stats = df.groupby('stockfish_level')['score'].agg(['mean', 'count']).reset_index()
        ax1.plot(level_stats['stockfish_level'], level_stats['mean'], 
                marker='o', linewidth=2, markersize=8,
                label=model.replace('_pt', '').replace('_', ' '))

ax1.set_xlabel('Stockfish Level')
ax1.set_ylabel('Average Score (including draws as 0.5)')
ax1.set_title('Performance vs Stockfish Level - THE KEY METRIC')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Set better axis scaling for clearer distinctions
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.1, 0.1))  # 0.1 increments
ax1.set_xticks(range(10))  # Every integer stockfish level 0-9

# 2. Average score rate comparison (normalized win rate)
ax2 = plt.subplot(4, 4, 2)
avg_scores = []
for model, df in all_data.items():
    avg_score = df['score'].mean()
    total_games = len(df)
    avg_scores.append({
        'Model': model.replace('_pt', '').replace('_', ' '),
        'Avg Score': avg_score,
        'Games': total_games
    })

scores_df = pd.DataFrame(avg_scores).sort_values('Avg Score')
bars = ax2.bar(range(len(scores_df)), scores_df['Avg Score'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(scores_df))))
ax2.set_xticks(range(len(scores_df)))
ax2.set_xticklabels(scores_df['Model'], rotation=60, ha='right', fontsize=8)
ax2.set_ylabel('Average Score Rate')
ax2.set_ylim(0, 1.0)
ax2.set_title('Average Score Rate (Normalized Win Rate)')

# Add score labels on bars
for i, (score, games) in enumerate(zip(scores_df['Avg Score'], scores_df['Games'])):
    ax2.text(i, score + 0.01, f'{score:.3f}\n({games} games)', ha='center', fontsize=9)

# 3. Game length distribution (all games)
ax3 = plt.subplot(4, 4, 3)
all_game_lengths = []
for model, df in all_data.items():
    if 'number_of_moves' in df.columns:
        game_lengths = df['number_of_moves'].dropna()
        all_game_lengths.extend([(model.replace('_pt', '').replace('_', ' '), length) 
                                for length in game_lengths])

if all_game_lengths:
    length_df = pd.DataFrame(all_game_lengths, columns=['Model', 'Moves'])
    
    # Order models by average of per-level averages for consistency
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df.groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    # Sort models by average score (best to worst) for violin plot
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    
    # Create violin plot with sorted order
    sns.violinplot(data=length_df, x='Model', y='Moves', order=sorted_models, ax=ax3)
    ax3.tick_params(axis='x', rotation=60, labelsize=8)
    ax3.set_title('Game Length Distribution')
    ax3.set_ylabel('Number of Moves')

# 4. Game length distribution (WINS ONLY) - NEW ANALYSIS
ax4 = plt.subplot(4, 4, 4)
won_game_lengths = []
for model, df in all_data.items():
    if 'number_of_moves' in df.columns:
        # Only include games where the model won (score = 1.0)
        won_games = df[df['score'] == 1.0]
        game_lengths = won_games['number_of_moves'].dropna()
        won_game_lengths.extend([(model.replace('_pt', '').replace('_', ' '), length) 
                                for length in game_lengths])

if won_game_lengths:
    won_length_df = pd.DataFrame(won_game_lengths, columns=['Model', 'Moves'])
    
    # Order models by average of per-level averages for consistency
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df.groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    # Sort models by average score (best to worst) for violin plot
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    
    # Create violin plot with sorted order
    sns.violinplot(data=won_length_df, x='Model', y='Moves', order=sorted_models, ax=ax4)
    ax4.tick_params(axis='x', rotation=60, labelsize=8)
    ax4.set_title('Game Length Distribution (WINS ONLY)')
    ax4.set_ylabel('Number of Moves')

# 5. Illegal move rates
ax5 = plt.subplot(4, 4, 5)
illegal_stats = []
for model, df in all_data.items():
    if 'player_one_illegal_moves' in df.columns:
        total_illegal = df['player_one_illegal_moves'].sum()
        total_games = len(df)
        avg_illegal = total_illegal / total_games if total_games > 0 else 0
        illegal_stats.append({
            'Model': model.replace('_pt', '').replace('_', ' '),
            'Avg Illegal Moves': avg_illegal,
            'Total Illegal': total_illegal
        })

if illegal_stats:
    illegal_df = pd.DataFrame(illegal_stats).sort_values('Avg Illegal Moves')
    bars = ax5.bar(range(len(illegal_df)), illegal_df['Avg Illegal Moves'],
                   color=plt.cm.Reds(np.linspace(0.3, 0.9, len(illegal_df))))
    ax5.set_xticks(range(len(illegal_df)))
    ax5.set_xticklabels(illegal_df['Model'], rotation=60, ha='right', fontsize=8)
    ax5.set_ylabel('Average Illegal Moves per Game')
    ax5.set_title('Illegal Move Rates')
    
    # Add values on bars
    for i, (avg, total) in enumerate(zip(illegal_df['Avg Illegal Moves'], illegal_df['Total Illegal'])):
        ax5.text(i, avg + 0.001, f'{avg:.3f}\n({int(total)})', ha='center', fontsize=9)

# 6. Score rate heatmap by model and stockfish level
ax6 = plt.subplot(4, 4, 6)
heatmap_data = []
for model, df in all_data.items():
    if 'stockfish_level' in df.columns:
        for level in range(10):  # Stockfish levels 0-9
            level_games = df[df['stockfish_level'] == level]
            if len(level_games) > 0:
                score_rate = level_games['score'].mean()  # Proper chess scoring with draws as 0.5
                heatmap_data.append({
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Stockfish Level': level,
                    'Score Rate': score_rate,
                    'Games': len(level_games)
                })

if heatmap_data:
    heatmap_df = pd.DataFrame(heatmap_data)
    pivot_df = heatmap_df.pivot(index='Model', columns='Stockfish Level', values='Score Rate')
    
    # Calculate average of per-level averages for fair ordering
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            # Calculate average score for each level, then average those
            level_means = df.groupby('stockfish_level')['score'].mean()
            # Only include levels where we have data
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            # Fallback to overall average if no level data
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    # Debug: Print model scores for verification
    print("\n=== HEATMAP ORDERING DEBUG ===")
    print("Ordering by: Average of per-level averages (fair comparison)")
    for model_name, avg_score in sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {avg_score:.4f}")
    print("="*30 + "\n")
    
    # Sort models by average score (best to worst)
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    
    # Check if all models in pivot_df are in sorted_models
    missing_models = set(pivot_df.index) - set(sorted_models)
    if missing_models:
        print(f"WARNING: Models in pivot table but not in sorted list: {missing_models}")
    
    # Only reorder models that exist in both
    models_to_reorder = [m for m in sorted_models if m in pivot_df.index]
    pivot_df = pivot_df.reindex(models_to_reorder)
    
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax6, 
                vmin=0, vmax=1, cbar_kws={'label': 'Score Rate (draws=0.5)'}, 
                annot_kws={'fontsize': 8})
    ax6.set_title('Score Rate Heatmap: Model vs Stockfish Level (ordered by avg performance)')

# 7. Sample count heatmap
ax7 = plt.subplot(4, 4, 7)
sample_count_data = []
for model, df in all_data.items():
    if 'stockfish_level' in df.columns:
        for level in range(10):  # Stockfish levels 0-9
            level_games = df[df['stockfish_level'] == level]
            sample_count_data.append({
                'Model': model.replace('_pt', '').replace('_', ' '),
                'Stockfish Level': level,
                'Sample Count': len(level_games)
            })

if sample_count_data:
    sample_count_df = pd.DataFrame(sample_count_data)
    sample_pivot_df = sample_count_df.pivot(index='Model', columns='Stockfish Level', values='Sample Count')
    
    # Order by average of per-level averages (same as main heatmap)
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df.groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    sample_pivot_df = sample_pivot_df.reindex(sorted_models)
    
    # Fill NaN values with 0 for models that haven't played certain levels
    sample_pivot_df = sample_pivot_df.fillna(0)
    
    sns.heatmap(sample_pivot_df, annot=True, fmt='.0f', cmap='Blues', ax=ax7, 
                cbar_kws={'label': 'Number of Games'}, annot_kws={'fontsize': 8})
    ax7.set_title('Sample Count: Games per Model vs Stockfish Level')

# 8. Win Rate by Game Length Bins - NEW ANALYSIS
ax8 = plt.subplot(4, 4, 8)
win_rate_by_length = []
for model, df in all_data.items():
    if 'number_of_moves' in df.columns:
        # Create game length bins
        df_clean = df.dropna(subset=['number_of_moves', 'score'])
        df_clean['length_bin'] = pd.cut(df_clean['number_of_moves'], 
                                       bins=[0, 20, 40, 60, 80, 100, float('inf')], 
                                       labels=['0-20', '21-40', '41-60', '61-80', '81-100', '100+'])
        
        for bin_name in ['0-20', '21-40', '41-60', '61-80', '81-100', '100+']:
            bin_games = df_clean[df_clean['length_bin'] == bin_name]
            if len(bin_games) > 0:
                win_rate = bin_games['score'].mean()
                win_rate_by_length.append({
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Length Bin': bin_name,
                    'Win Rate': win_rate,
                    'Games': len(bin_games)
                })

if win_rate_by_length:
    length_win_df = pd.DataFrame(win_rate_by_length)
    length_pivot = length_win_df.pivot(index='Model', columns='Length Bin', values='Win Rate')
    
    # Order by overall performance
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df.groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    length_pivot = length_pivot.reindex([m for m in sorted_models if m in length_pivot.index])
    
    sns.heatmap(length_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax8, 
                vmin=0, vmax=1, cbar_kws={'label': 'Win Rate'}, annot_kws={'fontsize': 8})
    ax8.set_title('Win Rate by Game Length')

# 9. Illegal Move Rate per Move - NEW ANALYSIS
ax9 = plt.subplot(4, 4, 9)
illegal_per_move_stats = []
for model, df in all_data.items():
    if 'player_one_illegal_moves' in df.columns and 'total_moves' in df.columns:
        df_clean = df.dropna(subset=['player_one_illegal_moves', 'total_moves'])
        df_clean = df_clean[df_clean['total_moves'] > 0]  # Avoid division by zero
        
        if len(df_clean) > 0:
            illegal_per_move = (df_clean['player_one_illegal_moves'] / df_clean['total_moves']).mean()
            illegal_per_move_stats.append({
                'Model': model.replace('_pt', '').replace('_', ' '),
                'Illegal Moves per Move': illegal_per_move
            })

if illegal_per_move_stats:
    illegal_per_move_df = pd.DataFrame(illegal_per_move_stats).sort_values('Illegal Moves per Move')
    bars = ax9.bar(range(len(illegal_per_move_df)), illegal_per_move_df['Illegal Moves per Move'],
                   color=plt.cm.Reds(np.linspace(0.3, 0.9, len(illegal_per_move_df))))
    ax9.set_xticks(range(len(illegal_per_move_df)))
    ax9.set_xticklabels(illegal_per_move_df['Model'], rotation=60, ha='right', fontsize=8)
    ax9.set_ylabel('Illegal Moves per Total Move')
    ax9.set_title('Illegal Move Rate (per move)')
    
    # Add values on bars
    for i, rate in enumerate(illegal_per_move_df['Illegal Moves per Move']):
        ax9.text(i, rate + 0.0001, f'{rate:.4f}', ha='center', fontsize=9)

# 10. Illegal Move Rate vs Move Number (Do models deteriorate over time?)
ax10 = plt.subplot(4, 4, 10)

# Create consistent model ordering based on performance
model_avg_scores = {}
for model, df in all_data.items():
    clean_model_name = model.replace('_pt', '').replace('_', ' ')
    if 'stockfish_level' in df.columns:
        level_means = df.groupby('stockfish_level')['score'].mean()
        model_avg_scores[clean_model_name] = level_means.mean()
    else:
        model_avg_scores[clean_model_name] = df['score'].mean()

# Sort models by average score (best to worst)
model_order = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)

# Map back to raw model names for data access
raw_model_order = []
for clean_name in model_order:
    for raw_name in all_data.keys():
        if raw_name.replace('_pt', '').replace('_', ' ') == clean_name:
            raw_model_order.append(raw_name)
            break

# Calculate illegal move rate per move number for top models
move_bins = list(range(10, 101, 10))  # 10-20, 20-30, ..., 90-100
colors = plt.cm.tab10(np.linspace(0, 1, len(raw_model_order)))

for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models for clarity
    if model_name in all_data:
        df = all_data[model_name]
        df_clean = df.dropna(subset=['number_of_moves', 'player_one_illegal_moves'])
        
        if len(df_clean) > 0:
            illegal_rates = []
            move_labels = []
            
            for j in range(len(move_bins)-1):
                start_move = move_bins[j]
                end_move = move_bins[j+1]
                
                games_in_range = df_clean[
                    (df_clean['number_of_moves'] >= start_move) & 
                    (df_clean['number_of_moves'] < end_move)
                ]
                
                if len(games_in_range) > 50:  # Only include bins with sufficient data
                    total_illegal = games_in_range['player_one_illegal_moves'].sum()
                    total_moves = games_in_range['number_of_moves'].sum()
                    illegal_rate = (total_illegal / total_moves * 100) if total_moves > 0 else 0
                    
                    illegal_rates.append(illegal_rate)
                    if i == 0:  # Only add labels once
                        move_labels.append(f"{start_move}-{end_move}")
            
            if illegal_rates:
                x_pos = range(len(illegal_rates))
                plt.plot(x_pos, illegal_rates, marker='o', label=model_name.replace('_', ' '), 
                        linewidth=2, color=colors[i])

plt.xlabel('Game Length Bins (moves)')
plt.ylabel('Illegal Move Rate (%)')
plt.title('Illegal Move Rate vs Game Length\n(Do models deteriorate in longer games?)')
plt.xticks(range(len(move_labels)), move_labels, rotation=45)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 11. Average Game Length vs Stockfish Level
ax11 = plt.subplot(4, 4, 11)

# Calculate average game length for each model at each Stockfish level
for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models
    if model_name in all_data:
        df = all_data[model_name]
        if 'stockfish_level' in df.columns and 'number_of_moves' in df.columns:
            df_clean = df.dropna(subset=['stockfish_level', 'number_of_moves'])
            
            if len(df_clean) > 0:
                level_stats = df_clean.groupby('stockfish_level')['number_of_moves'].agg(['mean', 'count']).reset_index()
                # Only include levels with sufficient games
                level_stats = level_stats[level_stats['count'] >= 50]
                
                if len(level_stats) > 0:
                    plt.plot(level_stats['stockfish_level'], level_stats['mean'], 
                            marker='o', label=model_name.replace('_', ' '), 
                            linewidth=2, color=colors[i])

plt.xlabel('Stockfish Level')
plt.ylabel('Average Game Length (moves)')
plt.title('Game Length vs Opponent Strength\n(Do stronger opponents lead to longer games?)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 12. Illegal Move Rate vs Stockfish Level
ax12 = plt.subplot(4, 4, 12)

# Calculate illegal move rate for each model at each Stockfish level
for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models
    if model_name in all_data:
        df = all_data[model_name]
        if 'stockfish_level' in df.columns and 'player_one_illegal_moves' in df.columns:
            df_clean = df.dropna(subset=['stockfish_level', 'player_one_illegal_moves', 'number_of_moves'])
            
            if len(df_clean) > 0:
                level_stats = df_clean.groupby('stockfish_level').agg({
                    'player_one_illegal_moves': 'sum',
                    'number_of_moves': 'sum',
                    'game_id': 'count'  # Count games
                }).reset_index()
                
                # Only include levels with sufficient games
                level_stats = level_stats[level_stats['game_id'] >= 50]
                
                if len(level_stats) > 0:
                    # Calculate illegal move rate per move
                    level_stats['illegal_rate'] = (level_stats['player_one_illegal_moves'] / 
                                                  level_stats['number_of_moves'] * 100)
                    
                    plt.plot(level_stats['stockfish_level'], level_stats['illegal_rate'], 
                            marker='o', label=model_name.replace('_', ' '), 
                            linewidth=2, color=colors[i])

plt.xlabel('Stockfish Level')
plt.ylabel('Illegal Move Rate (%)')
plt.title('Illegal Move Rate vs Opponent Strength\n(Do models make more mistakes vs stronger opponents?)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 13. Win Rate vs Average Game Length (Scatter Plot)
ax13 = plt.subplot(4, 4, 13)

scatter_data = []
for model_name in raw_model_order:
    if model_name in all_data:
        df = all_data[model_name]
        if 'number_of_moves' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['number_of_moves', 'score'])
            
            if len(df_clean) > 0:
                avg_game_length = df_clean['number_of_moves'].mean()
                win_rate = df_clean['score'].mean()
                scatter_data.append({
                    'Model': model_name.replace('_pt', '').replace('_', ' '),
                    'Avg Game Length': avg_game_length,
                    'Win Rate': win_rate
                })

if scatter_data:
    scatter_df = pd.DataFrame(scatter_data)
    plt.scatter(scatter_df['Avg Game Length'], scatter_df['Win Rate'], s=100, alpha=0.7)
    
    # Add model labels
    for _, row in scatter_df.iterrows():
        plt.annotate(row['Model'], (row['Avg Game Length'], row['Win Rate']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Average Game Length (moves)')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Average Game Length\n(Do longer games correlate with better/worse performance?)')
    plt.grid(True, alpha=0.3)

# 14. Win Rate vs Stockfish Level (Line Plot for Clarity)
ax14 = plt.subplot(4, 4, 14)

# Create a cleaner line plot showing win rate trends
for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models
    if model_name in all_data:
        df = all_data[model_name]
        if 'stockfish_level' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['stockfish_level', 'score'])
            
            if len(df_clean) > 0:
                level_stats = df_clean.groupby('stockfish_level')['score'].agg(['mean', 'count']).reset_index()
                # Only include levels with sufficient games
                level_stats = level_stats[level_stats['count'] >= 50]
                
                if len(level_stats) > 0:
                    plt.plot(level_stats['stockfish_level'], level_stats['mean'], 
                            marker='o', label=model_name.replace('_', ' '), 
                            linewidth=2, color=colors[i])

plt.xlabel('Stockfish Level')
plt.ylabel('Win Rate')
plt.title('Win Rate vs Stockfish Level\n(Performance degradation with opponent strength)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# 15. Performance Consistency (Standard Deviation across Stockfish levels)
ax15 = plt.subplot(4, 4, 15)

consistency_stats = []
for model_name in raw_model_order:
    if model_name in all_data:
        df = all_data[model_name]
        if 'stockfish_level' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['stockfish_level', 'score'])
            
            if len(df_clean) > 0:
                level_means = df_clean.groupby('stockfish_level')['score'].mean()
                # Only include levels with sufficient data
                level_means = level_means[level_means.index <= 9]  # Standard levels 0-9
                
                if len(level_means) >= 5:  # Need at least 5 levels for meaningful std
                    consistency = level_means.std()  # Lower std = more consistent
                    avg_performance = level_means.mean()
                    consistency_stats.append({
                        'Model': model_name.replace('_pt', '').replace('_', ' '),
                        'Performance Std': consistency,
                        'Avg Performance': avg_performance
                    })

if consistency_stats:
    consist_df = pd.DataFrame(consistency_stats).sort_values('Performance Std')
    bars = plt.bar(range(len(consist_df)), consist_df['Performance Std'],
                   color=plt.cm.Blues(np.linspace(0.3, 0.9, len(consist_df))))
    plt.xticks(range(len(consist_df)), consist_df['Model'], rotation=60, ha='right', fontsize=8)
    plt.ylabel('Performance Standard Deviation')
    plt.title('Performance Consistency\n(Lower = more consistent across Stockfish levels)')
    
    # Add values on bars
    for i, std in enumerate(consist_df['Performance Std']):
        plt.text(i, std + 0.005, f'{std:.3f}', ha='center', fontsize=9)

# 16. Early Game vs Late Game Performance Comparison
ax16 = plt.subplot(4, 4, 16)

early_late_stats = []
for model_name in raw_model_order:
    if model_name in all_data:
        df = all_data[model_name]
        if 'number_of_moves' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['number_of_moves', 'score'])
            
            if len(df_clean) > 0:
                early_games = df_clean[df_clean['number_of_moves'] <= 40]
                late_games = df_clean[df_clean['number_of_moves'] > 60]
                
                if len(early_games) > 50 and len(late_games) > 50:
                    early_win_rate = early_games['score'].mean()
                    late_win_rate = late_games['score'].mean()
                    performance_diff = late_win_rate - early_win_rate
                    
                    early_late_stats.append({
                        'Model': model_name.replace('_pt', '').replace('_', ' '),
                        'Early Win Rate': early_win_rate,
                        'Late Win Rate': late_win_rate,
                        'Performance Difference': performance_diff
                    })

if early_late_stats:
    early_late_df = pd.DataFrame(early_late_stats).sort_values('Performance Difference')
    
    # Create grouped bar chart
    x = range(len(early_late_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], early_late_df['Early Win Rate'], width, 
            label='Early Game (≤40 moves)', alpha=0.8, color='lightblue')
    plt.bar([i + width/2 for i in x], early_late_df['Late Win Rate'], width,
            label='Late Game (>60 moves)', alpha=0.8, color='darkblue')
    
    plt.xticks(x, early_late_df['Model'], rotation=60, ha='right', fontsize=8)
    plt.ylabel('Win Rate')
    plt.title('Early vs Late Game Performance\n(Do models perform differently in short vs long games?)')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chess_results_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("Note: 'Score Rate' includes draws as 0.5 points (proper chess scoring)")
print("Overall rates are misleading - see performance by Stockfish level!")
for model, df in all_data.items():
    print(f"\n{model}:")
    print(f"  Total games: {len(df)}")
    print(f"  Score rate (proper): {df['score'].mean():.1%}")
    print(f"  Pure win rate: {(df['score'] == 1.0).mean():.1%}")
    print(f"  Draw rate: {(df['score'] == 0.5).mean():.1%}")
    print(f"  Loss rate: {(df['score'] == 0.0).mean():.1%}")
    if 'number_of_moves' in df.columns:
        print(f"  Average game length: {df['number_of_moves'].mean():.1f} moves")
    if 'player_one_illegal_moves' in df.columns:
        print(f"  Total illegal moves: {df['player_one_illegal_moves'].sum()}")
    
    # Show performance by Stockfish level if available
    if 'stockfish_level' in df.columns:
        print(f"  Performance by Stockfish level:")
        level_stats = df.groupby('stockfish_level')['score'].agg(['mean', 'count']).reset_index()
        for _, row in level_stats.iterrows():
            level, score, count = row['stockfish_level'], row['mean'], row['count']
            print(f"    Level {level}: {score:.1%} ({count} games)") 