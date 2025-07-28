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
    'data/games/small-8-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/small-16-600k_iters_pt_vs_stockfish_sweep.csv', 
    'data/games/small-24-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/small-36-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/medium-12-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/medium-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/large-16-600k_iters_pt_vs_stockfish_sweep.csv',
    'data/games/adam_lichess_8layers_pt_vs_stockfish_sweep.csv',
    'data/games/adam_stockfish_8layers_pt_vs_stockfish_sweep.csv',
    'data/games/adam_lichess_16layers_pt_vs_stockfish_sweep.csv',
    'data/games/adam_stockfish_16layers_pt_vs_stockfish_sweep.csv'
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

# Check if any data was loaded
if not all_data:
    print("ERROR: No data files were found or loaded successfully!")
    print("Please check that the CSV files exist in the data/games/ directory.")
    exit(1)

print(f"\nSuccessfully loaded data from {len(all_data)} model(s)")

# Create figure with multiple subplots (4x4 layout for comprehensive analysis)
fig = plt.figure(figsize=(20, 24))

# 1. Performance vs Stockfish level - THE KEY METRIC
ax1 = plt.subplot(4, 4, 1)

# Calculate average of per-level averages for fair ordering
model_avg_scores = {}
for model, df in all_data.items():
    if 'stockfish_level' in df.columns:
        # Only consider levels 0-9 for average calculation
        level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
        model_avg_scores[model] = level_means.mean()
    else:
        model_avg_scores[model] = df['score'].mean()

# Use a colormap with distinct colors
colors = plt.cm.tab10(np.linspace(0, 1, len(model_avg_scores)))
color_idx = 0

# Plot in order of average score (best to worst)
for model in sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True):
    df = all_data[model]
    if 'stockfish_level' in df.columns:
        # Filter to only show levels 0-9
        level_stats = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].agg(['mean', 'count']).reset_index()
        ax1.plot(level_stats['stockfish_level'], level_stats['mean'], 
                marker='o', linewidth=2, markersize=8,
                label=model.replace('_pt', '').replace('_', ' '),
                color=colors[color_idx])
        color_idx += 1

ax1.set_xlabel('Stockfish Level')
ax1.set_ylabel('Average Score (including draws as 0.5)')
ax1.set_title('Performance vs Stockfish Level - THE KEY METRIC')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Set better axis scaling for clearer distinctions
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.1, 0.1))  # 0.1 increments
ax1.set_xticks(range(10))  # Every integer stockfish level 0-9
ax1.set_xticklabels([str(i) for i in range(10)])

# 2. Average score rate comparison (normalized win rate)
ax2 = plt.subplot(4, 4, 2)
avg_scores = []
for model, df in all_data.items():
    avg_score = df['score'].mean()
    avg_scores.append({
        'Model': model.replace('_pt', '').replace('_', ' '),
        'Avg Score': avg_score
    })

if not avg_scores:
    print("ERROR: No valid score data found!")
    exit(1)

scores_df = pd.DataFrame(avg_scores).sort_values('Avg Score')
bars = ax2.bar(range(len(scores_df)), scores_df['Avg Score'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(scores_df))))
ax2.set_xticks(range(len(scores_df)))
ax2.set_xticklabels(scores_df['Model'], rotation=60, ha='right', fontsize=8)
ax2.set_ylabel('Average Score Rate')
ax2.set_ylim(0, 1.0)
ax2.set_title('Average Score Rate')

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
            # Only consider levels 0-9 for consistent ordering
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
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
            # Only consider levels 0-9 for consistent ordering
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
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
            # Only consider levels 0-9 for consistent ordering
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
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
            # Only consider levels 0-9 for consistent ordering
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
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
        # Create game length bins with new structure
        df_clean = df.dropna(subset=['number_of_moves', 'score']).copy()
        df_clean['length_bin'] = pd.cut(df_clean['number_of_moves'], 
                                       bins=[0, 40, 50, 60, 70, 80, 90, 100], 
                                       labels=['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
        
        for bin_name in ['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']:
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
            # Only consider levels 0-9 for consistent ordering
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
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

# Create move labels first
move_labels = []
for j in range(len(move_bins)-1):
    start_move = move_bins[j]
    end_move = move_bins[j+1]
    move_labels.append(f"{start_move}-{end_move}")

for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models for clarity
    if model_name in all_data:
        df = all_data[model_name]
        df_clean = df.dropna(subset=['number_of_moves', 'player_one_illegal_moves'])
        
        if len(df_clean) > 0:
            illegal_rates = []
            
            for j in range(len(move_bins)-1):
                start_move = move_bins[j]
                end_move = move_bins[j+1]
                
                games_in_range = df_clean[
                    (df_clean['number_of_moves'] >= start_move) & 
                    (df_clean['number_of_moves'] < end_move)
                ]
                
                if len(games_in_range) > 30:  # Only include bins with sufficient data
                    total_illegal = games_in_range['player_one_illegal_moves'].sum()
                    total_moves = games_in_range['number_of_moves'].sum()
                    illegal_rate = (total_illegal / total_moves * 100) if total_moves > 0 else 0
                    
                    illegal_rates.append(illegal_rate)
            
            if illegal_rates and len(illegal_rates) == len(move_labels):
                x_pos = range(len(illegal_rates))
                ax10.plot(x_pos, illegal_rates, marker='o', label=model_name.replace('_', ' '), 
                        linewidth=2, color=colors[i])

ax10.set_xlabel('Game Length Bins (moves)')
ax10.set_ylabel('Illegal Moves per Game (%)')
ax10.set_title('Illegal Moves per Game vs Game Length\n(Do models make more illegal moves in longer games?)')
ax10.set_xticks(range(len(move_labels)))
ax10.set_xticklabels(move_labels, rotation=45)
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3)

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

# 12. Win Rate vs Game Length Line Graph (Clearer Bins)
ax12 = plt.subplot(4, 4, 12)

# Create clearer line graph with new bin structure
for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models
    if model_name in all_data:
        df = all_data[model_name]
        if 'number_of_moves' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['number_of_moves', 'score'])
            
            # Use new bin structure with clear labels (no 100+ since no games reach that)
            bins = [0, 40, 50, 60, 70, 80, 90, 100]
            bin_centers = [35, 45, 55, 65, 75, 85, 95]
            bin_labels = ['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
            
            df_clean = df_clean.copy()
            df_clean['length_bin'] = pd.cut(df_clean['number_of_moves'], bins=bins, labels=bin_labels)
            
            win_rates = []
            valid_centers = []
            valid_labels = []
            
            for j, (bin_label, center) in enumerate(zip(bin_labels, bin_centers)):
                bin_games = df_clean[df_clean['length_bin'] == bin_label]
                if len(bin_games) > 30:  # Only include bins with sufficient data
                    win_rate = bin_games['score'].mean()
                    win_rates.append(win_rate)
                    valid_centers.append(center)
                    valid_labels.append(bin_label)
            
            if win_rates:
                ax12.plot(valid_centers, win_rates, marker='o', 
                         label=model_name.replace('_pt', '').replace('_', ' '), 
                         linewidth=2, color=colors[i], markersize=6)

ax12.set_xlabel('Game Length Bins')
ax12.set_ylabel('Win Rate')
ax12.set_title('Win Rate vs Game Length Bins\n(Performance across different game durations)')
ax12.legend(fontsize=8)
ax12.grid(True, alpha=0.3)
ax12.set_ylim(0, 1)

# Set x-axis labels for the bins
bin_labels_display = ['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
bin_centers_display = [35, 45, 55, 65, 75, 85, 95]
ax12.set_xticks(bin_centers_display)
ax12.set_xticklabels(bin_labels_display, rotation=45)

# 13. Reverse Cumulative Win Rate Line Graph (>X moves)
ax13 = plt.subplot(4, 4, 13)

# Create reverse cumulative line graph
for i, model_name in enumerate(raw_model_order[:6]):  # Top 6 models
    if model_name in all_data:
        df = all_data[model_name]
        if 'number_of_moves' in df.columns and 'score' in df.columns:
            df_clean = df.dropna(subset=['number_of_moves', 'score'])
            
            # Create reverse cumulative data points
            reverse_thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            win_rates = []
            valid_thresholds = []
            
            for threshold in reverse_thresholds:
                games_over_threshold = df_clean[df_clean['number_of_moves'] > threshold]
                if len(games_over_threshold) > 50:  # Only include with sufficient data
                    win_rate = games_over_threshold['score'].mean()
                    win_rates.append(win_rate)
                    valid_thresholds.append(threshold)
            
            if len(win_rates) > 2:  # Need at least 3 points for a meaningful line
                ax13.plot(valid_thresholds, win_rates, marker='o', 
                         label=model_name.replace('_pt', '').replace('_', ' '), 
                         linewidth=2, color=colors[i], markersize=6)

ax13.set_xlabel('Minimum Game Length (>X moves)')
ax13.set_ylabel('Win Rate')
ax13.set_title('Win Rate in Games >X Moves\n(Performance in extended games)')
ax13.legend(fontsize=8)
ax13.grid(True, alpha=0.3)
ax13.set_ylim(0, 1)

# 14. Reverse Cumulative Win Rate (>X moves) - NEW VISUALIZATION
ax14 = plt.subplot(4, 4, 14)

reverse_cumulative_win_rate = []
for model, df in all_data.items():
    if 'number_of_moves' in df.columns and 'score' in df.columns:
        df_clean = df.dropna(subset=['number_of_moves', 'score'])
        
        # Create reverse cumulative bins (games longer than X moves)
        reverse_bins = [0, 20, 40, 60, 80, 100]
        
        for min_moves in reverse_bins:
            games_over_threshold = df_clean[df_clean['number_of_moves'] > min_moves]
            if len(games_over_threshold) > 0:
                win_rate = games_over_threshold['score'].mean()
                reverse_cumulative_win_rate.append({
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Min Moves': f'>{min_moves}',
                    'Win Rate': win_rate,
                    'Games': len(games_over_threshold)
                })

if reverse_cumulative_win_rate:
    reverse_df = pd.DataFrame(reverse_cumulative_win_rate)
    reverse_pivot = reverse_df.pivot(index='Model', columns='Min Moves', values='Win Rate')
    
    # Order by overall performance (using same ordering as other heatmaps)
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    reverse_pivot = reverse_pivot.reindex([m for m in sorted_models if m in reverse_pivot.index])
    
    # Reorder columns to show in logical order (move >100 to end)
    col_order = ['>0', '>20', '>40', '>60', '>80', '>100']
    reverse_pivot = reverse_pivot[[col for col in col_order if col in reverse_pivot.columns]]
    
    sns.heatmap(reverse_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax14, 
                vmin=0, vmax=1, cbar_kws={'label': 'Win Rate'}, annot_kws={'fontsize': 8})
    ax14.set_title('Win Rate by Game Length (>X moves)\n(Performance in games longer than X moves)')

# 15. Win Rate by Game Length (Cumulative) - NEW VISUALIZATION
ax15 = plt.subplot(4, 4, 15)
cumulative_win_rate_by_length = []

for model, df in all_data.items():
    if 'number_of_moves' in df.columns and 'score' in df.columns:
        df_clean = df.dropna(subset=['number_of_moves', 'score'])
        
        # Create cumulative bins in correct order
        cumulative_bins = [40, 50, 60, 70, 80, 90, 100]
         
        for max_moves in cumulative_bins:
            games_under_threshold = df_clean[df_clean['number_of_moves'] < max_moves]
            if len(games_under_threshold) > 0:
                win_rate = games_under_threshold['score'].mean()
                cumulative_win_rate_by_length.append({
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Max Moves': f'<{max_moves}',
                    'Win Rate': win_rate,
                    'Games': len(games_under_threshold)
                })

if cumulative_win_rate_by_length:
    cumulative_df = pd.DataFrame(cumulative_win_rate_by_length)
    cumulative_pivot = cumulative_df.pivot(index='Model', columns='Max Moves', values='Win Rate')
    
    # Order by overall performance (using same ordering as other heatmaps)
    model_avg_scores = {}
    for model, df in all_data.items():
        clean_model_name = model.replace('_pt', '').replace('_', ' ')
        if 'stockfish_level' in df.columns:
            level_means = df[df['stockfish_level'] <= 9].groupby('stockfish_level')['score'].mean()
            model_avg_scores[clean_model_name] = level_means.mean()
        else:
            model_avg_scores[clean_model_name] = df['score'].mean()
    
    sorted_models = sorted(model_avg_scores.keys(), key=lambda x: model_avg_scores[x], reverse=True)
    cumulative_pivot = cumulative_pivot.reindex([m for m in sorted_models if m in cumulative_pivot.index])
    
    sns.heatmap(cumulative_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax15, 
                vmin=0, vmax=1, cbar_kws={'label': 'Win Rate'}, annot_kws={'fontsize': 8})
    ax15.set_title('Win Rate by Game Length (Cumulative)')

# 16. Model Performance Scatter Plot (Fixed)
ax16 = plt.subplot(4, 4, 16)

scatter_data = []
for i, model_name in enumerate(raw_model_order):
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
                    'Win Rate': win_rate,
                    'Color': colors[i % len(colors)]
                })

if scatter_data:
    scatter_df = pd.DataFrame(scatter_data)
    
    # Use distinct colors for each model
    for i, row in scatter_df.iterrows():
        ax16.scatter(row['Avg Game Length'], row['Win Rate'], 
                    s=150, c=[row['Color']], alpha=0.8, edgecolors='black', linewidth=1,
                    label=row['Model'])
    
    # Add simple model labels without overlap issues
    for _, row in scatter_df.iterrows():
        ax16.annotate(row['Model'], (row['Avg Game Length'], row['Win Rate']), 
                     xytext=(3, 3), textcoords='offset points', fontsize=7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax16.set_xlabel('Average Game Length (moves)')
    ax16.set_ylabel('Win Rate')
    ax16.set_title('Model Performance Overview\n(Win Rate vs Game Duration)')
    ax16.grid(True, alpha=0.3)
    ax16.set_ylim(0, 1)
    
    # No legend needed since labels are on points
    # ax16.legend(fontsize=6, loc='upper right')

plt.tight_layout()
plt.savefig('chess_results_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a comprehensive summary statistics table
print("\n" + "="*120)
print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY TABLE")
print("="*120)

# Prepare summary data for table
summary_rows = []
for model, df in all_data.items():
    model_stats = {
        'Model': model.replace('_pt', '').replace('_', ' '),
        'Total Games': len(df),
        'Avg Score': f"{df['score'].mean():.3f}",
        'Win Rate': f"{(df['score'] == 1.0).mean():.3f}",
        'Draw Rate': f"{(df['score'] == 0.5).mean():.3f}",
        'Loss Rate': f"{(df['score'] == 0.0).mean():.3f}",
        'Avg Game Length': f"{df['number_of_moves'].mean():.1f}" if 'number_of_moves' in df.columns else 'N/A',
        'Illegal Moves/Game': f"{(df['player_one_illegal_moves'].sum() / len(df)):.2f}" if 'player_one_illegal_moves' in df.columns else 'N/A'
    }
    
    # Add performance vs different Stockfish levels
    if 'stockfish_level' in df.columns:
        for level in [0, 3, 6, 9]:
            level_games = df[df['stockfish_level'] == level]
            if len(level_games) > 0:
                model_stats[f'vs SF{level}'] = f"{level_games['score'].mean():.3f}"
            else:
                model_stats[f'vs SF{level}'] = 'N/A'
    
    summary_rows.append(model_stats)

# Sort by average score (best to worst)
summary_rows.sort(key=lambda x: float(x['Avg Score']), reverse=True)

# Print table header
headers = list(summary_rows[0].keys())
col_widths = {}
for header in headers:
    col_widths[header] = max(len(header), max(len(str(row.get(header, ''))) for row in summary_rows))

# Print header row
header_row = ' | '.join(f"{header:<{col_widths[header]}}" for header in headers)
print(header_row)
print('-' * len(header_row))

# Print data rows
for row in summary_rows:
    data_row = ' | '.join(f"{str(row.get(header, '')):<{col_widths[header]}}" for header in headers)
    print(data_row)

print("\n" + "="*120)

# Print additional insights
print("\nKEY INSIGHTS:")
print("-" * 50)

# Best performing model
best_model = summary_rows[0]
print(f"1. Best Overall Model: {best_model['Model']} (Avg Score: {best_model['Avg Score']})")

# Model with fewest illegal moves
if any('Illegal Moves/Game' in row and row['Illegal Moves/Game'] != 'N/A' for row in summary_rows):
    cleanest_model = min([row for row in summary_rows if row['Illegal Moves/Game'] != 'N/A'], 
                        key=lambda x: float(x['Illegal Moves/Game']))
    print(f"2. Cleanest Play: {cleanest_model['Model']} ({cleanest_model['Illegal Moves/Game']} illegal moves/game)")

# Performance drop analysis
if 'vs SF0' in summary_rows[0] and 'vs SF9' in summary_rows[0]:
    for row in summary_rows[:3]:  # Top 3 models
        if row['vs SF0'] != 'N/A' and row['vs SF9'] != 'N/A':
            drop = float(row['vs SF0']) - float(row['vs SF9'])
            print(f"3. {row['Model']} performance drop: {drop:.3f} (SF0: {row['vs SF0']} → SF9: {row['vs SF9']})")

print("\n" + "="*120)

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