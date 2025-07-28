#!/usr/bin/env python3
"""
Generate Analysis Summary from Stockfish Data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def generate_summary():
    print("=" * 80)
    print("CHESS MODEL STOCKFISH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_dir = Path('data/analysis')
    
    # Load all summary files from June 26-27 (the complete analysis)
    model_data = {}
    total_games = 0
    
    for file in results_dir.glob('*_summary_202506*.csv'):
        if '20250626' in file.name or '20250627' in file.name:
            model_name = file.name.split('_vs_')[0]
            try:
                df = pd.read_csv(file)
                if 'average_centipawn_loss' in df.columns and len(df) > 0:
                    model_data[model_name] = {
                        'games': len(df),
                        'avg_cp_loss': df['average_centipawn_loss'].mean(),
                        'median_cp_loss': df['average_centipawn_loss'].median(),
                        'std_cp_loss': df['average_centipawn_loss'].std(),
                        'avg_blunders': df['blunders'].mean(),
                        'avg_mistakes': df['mistakes'].mean(),
                        'avg_inaccuracies': df['inaccuracies'].mean(),
                        'opening_accuracy': df['opening_accuracy'].mean(),
                        'middlegame_accuracy': df['middlegame_accuracy'].mean(),
                        'endgame_accuracy': df['endgame_accuracy'].mean(),
                        'avg_game_length': df['total_moves'].mean()
                    }
                    total_games += len(df)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
    
    print(f"Total models analyzed: {len(model_data)}")
    print(f"Total games analyzed: {total_games:,}")
    print()
    
    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame.from_dict(model_data, orient='index')
    stats_df['model'] = stats_df.index
    stats_df = stats_df.sort_values('avg_cp_loss')
    
    # 1. Best Performing Models
    print("TOP 10 BEST PERFORMING MODELS (by average centipawn loss)")
    print("-" * 60)
    for i, (model, row) in enumerate(stats_df.head(10).iterrows(), 1):
        clean_name = model.replace('_pt', '').replace('_', ' ')
        print(f"{i:2d}. {clean_name:40s} {row['avg_cp_loss']:6.1f} cp ({row['games']:,} games)")
    
    # 2. Model Size Analysis
    print("\n\nPERFORMANCE BY MODEL SIZE")
    print("-" * 60)
    for size in ['small', 'medium', 'large']:
        size_models = [m for m in stats_df.index if size in m]
        if size_models:
            size_data = stats_df.loc[size_models]
            print(f"{size.capitalize():8s}: {size_data['avg_cp_loss'].mean():6.1f} cp "
                  f"(σ={size_data['avg_cp_loss'].std():5.1f}, n={len(size_models)} models)")
    
    # 3. Training Dataset Comparison
    print("\n\nTRAINING DATASET COMPARISON")
    print("-" * 60)
    lichess_models = [m for m in stats_df.index if 'lichess' in m]
    stockfish_models = [m for m in stats_df.index if 'stockfish' in m and 'adam' in m]
    standard_models = [m for m in stats_df.index if m not in lichess_models + stockfish_models]
    
    for dataset, models in [('Lichess', lichess_models), 
                           ('Stockfish', stockfish_models), 
                           ('Standard', standard_models)]:
        if models:
            data = stats_df.loc[models]
            print(f"{dataset:10s}: {data['avg_cp_loss'].mean():6.1f} cp "
                  f"(best: {data['avg_cp_loss'].min():6.1f}, worst: {data['avg_cp_loss'].max():6.1f})")
    
    # 4. Phase Performance
    print("\n\nGAME PHASE SPECIALISTS")
    print("-" * 60)
    for phase in ['opening', 'middlegame', 'endgame']:
        col = f'{phase}_accuracy'
        best = stats_df.nlargest(1, col).iloc[0]
        print(f"{phase.capitalize():10s}: {best['model'].replace('_pt', ''):30s} "
              f"({best[col]:5.1f}% accuracy)")
    
    # 5. Consistency Analysis
    print("\n\nMOST CONSISTENT MODELS (lowest std deviation)")
    print("-" * 60)
    for i, (model, row) in enumerate(stats_df.nsmallest(5, 'std_cp_loss').iterrows(), 1):
        clean_name = model.replace('_pt', '').replace('_', ' ')
        print(f"{i}. {clean_name:40s} σ={row['std_cp_loss']:5.1f}")
    
    # 6. Key Statistics
    print("\n\nKEY STATISTICS")
    print("-" * 60)
    print(f"Best overall model: {stats_df.index[0].replace('_pt', '')}")
    print(f"  - Average CP loss: {stats_df.iloc[0]['avg_cp_loss']:.1f}")
    print(f"  - Blunder rate: {stats_df.iloc[0]['avg_blunders']:.2f} per game")
    print(f"  - Games analyzed: {stats_df.iloc[0]['games']:,}")
    
    # Performance spread
    print(f"\nPerformance spread:")
    print(f"  - Best model: {stats_df['avg_cp_loss'].min():.1f} cp")
    print(f"  - Worst model: {stats_df['avg_cp_loss'].max():.1f} cp")
    print(f"  - Difference: {stats_df['avg_cp_loss'].max() - stats_df['avg_cp_loss'].min():.1f} cp")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Chess Model Performance Analysis Summary', fontsize=16)
    
    # 1. Top models bar chart
    ax = axes[0, 0]
    top10 = stats_df.head(10)
    ax.barh(range(len(top10)), top10['avg_cp_loss'])
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([m.replace('_pt', '').replace('_', ' ') for m in top10.index])
    ax.set_xlabel('Average Centipawn Loss')
    ax.set_title('Top 10 Best Performing Models')
    ax.invert_yaxis()
    
    # 2. Size comparison
    ax = axes[0, 1]
    size_data = []
    for size in ['small', 'medium', 'large']:
        models = [m for m in stats_df.index if size in m]
        if models:
            size_data.append({
                'size': size,
                'avg': stats_df.loc[models]['avg_cp_loss'].mean(),
                'std': stats_df.loc[models]['avg_cp_loss'].std()
            })
    
    if size_data:
        sizes = [d['size'] for d in size_data]
        avgs = [d['avg'] for d in size_data]
        stds = [d['std'] for d in size_data]
        ax.bar(sizes, avgs, yerr=stds, capsize=5)
        ax.set_ylabel('Average Centipawn Loss')
        ax.set_title('Performance by Model Size')
    
    # 3. Phase performance scatter
    ax = axes[1, 0]
    ax.scatter(stats_df['opening_accuracy'], stats_df['endgame_accuracy'], 
              s=100, alpha=0.6, c=stats_df['avg_cp_loss'], cmap='RdYlGn_r')
    ax.set_xlabel('Opening Accuracy (%)')
    ax.set_ylabel('Endgame Accuracy (%)')
    ax.set_title('Opening vs Endgame Performance')
    
    # 4. Blunder rate vs CP loss
    ax = axes[1, 1]
    ax.scatter(stats_df['avg_blunders'], stats_df['avg_cp_loss'], s=100, alpha=0.6)
    ax.set_xlabel('Average Blunders per Game')
    ax.set_ylabel('Average Centipawn Loss')
    ax.set_title('Blunder Rate vs Performance')
    
    # Add best models annotation
    for i in range(min(3, len(stats_df))):
        model = stats_df.index[i]
        ax.annotate(model.split('-')[0], 
                   (stats_df.iloc[i]['avg_blunders'], stats_df.iloc[i]['avg_cp_loss']),
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('stockfish_analysis_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: stockfish_analysis_summary.png")
    
    # Save detailed results to CSV
    stats_df.to_csv('model_performance_summary.csv', index=False)
    print(f"Detailed results saved to: model_performance_summary.csv")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    generate_summary() 