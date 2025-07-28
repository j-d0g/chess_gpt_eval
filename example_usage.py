#!/usr/bin/env python3
"""
Example usage of the Chess Language Model Evaluation Dataset from Hugging Face
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns

def load_chess_dataset(repo_id: str):
    """
    Load the chess evaluation dataset from Hugging Face.
    
    Args:
        repo_id: Repository ID in format "username/dataset-name"
    
    Returns:
        dict: Dictionary containing loaded dataframes
    """
    
    print(f"Loading chess evaluation dataset from {repo_id}...")
    
    # Download main performance summary
    perf_file = hf_hub_download(
        repo_id=repo_id,
        filename="model_performance_summary.csv",
        repo_type="dataset"
    )
    performance_df = pd.read_csv(perf_file)
    
    # Download games summary (example)
    games_file = hf_hub_download(
        repo_id=repo_id,
        filename="data/games/games.csv",
        repo_type="dataset"
    )
    games_df = pd.read_csv(games_file)
    
    # Download analysis example
    analysis_file = hf_hub_download(
        repo_id=repo_id,
        filename="data/analysis/stockfish_analysis/games_summary_20250626_165502.csv",
        repo_type="dataset"
    )
    analysis_df = pd.read_csv(analysis_file)
    
    return {
        'performance': performance_df,
        'games': games_df,
        'analysis': analysis_df
    }

def analyze_model_performance(data: dict):
    """Analyze and visualize model performance."""
    
    perf_df = data['performance']
    
    print("=== Model Performance Summary ===")
    print(f"Total models evaluated: {len(perf_df)}")
    print(f"Total games played: {perf_df['games'].sum():,}")
    print(f"Average centipawn loss range: {perf_df['avg_cp_loss'].min():.1f} - {perf_df['avg_cp_loss'].max():.1f}")
    
    # Top 5 models by performance (lowest centipawn loss)
    print("\n=== Top 5 Models (Lowest Avg Centipawn Loss) ===")
    top_models = perf_df.nsmallest(5, 'avg_cp_loss')[['model', 'avg_cp_loss', 'games']]
    for idx, row in top_models.iterrows():
        print(f"{row['model']}: {row['avg_cp_loss']:.1f} CP loss ({row['games']} games)")
    
    # Create performance visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Centipawn loss vs Games
    plt.subplot(2, 2, 1)
    plt.scatter(perf_df['games'], perf_df['avg_cp_loss'], alpha=0.7)
    plt.xlabel('Number of Games')
    plt.ylabel('Average Centipawn Loss')
    plt.title('Model Performance vs Sample Size')
    
    # Plot 2: Blunders vs Mistakes
    plt.subplot(2, 2, 2)
    plt.scatter(perf_df['avg_blunders'], perf_df['avg_mistakes'], alpha=0.7)
    plt.xlabel('Average Blunders per Game')
    plt.ylabel('Average Mistakes per Game')
    plt.title('Blunders vs Mistakes')
    
    # Plot 3: Opening vs Endgame Accuracy
    plt.subplot(2, 2, 3)
    plt.scatter(perf_df['opening_accuracy'], perf_df['endgame_accuracy'], alpha=0.7)
    plt.xlabel('Opening Accuracy (%)')
    plt.ylabel('Endgame Accuracy (%)')
    plt.title('Opening vs Endgame Performance')
    
    # Plot 4: Model comparison bar chart
    plt.subplot(2, 2, 4)
    top_5 = perf_df.nsmallest(5, 'avg_cp_loss')
    plt.barh(range(len(top_5)), top_5['avg_cp_loss'])
    plt.yticks(range(len(top_5)), [m.replace('_', '\n') for m in top_5['model']])
    plt.xlabel('Average Centipawn Loss')
    plt.title('Top 5 Models Performance')
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_game_phases(data: dict):
    """Analyze performance across different game phases."""
    
    perf_df = data['performance']
    
    # Create phase comparison
    phases = ['opening_accuracy', 'middlegame_accuracy', 'endgame_accuracy']
    phase_data = perf_df[phases + ['model']].copy()
    
    # Melt for visualization
    phase_melted = phase_data.melt(
        id_vars=['model'], 
        value_vars=phases,
        var_name='phase', 
        value_name='accuracy'
    )
    phase_melted['phase'] = phase_melted['phase'].str.replace('_accuracy', '')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=phase_melted, x='phase', y='accuracy')
    plt.title('Model Performance Across Game Phases')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Game Phase')
    plt.savefig('phase_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Game Phase Analysis ===")
    for phase in phases:
        phase_name = phase.replace('_accuracy', '')
        avg_acc = perf_df[phase].mean()
        std_acc = perf_df[phase].std()
        print(f"{phase_name.capitalize()}: {avg_acc:.1f}% ± {std_acc:.1f}%")

def example_usage():
    """Example usage of the dataset."""
    
    # Replace with your actual repository ID
    repo_id = "your-username/chess-gpt-evaluation"
    
    print("Chess Language Model Evaluation Dataset - Example Usage")
    print("=" * 60)
    
    try:
        # Load dataset
        data = load_chess_dataset(repo_id)
        
        # Analyze performance
        analyze_model_performance(data)
        
        # Analyze game phases
        analyze_game_phases(data)
        
        print("\n=== Dataset Structure ===")
        for key, df in data.items():
            print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        
        print("\n✅ Analysis complete! Check the generated visualizations.")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nMake sure to:")
        print("1. Replace 'your-username/chess-gpt-evaluation' with the actual repo ID")
        print("2. Ensure the dataset has been uploaded to Hugging Face")
        print("3. Check your internet connection")

if __name__ == "__main__":
    example_usage() 