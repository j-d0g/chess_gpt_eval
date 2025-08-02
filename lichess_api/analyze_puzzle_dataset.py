#!/usr/bin/env python3
"""
Analyze Lichess puzzle dataset metadata to help decide filtering criteria
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
from pathlib import Path

def analyze_puzzle_metadata(csv_file: str, sample_size: int = None):
    """Analyze puzzle metadata to understand the dataset"""
    
    print("🔍 Analyzing Lichess Puzzle Dataset Metadata")
    print("=" * 60)
    
    # Load data (sample if requested)
    print(f"📖 Loading puzzle data from {csv_file}...")
    if sample_size:
        print(f"   Taking random sample of {sample_size:,} puzzles...")
        df = pd.read_csv(csv_file).sample(n=sample_size, random_state=42)
    else:
        df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df):,} puzzles")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS")
    print("-" * 30)
    print(f"Total puzzles: {len(df):,}")
    print(f"Date range: Analysis focused on puzzle characteristics")
    
    # Rating analysis
    print(f"\n🎯 RATING DISTRIBUTION")
    print("-" * 30)
    ratings = df['Rating'].dropna()
    print(f"Rating range: {ratings.min()}-{ratings.max()}")
    print(f"Average rating: {ratings.mean():.0f}")
    print(f"Median rating: {ratings.median():.0f}")
    print(f"Standard deviation: {ratings.std():.0f}")
    
    # Rating distribution by ranges
    rating_ranges = [
        (0, 1000, "Beginner"),
        (1000, 1200, "Novice"),
        (1200, 1400, "Intermediate"),
        (1400, 1600, "Advanced"),
        (1600, 1800, "Expert"),
        (1800, 2000, "Master"),
        (2000, 2200, "Strong Master"),
        (2200, 2500, "Expert Master"),
        (2500, 3000, "Elite")
    ]
    
    print(f"\n📈 Rating Distribution by Skill Level:")
    for min_r, max_r, label in rating_ranges:
        count = len(df[(df['Rating'] >= min_r) & (df['Rating'] < max_r)])
        pct = count / len(df) * 100
        print(f"  {label:15} ({min_r:4}-{max_r:4}): {count:7,} ({pct:5.1f}%)")
    
    # Theme analysis
    print(f"\n🏷️  THEME ANALYSIS")
    print("-" * 30)
    
    # Parse themes (space-separated)
    all_themes = []
    for themes_str in df['Themes'].dropna():
        themes = str(themes_str).split()
        all_themes.extend(themes)
    
    theme_counts = Counter(all_themes)
    print(f"Total unique themes: {len(theme_counts)}")
    print(f"Most common themes:")
    
    for theme, count in theme_counts.most_common(20):
        pct = count / len(df) * 100
        print(f"  {theme:20}: {count:7,} ({pct:5.1f}%)")
    
    # Popularity analysis
    print(f"\n⭐ POPULARITY ANALYSIS")
    print("-" * 30)
    popularity = df['Popularity'].dropna()
    print(f"Popularity range: {popularity.min()}-{popularity.max()}")
    print(f"Average popularity: {popularity.mean():.1f}")
    print(f"High popularity (>80): {len(df[df['Popularity'] > 80]):,} ({len(df[df['Popularity'] > 80])/len(df)*100:.1f}%)")
    print(f"Medium popularity (50-80): {len(df[(df['Popularity'] >= 50) & (df['Popularity'] <= 80)]):,}")
    print(f"Low popularity (<50): {len(df[df['Popularity'] < 50]):,}")
    
    # Play count analysis
    print(f"\n🎮 PLAY COUNT ANALYSIS")
    print("-" * 30)
    nb_plays = df['NbPlays'].dropna()
    print(f"Play count range: {nb_plays.min():,}-{nb_plays.max():,}")
    print(f"Average plays: {nb_plays.mean():.0f}")
    print(f"Median plays: {nb_plays.median():.0f}")
    
    # Well-tested puzzles
    high_play_thresholds = [100, 500, 1000, 5000, 10000]
    for threshold in high_play_thresholds:
        count = len(df[df['NbPlays'] >= threshold])
        pct = count / len(df) * 100
        print(f"  Played ≥{threshold:5,} times: {count:7,} ({pct:5.1f}%)")

    # Rating deviation analysis
    print(f"\n📏 RATING RELIABILITY")
    print("-" * 30)
    rating_dev = df['RatingDeviation'].dropna()
    print(f"Rating deviation range: {rating_dev.min()}-{rating_dev.max()}")
    print(f"Average deviation: {rating_dev.mean():.1f}")
    
    # Reliable ratings (low deviation)
    reliable_counts = [
        (0, 75, "Very reliable"),
        (75, 100, "Reliable"), 
        (100, 150, "Somewhat reliable"),
        (150, 1000, "Less reliable")
    ]
    
    for min_dev, max_dev, label in reliable_counts:
        count = len(df[(df['RatingDeviation'] >= min_dev) & (df['RatingDeviation'] < max_dev)])
        pct = count / len(df) * 100
        print(f"  {label:18}: {count:7,} ({pct:5.1f}%)")

def recommend_filtering_criteria(csv_file: str):
    """Recommend filtering criteria based on dataset analysis"""
    
    print(f"\n💡 RECOMMENDED FILTERING STRATEGIES")
    print("=" * 60)
    
    df = pd.read_csv(csv_file)
    
    strategies = [
        {
            "name": "High-Quality Training Set",
            "description": "Well-tested, reliable puzzles for core training",
            "criteria": {
                "rating_min": 1200,
                "rating_max": 2200,  
                "popularity_min": 70,
                "nb_plays_min": 500,
                "rating_deviation_max": 100,
                "themes": ["middlegame", "endgame", "tactics"]
            }
        },
        {
            "name": "Beginner-Friendly Set", 
            "description": "Easier puzzles for foundational learning",
            "criteria": {
                "rating_min": 800,
                "rating_max": 1400,
                "popularity_min": 80,
                "nb_plays_min": 1000,
                "rating_deviation_max": 80,
                "themes": ["mate", "basic"]
            }
        },
        {
            "name": "Advanced Tactical Set",
            "description": "Complex puzzles for advanced training",
            "criteria": {
                "rating_min": 1800,
                "rating_max": 2800,
                "popularity_min": 60,
                "nb_plays_min": 100,
                "rating_deviation_max": 120,
                "themes": ["sacrifice", "attack", "defense"]
            }
        },
        {
            "name": "Comprehensive Balanced Set",
            "description": "Broad range covering all skill levels",
            "criteria": {
                "rating_min": 1000,
                "rating_max": 2400,
                "popularity_min": 50,
                "nb_plays_min": 200,
                "rating_deviation_max": 150,
                "themes": None  # All themes
            }
        }
    ]
    
    for strategy in strategies:
        criteria = strategy["criteria"]
        
        # Apply filters
        filtered = df.copy()
        filtered = filtered[
            (filtered['Rating'] >= criteria['rating_min']) &
            (filtered['Rating'] <= criteria['rating_max']) &
            (filtered['Popularity'] >= criteria['popularity_min']) &
            (filtered['NbPlays'] >= criteria['nb_plays_min']) &
            (filtered['RatingDeviation'] <= criteria['rating_deviation_max'])
        ]
        
        # Theme filtering (if specified)
        if criteria['themes']:
            theme_mask = filtered['Themes'].apply(
                lambda x: any(theme in str(x).lower() for theme in criteria['themes'])
            )
            filtered = filtered[theme_mask]
        
        count = len(filtered)
        pct = count / len(df) * 100
        
        print(f"\n🎯 {strategy['name']}")
        print(f"   {strategy['description']}")
        print(f"   Criteria: Rating {criteria['rating_min']}-{criteria['rating_max']}, "
              f"Popularity ≥{criteria['popularity_min']}, "
              f"Plays ≥{criteria['nb_plays_min']}")
        print(f"   Result: {count:,} puzzles ({pct:.1f}% of total)")
        
        if criteria['themes']:
            print(f"   Themes: {', '.join(criteria['themes'])}")
        
        # Estimate processing time (1 puzzle/sec)
        hours = count / 3600
        print(f"   Estimated PGN extraction time: {hours:.1f} hours")

def create_filtered_dataset(csv_file: str, output_file: str, **criteria):
    """Create a filtered subset of puzzles based on criteria"""
    
    print(f"\n🔽 Creating filtered dataset...")
    df = pd.read_csv(csv_file)
    
    # Apply filters
    filtered = df.copy()
    
    if 'rating_min' in criteria:
        filtered = filtered[filtered['Rating'] >= criteria['rating_min']]
    if 'rating_max' in criteria:
        filtered = filtered[filtered['Rating'] <= criteria['rating_max']]
    if 'popularity_min' in criteria:
        filtered = filtered[filtered['Popularity'] >= criteria['popularity_min']]
    if 'nb_plays_min' in criteria:
        filtered = filtered[filtered['NbPlays'] >= criteria['nb_plays_min']]
    if 'rating_deviation_max' in criteria:
        filtered = filtered[filtered['RatingDeviation'] <= criteria['rating_deviation_max']]
    
    # Theme filtering
    if 'themes' in criteria and criteria['themes']:
        theme_mask = filtered['Themes'].apply(
            lambda x: any(theme in str(x).lower() for theme in criteria['themes'])
        )
        filtered = filtered[theme_mask]
    
    # Save filtered dataset
    filtered.to_csv(output_file, index=False)
    
    print(f"✅ Filtered dataset saved: {output_file}")
    print(f"📊 Original: {len(df):,} puzzles → Filtered: {len(filtered):,} puzzles ({len(filtered)/len(df)*100:.1f}%)")
    
    return len(filtered)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Lichess puzzle dataset for filtering")
    parser.add_argument("--csv-file", default="data/lichess_puzzles/lichess_db_puzzle.csv", 
                       help="Path to CSV file")
    parser.add_argument("--sample", type=int, help="Sample size for quick analysis")
    parser.add_argument("--create-filter", help="Create filtered dataset with given name")
    parser.add_argument("--strategy", choices=["high-quality", "beginner", "advanced", "balanced"],
                       help="Use predefined filtering strategy")
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        print("Run download_lichess_puzzles.py --download-only first")
        return
    
    # Analyze dataset
    analyze_puzzle_metadata(args.csv_file, args.sample)
    
    # Show recommendations
    if not args.sample:  # Only for full dataset
        recommend_filtering_criteria(args.csv_file)
    
    # Create filtered dataset if requested
    if args.create_filter and args.strategy:
        strategies = {
            "high-quality": {
                "rating_min": 1200, "rating_max": 2200,
                "popularity_min": 70, "nb_plays_min": 500,
                "rating_deviation_max": 100,
                "themes": ["middlegame", "endgame", "tactics"]
            },
            "beginner": {
                "rating_min": 800, "rating_max": 1400,
                "popularity_min": 80, "nb_plays_min": 1000,
                "rating_deviation_max": 80
            },
            "advanced": {
                "rating_min": 1800, "rating_max": 2800,
                "popularity_min": 60, "nb_plays_min": 100,
                "rating_deviation_max": 120
            },
            "balanced": {
                "rating_min": 1000, "rating_max": 2400,
                "popularity_min": 50, "nb_plays_min": 200,
                "rating_deviation_max": 150
            }
        }
        
        criteria = strategies[args.strategy]
        output_file = f"data/lichess_puzzles/filtered_{args.strategy}_{args.create_filter}.csv"
        
        create_filtered_dataset(args.csv_file, output_file, **criteria)

if __name__ == "__main__":
    main()