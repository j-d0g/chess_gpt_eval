#!/usr/bin/env python3
"""
Curate the best 10,000 puzzles from the 5M Lichess puzzle dataset for LLM evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import argparse

class PuzzleCurator:
    """Intelligent puzzle curation for LLM evaluation"""
    
    def __init__(self, csv_file: str, target_count: int = 10000):
        self.csv_file = csv_file
        self.target_count = target_count
        self.df = None
        
    def load_data(self):
        """Load the puzzle dataset"""
        print(f"📖 Loading puzzle dataset from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df):,} puzzles")
        return self.df
    
    def apply_quality_filters(self):
        """Apply base quality filters to ensure reliability"""
        print("\n🔍 Applying quality filters...")
        
        initial_count = len(self.df)
        
        # Filter 1: Rating deviation (reliability)
        self.df = self.df[self.df['RatingDeviation'] <= 100]  # Reliable ratings
        print(f"   After rating reliability filter: {len(self.df):,} puzzles")
        
        # Filter 2: Popularity (community validation)
        self.df = self.df[self.df['Popularity'] >= 70]  # Well-liked puzzles
        print(f"   After popularity filter: {len(self.df):,} puzzles")
        
        # Filter 3: Play count (well-tested)
        self.df = self.df[self.df['NbPlays'] >= 100]  # Reasonably tested
        print(f"   After play count filter: {len(self.df):,} puzzles")
        
        # Filter 4: Valid game URL (for potential PGN extraction)
        self.df = self.df[self.df['GameUrl'].notna()]
        print(f"   After game URL filter: {len(self.df):,} puzzles")
        
        # Filter 5: Valid themes
        self.df = self.df[self.df['Themes'].notna()]
        print(f"   After themes filter: {len(self.df):,} puzzles")
        
        removed = initial_count - len(self.df)
        print(f"   🗑️  Removed {removed:,} low-quality puzzles ({removed/initial_count*100:.1f}%)")
        
        return self.df
    
    def create_balanced_difficulty_tiers(self):
        """Create balanced difficulty tiers for comprehensive evaluation"""
        print("\n⚖️  Creating balanced difficulty distribution...")
        
        # Define difficulty tiers with target distributions
        difficulty_tiers = [
            {"name": "Foundation", "rating_range": (800, 1200), "target_pct": 15, "themes": ["mate", "mateIn1", "mateIn2"]},
            {"name": "Beginner", "rating_range": (1200, 1400), "target_pct": 20, "themes": ["short", "endgame", "middlegame"]},
            {"name": "Intermediate", "rating_range": (1400, 1600), "target_pct": 25, "themes": ["tactics", "fork", "pin"]},
            {"name": "Advanced", "rating_range": (1600, 1800), "target_pct": 20, "themes": ["sacrifice", "attack", "defense"]},
            {"name": "Expert", "rating_range": (1800, 2000), "target_pct": 12, "themes": ["master", "complex"]},
            {"name": "Master", "rating_range": (2000, 2400), "target_pct": 8, "themes": ["master", "veryLong", "complex"]}
        ]
        
        curated_puzzles = []
        
        for tier in difficulty_tiers:
            tier_target = int(self.target_count * tier["target_pct"] / 100)
            min_rating, max_rating = tier["rating_range"]
            
            # Filter puzzles in this tier
            tier_puzzles = self.df[
                (self.df['Rating'] >= min_rating) & 
                (self.df['Rating'] < max_rating)
            ].copy()
            
            if len(tier_puzzles) == 0:
                print(f"   ⚠️  No puzzles found for {tier['name']} tier ({min_rating}-{max_rating})")
                continue
                
            # Prefer puzzles with target themes
            if tier["themes"]:
                theme_mask = tier_puzzles['Themes'].apply(
                    lambda x: any(theme.lower() in str(x).lower() for theme in tier["themes"])
                )
                themed_puzzles = tier_puzzles[theme_mask]
                
                if len(themed_puzzles) >= tier_target:
                    tier_puzzles = themed_puzzles
            
            # Sort by quality score (combination of popularity, play count, and rating stability)
            tier_puzzles['quality_score'] = (
                tier_puzzles['Popularity'] / 100 * 0.4 +
                np.log1p(tier_puzzles['NbPlays']) / np.log1p(tier_puzzles['NbPlays'].max()) * 0.4 +
                (100 - tier_puzzles['RatingDeviation']) / 100 * 0.2
            )
            
            # Select top puzzles for this tier
            selected = tier_puzzles.nlargest(tier_target, 'quality_score')
            curated_puzzles.append(selected)
            
            print(f"   {tier['name']:12} ({min_rating:4}-{max_rating:4}): {len(selected):4,} puzzles (target: {tier_target:4,})")
        
        # Combine all tiers
        final_df = pd.concat(curated_puzzles, ignore_index=True)
        print(f"\n✅ Curated {len(final_df):,} puzzles from {len(self.df):,} candidates")
        
        return final_df
    
    def add_theme_diversity(self, curated_df):
        """Ensure good theme diversity in the final set"""
        print("\n🎨 Analyzing theme diversity...")
        
        # Count themes in curated set
        all_themes = []
        for themes_str in curated_df['Themes'].dropna():
            themes = str(themes_str).split()
            all_themes.extend(themes)
        
        theme_counts = Counter(all_themes)
        print(f"   Found {len(theme_counts)} unique themes")
        
        # Show top themes
        print("   Top themes in curated set:")
        for theme, count in theme_counts.most_common(10):
            pct = count / len(curated_df) * 100
            print(f"     {theme:20}: {count:4,} ({pct:5.1f}%)")
        
        return curated_df
    
    def create_evaluation_metadata(self, curated_df):
        """Create metadata for the curated evaluation set"""
        print("\n📊 Creating evaluation metadata...")
        
        metadata = {
            "total_puzzles": len(curated_df),
            "source_dataset_size": len(pd.read_csv(self.csv_file)),
            "selection_criteria": {
                "rating_deviation_max": 100,
                "popularity_min": 70,
                "nb_plays_min": 100,
                "balanced_difficulty_tiers": True
            },
            "difficulty_distribution": {},
            "theme_distribution": {},
            "quality_statistics": {
                "avg_rating": float(curated_df['Rating'].mean()),
                "median_rating": float(curated_df['Rating'].median()),
                "rating_std": float(curated_df['Rating'].std()),
                "avg_popularity": float(curated_df['Popularity'].mean()),
                "avg_nb_plays": float(curated_df['NbPlays'].mean()),
                "avg_rating_deviation": float(curated_df['RatingDeviation'].mean())
            }
        }
        
        # Rating distribution
        rating_ranges = [(800, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2400)]
        for min_r, max_r in rating_ranges:
            count = len(curated_df[(curated_df['Rating'] >= min_r) & (curated_df['Rating'] < max_r)])
            pct = count / len(curated_df) * 100
            metadata["difficulty_distribution"][f"{min_r}-{max_r}"] = {"count": count, "percentage": pct}
        
        # Theme distribution
        all_themes = []
        for themes_str in curated_df['Themes'].dropna():
            themes = str(themes_str).split()
            all_themes.extend(themes)
        
        theme_counts = Counter(all_themes)
        for theme, count in theme_counts.most_common(15):
            pct = count / len(curated_df) * 100
            metadata["theme_distribution"][theme] = {"count": count, "percentage": pct}
        
        return metadata
    
    def save_curated_set(self, curated_df, output_file: str, metadata_file: str = None):
        """Save the curated puzzle set and metadata"""
        print(f"\n💾 Saving curated puzzle set to {output_file}...")
        
        # Save puzzles
        curated_df.to_csv(output_file, index=False)
        
        # Save metadata
        if metadata_file:
            metadata = self.create_evaluation_metadata(curated_df)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Saved metadata to {metadata_file}")
        
        print(f"✅ Successfully curated {len(curated_df):,} puzzles!")
        return curated_df
    
    def curate(self, output_file: str):
        """Main curation workflow"""
        print(f"🎯 Curating the best {self.target_count:,} puzzles for LLM evaluation")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Apply quality filters
        self.apply_quality_filters()
        
        # Create balanced selection
        curated_df = self.create_balanced_difficulty_tiers()
        
        # Analyze diversity
        curated_df = self.add_theme_diversity(curated_df)
        
        # Save results
        metadata_file = output_file.replace('.csv', '_metadata.json')
        final_df = self.save_curated_set(curated_df, output_file, metadata_file)
        
        return final_df

def main():
    parser = argparse.ArgumentParser(description='Curate best puzzles for LLM evaluation')
    parser.add_argument('--csv-file', default='data/lichess_db_puzzle.csv', 
                       help='Path to Lichess puzzle CSV file')
    parser.add_argument('--output', default='data/curated_10k_puzzles.csv',
                       help='Output file for curated puzzles')
    parser.add_argument('--count', type=int, default=10000,
                       help='Number of puzzles to curate')
    
    args = parser.parse_args()
    
    # Create curator and run curation
    curator = PuzzleCurator(args.csv_file, args.count)
    curated_puzzles = curator.curate(args.output)
    
    print(f"\n🎉 Curation complete! Created evaluation set with {len(curated_puzzles):,} puzzles")
    print(f"📁 Output: {args.output}")
    print(f"📁 Metadata: {args.output.replace('.csv', '_metadata.json')}")

if __name__ == "__main__":
    main()