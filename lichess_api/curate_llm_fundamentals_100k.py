#!/usr/bin/env python3
"""
Curate 100K chess puzzles for LLM evaluation focusing on conceptual complexity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import argparse

class LLMFundamentalsCurator:
    """Curate chess puzzles based on LLM conceptual complexity"""
    
    def __init__(self, csv_file: str, target_count: int = 100000):
        self.csv_file = csv_file
        self.target_count = target_count
        self.df = None
        
        # Define theme categories by LLM conceptual complexity
        self.theme_categories = {
            "basic_chess_ideas": [
                # Material & Rules
                "advantage", "crushing", "hangingPiece", "promotion",
                # King Safety
                "exposedKing", "kingsideAttack", "queensideAttack", 
                # Simple Tactics
                "fork", "pin", "trappedPiece"
            ],
            "advanced_tactics": [
                # Complex Tactical Patterns
                "skewer", "discoveredAttack", "deflection", "attraction",
                # Planning & Positional
                "sacrifice", "defensiveMove", "quietMove",
                # Advanced Patterns
                "doubleCheck", "backRankMate"
            ]
        }
        
        # Flatten to get all target themes
        self.target_themes = set()
        for category_themes in self.theme_categories.values():
            self.target_themes.update(category_themes)
        
        # Exclude mate-in-n themes (separate dataset)
        self.exclude_themes = {
            "mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
            "smotheredMate", "anastasiaMate", "arabianMate", "hookMate", 
            "dovetailMate", "bodenMate", "doubleBishopMate", "vukovicMate", 
            "killBoxMate"
        }
        
    def load_and_filter_data(self):
        """Load and apply quality filters without heavy stratification"""
        print(f"📖 Loading puzzle dataset from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df):,} puzzles")
        
        print("\n🔍 Applying quality filters...")
        initial_count = len(self.df)
        
        # Quality filters (reasonable but not too restrictive)
        self.df = self.df[self.df['RatingDeviation'] <= 120]  # Slightly relaxed
        self.df = self.df[self.df['Popularity'] >= 60]        # Slightly relaxed
        self.df = self.df[self.df['NbPlays'] >= 100]          # Slightly relaxed
        self.df = self.df[self.df['GameUrl'].notna()]         # Valid URLs
        self.df = self.df[self.df['Themes'].notna()]          # Valid themes
        
        print(f"   After quality filters: {len(self.df):,} puzzles")
        
        # Exclude mate-focused puzzles
        mate_mask = self.df['Themes'].apply(
            lambda x: not any(mate_theme in str(x).split() for mate_theme in self.exclude_themes)
        )
        self.df = self.df[mate_mask]
        print(f"   After excluding mate puzzles: {len(self.df):,} puzzles")
        
        # Keep puzzles with target themes
        target_mask = self.df['Themes'].apply(
            lambda x: any(theme in str(x).split() for theme in self.target_themes)
        )
        self.df = self.df[target_mask]
        print(f"   After target theme filter: {len(self.df):,} puzzles")
        
        removed = initial_count - len(self.df)
        print(f"   🗑️  Removed {removed:,} puzzles ({removed/initial_count*100:.1f}%)")
        
        return self.df
    
    def create_natural_sample(self):
        """Create sample with natural distributions, light stratification"""
        print(f"\n⚖️  Creating {self.target_count:,} sample with natural distributions...")
        
        # Calculate quality score for all puzzles
        self.df['quality_score'] = (
            self.df['Popularity'] / 100 * 0.3 +
            np.log1p(self.df['NbPlays']) / np.log1p(self.df['NbPlays'].max()) * 0.3 +
            (120 - self.df['RatingDeviation']) / 120 * 0.2 +
            # Slight preference for broader rating range
            (1 - abs(self.df['Rating'] - self.df['Rating'].median()) / (self.df['Rating'].max() - self.df['Rating'].min())) * 0.2
        )
        
        # Light stratification to ensure both categories are represented
        basic_puzzles = self.df[self.df['Themes'].apply(
            lambda x: any(theme in str(x).split() for theme in self.theme_categories["basic_chess_ideas"])
        )].copy()
        
        advanced_puzzles = self.df[self.df['Themes'].apply(
            lambda x: any(theme in str(x).split() for theme in self.theme_categories["advanced_tactics"])
        )].copy()
        
        print(f"Available basic puzzles: {len(basic_puzzles):,}")
        print(f"Available advanced puzzles: {len(advanced_puzzles):,}")
        
        # Sample roughly 60% basic, 40% advanced (natural but ensures representation)
        basic_target = int(self.target_count * 0.6)
        advanced_target = self.target_count - basic_target
        
        # Sample top-quality puzzles from each category
        basic_sample = basic_puzzles.nlargest(min(basic_target, len(basic_puzzles)), 'quality_score')
        advanced_sample = advanced_puzzles.nlargest(min(advanced_target, len(advanced_puzzles)), 'quality_score')
        
        print(f"Sampled basic puzzles: {len(basic_sample):,}")
        print(f"Sampled advanced puzzles: {len(advanced_sample):,}")
        
        # Combine samples
        final_df = pd.concat([basic_sample, advanced_sample], ignore_index=True)
        
        # If we need more puzzles, fill from remaining high-quality puzzles
        if len(final_df) < self.target_count:
            remaining_needed = self.target_count - len(final_df)
            used_ids = set(final_df['PuzzleId'])
            remaining_puzzles = self.df[~self.df['PuzzleId'].isin(used_ids)]
            
            if len(remaining_puzzles) > 0:
                additional_sample = remaining_puzzles.nlargest(remaining_needed, 'quality_score')
                final_df = pd.concat([final_df, additional_sample], ignore_index=True)
                print(f"Added {len(additional_sample):,} additional high-quality puzzles")
        
        print(f"\n✅ Final sample: {len(final_df):,} puzzles")
        return final_df
    
    def analyze_natural_distributions(self, curated_df):
        """Analyze the natural distributions that emerged"""
        print(f"\n📊 Analyzing natural distributions...")
        
        # Game phase distribution
        print(f"\nGame phase distribution (natural):")
        for phase in ['opening', 'middlegame', 'endgame']:
            count = len(curated_df[curated_df['Themes'].str.contains(phase, na=False)])
            pct = count / len(curated_df) * 100
            print(f"  {phase.title():12}: {count:6,} ({pct:5.1f}%)")
        
        # Rating distribution
        print(f"\nRating distribution:")
        rating_bins = [(600, 1000), (1000, 1200), (1200, 1400), (1400, 1600), 
                      (1600, 1800), (1800, 2000), (2000, 2200), (2200, 3000)]
        for min_r, max_r in rating_bins:
            count = len(curated_df[(curated_df['Rating'] >= min_r) & (curated_df['Rating'] < max_r)])
            pct = count / len(curated_df) * 100
            print(f"  {min_r:4}-{max_r:4}: {count:6,} ({pct:5.1f}%)")
        
        # Theme category distribution
        print(f"\nTheme category distribution:")
        category_counts = {}
        for category, themes in self.theme_categories.items():
            count = 0
            for themes_str in curated_df['Themes'].dropna():
                if any(theme in str(themes_str).split() for theme in themes):
                    count += 1
            category_counts[category] = count
            pct = count / len(curated_df) * 100
            print(f"  {category.replace('_', ' ').title():20}: {count:6,} ({pct:5.1f}%)")
        
        # Individual theme counts
        print(f"\nTop individual themes:")
        theme_counts = Counter()
        for themes_str in curated_df['Themes'].dropna():
            themes = str(themes_str).split()
            for theme in themes:
                if theme in self.target_themes:
                    theme_counts[theme] += 1
        
        for theme, count in theme_counts.most_common(15):
            pct = count / len(curated_df) * 100
            print(f"  {theme:20}: {count:6,} ({pct:5.1f}%)")
        
        return curated_df
    
    def save_curated_set(self, curated_df, output_file: str):
        """Save the curated puzzle set with metadata"""
        print(f"\n💾 Saving LLM fundamentals set to {output_file}...")
        
        # Save puzzles
        curated_df.to_csv(output_file, index=False)
        
        # Create metadata
        metadata = {
            "total_puzzles": len(curated_df),
            "focus": "LLM conceptual complexity - basic chess ideas vs advanced tactics",
            "sampling_approach": "Natural distributions with light stratification",
            "theme_categories": self.theme_categories,
            "excluded_themes": list(self.exclude_themes),
            "quality_filters": {
                "rating_deviation_max": 120,
                "popularity_min": 60,
                "nb_plays_min": 100
            },
            "natural_distributions": {
                "game_phases": {},
                "rating_ranges": {},
                "theme_categories": {}
            },
            "quality_statistics": {
                "avg_rating": float(curated_df['Rating'].mean()),
                "median_rating": float(curated_df['Rating'].median()),
                "rating_std": float(curated_df['Rating'].std()),
                "avg_popularity": float(curated_df['Popularity'].mean()),
                "avg_nb_plays": float(curated_df['NbPlays'].mean())
            }
        }
        
        # Record natural distributions
        for phase in ['opening', 'middlegame', 'endgame']:
            count = len(curated_df[curated_df['Themes'].str.contains(phase, na=False)])
            pct = count / len(curated_df) * 100
            metadata["natural_distributions"]["game_phases"][phase] = {"count": count, "percentage": pct}
        
        for category, themes in self.theme_categories.items():
            count = 0
            for themes_str in curated_df['Themes'].dropna():
                if any(theme in str(themes_str).split() for theme in themes):
                    count += 1
            pct = count / len(curated_df) * 100
            metadata["natural_distributions"]["theme_categories"][category] = {"count": count, "percentage": pct}
        
        # Save metadata
        metadata_file = output_file.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved metadata to {metadata_file}")
        print(f"✅ Successfully curated {len(curated_df):,} LLM-focused puzzles!")
        
        return curated_df
    
    def curate(self, output_file: str):
        """Main curation workflow"""
        print(f"🎯 Curating {self.target_count:,} LLM-focused chess puzzles")
        print("=" * 80)
        
        # Load and filter data
        self.load_and_filter_data()
        
        # Create natural sample
        curated_df = self.create_natural_sample()
        
        # Analyze natural distributions
        curated_df = self.analyze_natural_distributions(curated_df)
        
        # Save results
        final_df = self.save_curated_set(curated_df, output_file)
        
        return final_df

def main():
    parser = argparse.ArgumentParser(description='Curate LLM-focused chess puzzles')
    parser.add_argument('--csv-file', default='data/lichess_db_puzzle.csv', 
                       help='Path to Lichess puzzle CSV file')
    parser.add_argument('--output', default='data/llm_fundamentals_100k.csv',
                       help='Output file for curated puzzles')
    parser.add_argument('--count', type=int, default=100000,
                       help='Number of puzzles to curate')
    
    args = parser.parse_args()
    
    # Create curator and run curation
    curator = LLMFundamentalsCurator(args.csv_file, args.count)
    curated_puzzles = curator.curate(args.output)
    
    print(f"\n🎉 Curation complete! Created LLM evaluation set with {len(curated_puzzles):,} puzzles")
    print(f"📁 Output: {args.output}")

if __name__ == "__main__":
    main()