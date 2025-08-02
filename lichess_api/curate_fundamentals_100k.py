#!/usr/bin/env python3
"""
Curate 100K fundamental chess puzzles focusing on basics, tactics, and planning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import argparse

class FundamentalsPuzzleCurator:
    """Curate fundamental chess puzzles for LLM evaluation baseline"""
    
    def __init__(self, csv_file: str, target_count: int = 100000):
        self.csv_file = csv_file
        self.target_count = target_count
        self.df = None
        
        # Define fundamental theme categories
        self.fundamental_themes = {
            "basic_material": ["advantage", "crushing", "hangingPiece"],
            "core_tactics": ["fork", "pin", "skewer", "discoveredAttack", "deflection", "attraction"],
            "attack_defend": ["kingsideAttack", "queensideAttack", "defensiveMove", "exposedKing"],
            "planning_lookahead": ["short", "long", "veryLong", "quietMove"],
            "positional_tempo": ["sacrifice", "clearance", "intermezzo", "trappedPiece"],
            "advanced_tactics": ["xRayAttack", "interference", "capturingDefender", "doubleCheck"]
        }
        
        # Flatten to get all desired themes
        self.target_themes = set()
        for category_themes in self.fundamental_themes.values():
            self.target_themes.update(category_themes)
        
        # Exclude mate-focused themes (already have mate dataset)
        self.exclude_themes = {
            "mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
            "backRankMate", "smotheredMate", "anastasiaMate", "arabianMate", 
            "hookMate", "dovetailMate", "bodenMate", "doubleBishopMate", 
            "vukovicMate", "killBoxMate"
        }
        
    def load_and_filter_data(self):
        """Load and apply initial quality filters"""
        print(f"📖 Loading puzzle dataset from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df):,} puzzles")
        
        print("\n🔍 Applying quality and relevance filters...")
        initial_count = len(self.df)
        
        # Quality filters
        self.df = self.df[self.df['RatingDeviation'] <= 100]  # Reliable ratings
        self.df = self.df[self.df['Popularity'] >= 70]        # Well-liked
        self.df = self.df[self.df['NbPlays'] >= 200]          # Reasonably tested
        self.df = self.df[self.df['GameUrl'].notna()]         # Valid game URLs
        self.df = self.df[self.df['Themes'].notna()]          # Valid themes
        
        print(f"   After quality filters: {len(self.df):,} puzzles")
        
        # Filter out mate-focused puzzles (already covered)
        mate_mask = self.df['Themes'].apply(
            lambda x: not any(mate_theme in str(x).split() for mate_theme in self.exclude_themes)
        )
        self.df = self.df[mate_mask]
        print(f"   After excluding mate puzzles: {len(self.df):,} puzzles")
        
        # Keep only puzzles with fundamental themes
        fundamental_mask = self.df['Themes'].apply(
            lambda x: any(theme in str(x).split() for theme in self.target_themes)
        )
        self.df = self.df[fundamental_mask]
        print(f"   After fundamental theme filter: {len(self.df):,} puzzles")
        
        removed = initial_count - len(self.df)
        print(f"   🗑️  Removed {removed:,} puzzles ({removed/initial_count*100:.1f}%)")
        
        return self.df
    
    def create_balanced_sample(self):
        """Create balanced sample across game phases, difficulty, and theme categories"""
        print(f"\n⚖️  Creating balanced 100K sample...")
        
        # Define sampling structure
        game_phases = ['opening', 'middlegame', 'endgame']
        
        # Split equally across game phases (33.3% each)
        phase_target = self.target_count // 3
        
        # Define difficulty tiers within each phase
        difficulty_tiers = [
            (800, 1200, "Beginner"),
            (1200, 1400, "Intermediate"), 
            (1400, 1600, "Advanced"),
            (1600, 1800, "Expert"),
            (1800, 2200, "Master")
        ]
        
        sampled_puzzles = []
        
        for phase in game_phases:
            print(f"\n--- {phase.upper()} PHASE ---")
            
            # Get puzzles for this phase
            phase_puzzles = self.df[self.df['Themes'].str.contains(phase, na=False)].copy()
            print(f"Available {phase} puzzles: {len(phase_puzzles):,}")
            
            if len(phase_puzzles) == 0:
                print(f"⚠️  No {phase} puzzles available!")
                continue
            
            # Sample across difficulty tiers
            tier_target = phase_target // len(difficulty_tiers)
            
            for min_rating, max_rating, tier_name in difficulty_tiers:
                tier_puzzles = phase_puzzles[
                    (phase_puzzles['Rating'] >= min_rating) & 
                    (phase_puzzles['Rating'] < max_rating)
                ].copy()
                
                if len(tier_puzzles) == 0:
                    print(f"   {tier_name}: No puzzles available")
                    continue
                
                # Calculate quality score for ranking
                tier_puzzles['quality_score'] = (
                    tier_puzzles['Popularity'] / 100 * 0.4 +
                    np.log1p(tier_puzzles['NbPlays']) / np.log1p(tier_puzzles['NbPlays'].max()) * 0.4 +
                    (100 - tier_puzzles['RatingDeviation']) / 100 * 0.2
                )
                
                # Sample best puzzles for this tier
                sample_size = min(tier_target, len(tier_puzzles))
                tier_sample = tier_puzzles.nlargest(sample_size, 'quality_score')
                sampled_puzzles.append(tier_sample)
                
                print(f"   {tier_name} ({min_rating}-{max_rating}): {len(tier_sample):,} puzzles")
        
        # Combine all samples
        final_df = pd.concat(sampled_puzzles, ignore_index=True)
        print(f"\n✅ Sampled {len(final_df):,} fundamental puzzles")
        
        return final_df
    
    def analyze_theme_distribution(self, curated_df):
        """Analyze the theme distribution in the curated set"""
        print(f"\n📊 Analyzing theme distribution...")
        
        # Count themes by category
        category_counts = {category: 0 for category in self.fundamental_themes.keys()}
        theme_counts = Counter()
        
        for themes_str in curated_df['Themes'].dropna():
            themes = str(themes_str).split()
            for theme in themes:
                theme_counts[theme] += 1
                
                # Categorize theme
                for category, category_themes in self.fundamental_themes.items():
                    if theme in category_themes:
                        category_counts[category] += 1
        
        print(f"\nTheme category distribution:")
        for category, count in category_counts.items():
            pct = count / len(curated_df) * 100
            print(f"  {category.replace('_', ' ').title():20}: {count:6,} ({pct:5.1f}%)")
        
        print(f"\nTop individual themes:")
        for theme, count in theme_counts.most_common(15):
            if theme in self.target_themes:
                pct = count / len(curated_df) * 100
                print(f"  {theme:20}: {count:6,} ({pct:5.1f}%)")
        
        return curated_df
    
    def save_curated_set(self, curated_df, output_file: str):
        """Save the curated puzzle set with metadata"""
        print(f"\n💾 Saving curated fundamentals set to {output_file}...")
        
        # Save puzzles
        curated_df.to_csv(output_file, index=False)
        
        # Create metadata
        metadata = {
            "total_puzzles": len(curated_df),
            "focus": "Fundamental chess tactics and planning",
            "selection_criteria": {
                "excluded_themes": list(self.exclude_themes),
                "included_categories": list(self.fundamental_themes.keys()),
                "quality_filters": {
                    "rating_deviation_max": 100,
                    "popularity_min": 70,
                    "nb_plays_min": 200
                }
            },
            "game_phase_distribution": {},
            "difficulty_distribution": {},
            "quality_statistics": {
                "avg_rating": float(curated_df['Rating'].mean()),
                "median_rating": float(curated_df['Rating'].median()),
                "rating_std": float(curated_df['Rating'].std()),
                "avg_popularity": float(curated_df['Popularity'].mean()),
                "avg_nb_plays": float(curated_df['NbPlays'].mean())
            }
        }
        
        # Game phase distribution
        for phase in ['opening', 'middlegame', 'endgame']:
            count = len(curated_df[curated_df['Themes'].str.contains(phase, na=False)])
            pct = count / len(curated_df) * 100
            metadata["game_phase_distribution"][phase] = {"count": count, "percentage": pct}
        
        # Save metadata
        metadata_file = output_file.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved metadata to {metadata_file}")
        print(f"✅ Successfully curated {len(curated_df):,} fundamental puzzles!")
        
        return curated_df
    
    def curate(self, output_file: str):
        """Main curation workflow"""
        print(f"🎯 Curating {self.target_count:,} fundamental chess puzzles")
        print("=" * 80)
        
        # Load and filter data
        self.load_and_filter_data()
        
        # Create balanced sample
        curated_df = self.create_balanced_sample()
        
        # Analyze theme distribution
        curated_df = self.analyze_theme_distribution(curated_df)
        
        # Save results
        final_df = self.save_curated_set(curated_df, output_file)
        
        return final_df

def main():
    parser = argparse.ArgumentParser(description='Curate fundamental chess puzzles for LLM evaluation')
    parser.add_argument('--csv-file', default='data/lichess_db_puzzle.csv', 
                       help='Path to Lichess puzzle CSV file')
    parser.add_argument('--output', default='data/fundamentals_100k_puzzles.csv',
                       help='Output file for curated puzzles')
    parser.add_argument('--count', type=int, default=100000,
                       help='Number of puzzles to curate')
    
    args = parser.parse_args()
    
    # Create curator and run curation
    curator = FundamentalsPuzzleCurator(args.csv_file, args.count)
    curated_puzzles = curator.curate(args.output)
    
    print(f"\n🎉 Curation complete! Created fundamentals evaluation set with {len(curated_puzzles):,} puzzles")
    print(f"📁 Output: {args.output}")

if __name__ == "__main__":
    main()