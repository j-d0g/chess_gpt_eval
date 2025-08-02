#!/usr/bin/env python3
"""
Create processing partitions from the 100K fundamentals dataset:
1. Remove duplicates from 100K dataset
2. Remove puzzles already processed in 10K sample
3. Split remaining puzzles into two deterministic partitions
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
import json

class PartitionCreator:
    """Create deterministic processing partitions"""
    
    def __init__(self, full_dataset_path: str, processed_sample_path: str, output_dir: str = "data"):
        self.full_dataset_path = full_dataset_path
        self.processed_sample_path = processed_sample_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and basic validation of datasets"""
        print("📖 Loading datasets...")
        
        # Load full 100K dataset
        self.full_df = pd.read_csv(self.full_dataset_path)
        print(f"✅ Full dataset: {len(self.full_df):,} rows")
        
        # Load processed 10K sample
        self.processed_df = pd.read_csv(self.processed_sample_path)
        print(f"✅ Processed sample: {len(self.processed_df):,} rows")
        
        return self.full_df, self.processed_df
    
    def deduplicate_full_dataset(self):
        """Remove duplicates from full dataset, keeping first occurrence"""
        print("\n🔄 Deduplicating full dataset...")
        
        initial_count = len(self.full_df)
        
        # Keep first occurrence of each puzzle ID
        self.full_df_unique = self.full_df.drop_duplicates(subset=['PuzzleId'], keep='first')
        
        duplicates_removed = initial_count - len(self.full_df_unique)
        print(f"   Initial: {initial_count:,} rows")
        print(f"   Unique: {len(self.full_df_unique):,} rows")
        print(f"   Duplicates removed: {duplicates_removed:,}")
        
        return self.full_df_unique
    
    def remove_processed_puzzles(self):
        """Remove puzzles that were already processed in the 10K sample"""
        print("\n🔄 Removing already processed puzzles...")
        
        # Get processed puzzle IDs
        processed_ids = set(self.processed_df['PuzzleId'].unique())
        print(f"   Processed puzzle IDs: {len(processed_ids):,}")
        
        # Remove processed puzzles from unique dataset
        initial_count = len(self.full_df_unique)
        self.remaining_df = self.full_df_unique[~self.full_df_unique['PuzzleId'].isin(processed_ids)].copy()
        
        removed_count = initial_count - len(self.remaining_df)
        print(f"   Before removal: {initial_count:,} puzzles")
        print(f"   After removal: {len(self.remaining_df):,} puzzles")
        print(f"   Removed: {removed_count:,} puzzles")
        
        return self.remaining_df
    
    def create_deterministic_hash(self, puzzle_id: str) -> int:
        """Create deterministic hash from puzzle ID for consistent partitioning"""
        # Use MD5 hash of puzzle ID for deterministic but pseudo-random distribution
        hash_object = hashlib.md5(puzzle_id.encode())
        # Convert first 8 bytes to integer
        return int.from_bytes(hash_object.digest()[:8], byteorder='big')
    
    def split_into_partitions(self):
        """Split remaining puzzles into two deterministic partitions"""
        print("\n⚖️  Creating deterministic partitions...")
        
        # Add hash column for deterministic splitting
        self.remaining_df['partition_hash'] = self.remaining_df['PuzzleId'].apply(self.create_deterministic_hash)
        
        # Split based on hash (even/odd or modulo 2)
        partition_1_mask = self.remaining_df['partition_hash'] % 2 == 0
        
        self.partition_1 = self.remaining_df[partition_1_mask].copy()
        self.partition_2 = self.remaining_df[~partition_1_mask].copy()
        
        # Remove the hash column from final datasets
        self.partition_1 = self.partition_1.drop('partition_hash', axis=1)
        self.partition_2 = self.partition_2.drop('partition_hash', axis=1)
        
        print(f"   Partition 1: {len(self.partition_1):,} puzzles")
        print(f"   Partition 2: {len(self.partition_2):,} puzzles")
        print(f"   Difference: {abs(len(self.partition_1) - len(self.partition_2))} puzzles")
        
        # Verify no overlap
        p1_ids = set(self.partition_1['PuzzleId'])
        p2_ids = set(self.partition_2['PuzzleId'])
        overlap = p1_ids & p2_ids
        print(f"   Overlap check: {len(overlap)} puzzles (should be 0)")
        
        return self.partition_1, self.partition_2
    
    def analyze_partitions(self):
        """Analyze partition characteristics for balance verification"""
        print("\n📊 Partition analysis...")
        
        analyses = {}
        
        for partition_name, partition_df in [("Partition 1", self.partition_1), ("Partition 2", self.partition_2)]:
            print(f"\n--- {partition_name.upper()} ---")
            
            # Basic stats
            print(f"Count: {len(partition_df):,}")
            print(f"Rating - Mean: {partition_df['Rating'].mean():.1f}, Std: {partition_df['Rating'].std():.1f}")
            print(f"Popularity - Mean: {partition_df['Popularity'].mean():.1f}, Std: {partition_df['Popularity'].std():.1f}")
            
            if 'quality_score' in partition_df.columns:
                print(f"Quality - Mean: {partition_df['quality_score'].mean():.4f}, Std: {partition_df['quality_score'].std():.4f}")
            
            # Game phase distribution
            phase_counts = {}
            for phase in ['opening', 'middlegame', 'endgame']:
                count = partition_df['Themes'].str.contains(phase, na=False).sum()
                phase_counts[phase] = count
                print(f"{phase.capitalize()}: {count:,} ({count/len(partition_df)*100:.1f}%)")
            
            analyses[partition_name.lower().replace(' ', '_')] = {
                'count': int(len(partition_df)),
                'rating_mean': float(partition_df['Rating'].mean()),
                'popularity_mean': float(partition_df['Popularity'].mean()),
                'phase_distribution': {k: int(v) for k, v in phase_counts.items()}
            }
        
        return analyses
    
    def save_partitions(self):
        """Save partitions and metadata"""
        print("\n💾 Saving partitions and metadata...")
        
        # Save partitions
        partition_1_path = self.output_dir / "partition_1_unprocessed.csv"
        partition_2_path = self.output_dir / "partition_2_unprocessed.csv"
        
        self.partition_1.to_csv(partition_1_path, index=False)
        self.partition_2.to_csv(partition_2_path, index=False)
        
        print(f"✅ Partition 1: {partition_1_path}")
        print(f"✅ Partition 2: {partition_2_path}")
        
        # Save deduplicated full dataset
        full_unique_path = self.output_dir / "llm_fundamentals_100k_unique.csv"
        self.full_df_unique.to_csv(full_unique_path, index=False)
        print(f"✅ Deduplicated dataset: {full_unique_path}")
        
        # Save metadata
        metadata = {
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'original_dataset': str(self.full_dataset_path),
            'processed_sample': str(self.processed_sample_path),
            'original_count': int(len(self.full_df)),
            'unique_count': int(len(self.full_df_unique)),
            'processed_count': int(len(self.processed_df['PuzzleId'].unique())),
            'remaining_count': int(len(self.remaining_df)),
            'partition_1_count': int(len(self.partition_1)),
            'partition_2_count': int(len(self.partition_2)),
            'partitioning_method': 'md5_hash_modulo_2',
            'deterministic': True,
            'partition_analyses': self.analyze_partitions()
        }
        
        metadata_path = self.output_dir / "partition_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata: {metadata_path}")
        
        return partition_1_path, partition_2_path, metadata_path
    
    def run(self):
        """Main execution pipeline"""
        print("🚀 Creating processing partitions")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Process data
        self.deduplicate_full_dataset()
        self.remove_processed_puzzles()
        self.split_into_partitions()
        
        # Analyze and save
        self.analyze_partitions()
        partition_1_path, partition_2_path, metadata_path = self.save_partitions()
        
        print(f"\n🎉 Partitioning complete!")
        print(f"   Partition 1: {len(self.partition_1):,} puzzles → {partition_1_path}")
        print(f"   Partition 2: {len(self.partition_2):,} puzzles → {partition_2_path}")
        print(f"   Ready for parallel processing!")
        
        return partition_1_path, partition_2_path, metadata_path

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create processing partitions from fundamentals dataset')
    parser.add_argument('--full-dataset', default='data/llm_fundamentals_100k.csv', 
                       help='Path to full 100K dataset')
    parser.add_argument('--processed-sample', default='data/sample_10k_for_api.csv',
                       help='Path to already processed 10K sample')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for partitions')
    
    args = parser.parse_args()
    
    creator = PartitionCreator(
        full_dataset_path=args.full_dataset,
        processed_sample_path=args.processed_sample,
        output_dir=args.output_dir
    )
    
    creator.run()

if __name__ == "__main__":
    main()