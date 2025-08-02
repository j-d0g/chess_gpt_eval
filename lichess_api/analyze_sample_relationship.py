#!/usr/bin/env python3
"""
Analyze the relationship between the 100K fundamentals dataset and the 10K sample
to determine sampling method and identify remaining puzzles.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_sample_relationship():
    """Analyze the relationship between 100K and 10K datasets"""
    
    print("🔍 Analyzing 100K → 10K sample relationship")
    print("=" * 60)
    
    # Load both datasets
    print("📖 Loading datasets...")
    full_100k = pd.read_csv('data/llm_fundamentals_100k.csv')
    sample_10k = pd.read_csv('data/sample_10k_for_api.csv')
    
    print(f"✅ 100K dataset: {len(full_100k):,} puzzles")
    print(f"✅ 10K sample: {len(sample_10k):,} puzzles")
    
    # Check if all sample puzzles exist in the 100K dataset
    print("\n🔗 Checking puzzle ID relationships...")
    sample_ids = set(sample_10k['PuzzleId'])
    full_ids = set(full_100k['PuzzleId'])
    
    # Verify sample is subset of full dataset
    sample_in_full = sample_ids.issubset(full_ids)
    print(f"✅ All sample puzzles in 100K dataset: {sample_in_full}")
    
    if not sample_in_full:
        missing = sample_ids - full_ids
        print(f"⚠️  Missing from 100K: {len(missing)} puzzle IDs")
        return
    
    # Identify remaining puzzles
    remaining_ids = full_ids - sample_ids
    print(f"📊 Remaining puzzles in 100K: {len(remaining_ids):,}")
    
    # Analyze sampling pattern
    print("\n🎯 Analyzing sampling pattern...")
    
    # Check if sampling preserves original order
    full_100k['original_index'] = range(len(full_100k))
    sample_indices = full_100k[full_100k['PuzzleId'].isin(sample_ids)]['original_index'].values
    
    # Check if indices are sequential/sorted
    is_sorted = np.all(sample_indices[:-1] <= sample_indices[1:])
    is_sequential = np.all(np.diff(sample_indices) == 1)
    
    print(f"Sampled indices range: {sample_indices.min():,} to {sample_indices.max():,}")
    print(f"Indices are sorted: {is_sorted}")
    print(f"Indices are sequential: {is_sequential}")
    
    if is_sequential:
        print(f"📋 DETERMINISTIC: Sequential sample from rows {sample_indices.min()} to {sample_indices.max()}")
        method = "sequential"
    elif is_sorted:
        print("📋 DETERMINISTIC: Sorted selection (likely quality-based)")
        method = "sorted_selection"
    else:
        # Check if it might be random with a specific pattern
        gaps = np.diff(sorted(sample_indices))
        avg_gap = np.mean(gaps)
        print(f"Average gap between selected indices: {avg_gap:.2f}")
        
        if abs(avg_gap - 10) < 1:  # Every 10th approximately
            print("📋 PATTERN: Approximately every 10th puzzle selected")
            method = "systematic_sampling"
        else:
            print("📋 Pattern unclear - might be random or quality-based")
            method = "unknown"
    
    # Analyze quality score distribution if available
    if 'quality_score' in full_100k.columns:
        print("\n⭐ Quality score analysis...")
        sample_quality = full_100k[full_100k['PuzzleId'].isin(sample_ids)]['quality_score']
        remaining_quality = full_100k[full_100k['PuzzleId'].isin(remaining_ids)]['quality_score']
        
        print(f"Sample quality - Mean: {sample_quality.mean():.4f}, Std: {sample_quality.std():.4f}")
        print(f"Remaining quality - Mean: {remaining_quality.mean():.4f}, Std: {remaining_quality.std():.4f}")
        
        if sample_quality.mean() > remaining_quality.mean():
            print("✅ Sample has higher average quality → likely quality-based selection")
            if method == "unknown":
                method = "quality_based"
    
    # Create mapping files for tracking
    print("\n💾 Creating tracking files...")
    
    # Save sampled puzzle IDs
    sampled_df = pd.DataFrame({'PuzzleId': list(sample_ids)})
    sampled_df.to_csv('data/sampled_10k_puzzle_ids.csv', index=False)
    print(f"✅ Saved sampled puzzle IDs: data/sampled_10k_puzzle_ids.csv")
    
    # Save remaining puzzle IDs
    remaining_df = pd.DataFrame({'PuzzleId': list(remaining_ids)})
    remaining_df.to_csv('data/remaining_90k_puzzle_ids.csv', index=False)
    print(f"✅ Saved remaining puzzle IDs: data/remaining_90k_puzzle_ids.csv")
    
    # Save analysis summary
    summary = {
        'total_puzzles': len(full_100k),
        'sampled_puzzles': len(sample_10k),
        'remaining_puzzles': len(remaining_ids),
        'sampling_method': method,
        'is_deterministic': method in ['sequential', 'sorted_selection', 'systematic_sampling'],
        'sample_indices_min': int(sample_indices.min()),
        'sample_indices_max': int(sample_indices.max()),
        'indices_sorted': bool(is_sorted),
        'indices_sequential': bool(is_sequential)
    }
    
    if 'quality_score' in full_100k.columns:
        summary.update({
            'sample_quality_mean': float(sample_quality.mean()),
            'remaining_quality_mean': float(remaining_quality.mean()),
            'quality_based_selection': bool(sample_quality.mean() > remaining_quality.mean())
        })
    
    import json
    with open('data/sample_analysis_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved analysis report: data/sample_analysis_report.json")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   Sampling method: {method}")
    print(f"   Deterministic: {'Yes' if summary['is_deterministic'] else 'Unknown/Random'}")
    print(f"   Can track remaining: YES")
    
    return summary

if __name__ == "__main__":
    analyze_sample_relationship()