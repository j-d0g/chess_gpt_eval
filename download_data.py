#!/usr/bin/env python3
"""
Download Chess GPT Evaluation Dataset from HuggingFace
=====================================================

Downloads the complete chess evaluation dataset from:
https://huggingface.co/datasets/jd0g/chess-language-model-evaluation

This includes:
- Game CSV files with PGN transcripts  
- Stockfish analysis results (JSON, CSV)
- Model performance summaries
- Visualization outputs

Usage:
    python download_data.py [--subset SUBSET] [--force]
    
Options:
    --subset: Download specific subset (games, analysis, all) [default: all]
    --force: Overwrite existing files
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import shutil

# Repository configuration
REPO_ID = "jd0g/chess-language-model-evaluation"
LOCAL_DATA_DIR = Path("data")

def download_subset(subset: str, force: bool = False):
    """Download specific subset of the dataset"""
    
    print(f"🚀 Downloading Chess GPT Evaluation Dataset")
    print(f"📊 Repository: {REPO_ID}")
    print(f"📁 Local directory: {LOCAL_DATA_DIR}")
    print(f"🎯 Subset: {subset}")
    print("=" * 50)
    
    # Create local data directory
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
    
    try:
        if subset == "all":
            print("📦 Downloading complete dataset...")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=LOCAL_DATA_DIR,
                repo_type="dataset",
                resume_download=True,
                local_dir_use_symlinks=False
            )
            
        elif subset == "games":
            print("🎮 Downloading game files...")
            # Download games directory
            games_files = [
                "games/large-16-600k_iters_pt_vs_stockfish_sweep.csv",
                "games/medium-16-600k_iters_pt_vs_stockfish_sweep.csv", 
                "games/medium-12-600k_iters_pt_vs_stockfish_sweep.csv",
                "games/small-16-600k_iters_pt_vs_stockfish_sweep.csv",
                "games/small-8-600k_iters_pt_vs_stockfish_sweep.csv",
                "games/adam_lichess_8layers_pt_vs_stockfish_sweep.csv",
                "games/adam_stockfish_8layers_pt_vs_stockfish_sweep.csv",
            ]
            
            (LOCAL_DATA_DIR / "games").mkdir(exist_ok=True)
            
            for file_path in games_files:
                try:
                    print(f"   Downloading {file_path}...")
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=file_path,
                        local_dir=LOCAL_DATA_DIR,
                        repo_type="dataset",
                        resume_download=True,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"   ⚠️  Could not download {file_path}: {e}")
            
        elif subset == "analysis":
            print("📊 Downloading analysis files...")
            # Download analysis directory 
            analysis_patterns = [
                "analysis/*_summary_*.csv",
                "analysis/*_moves_*.csv", 
                "analysis/*_detailed_*.json",
                "model_performance_summary.csv"
            ]
            
            (LOCAL_DATA_DIR / "analysis").mkdir(exist_ok=True)
            
            # Download model performance summary
            try:
                print("   Downloading model_performance_summary.csv...")
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename="model_performance_summary.csv",
                    local_dir=LOCAL_DATA_DIR,
                    repo_type="dataset",
                    resume_download=True,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"   ⚠️  Could not download model_performance_summary.csv: {e}")
                
            # For analysis files, we'll download the whole analysis folder
            try:
                print("   Downloading analysis directory...")
                snapshot_download(
                    repo_id=REPO_ID,
                    local_dir=LOCAL_DATA_DIR,
                    repo_type="dataset",
                    allow_patterns=["analysis/*"],
                    resume_download=True,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"   ⚠️  Could not download analysis directory: {e}")
        
        else:
            print(f"❌ Unknown subset: {subset}")
            return False
            
        print("\n✅ Download completed successfully!")
        
        # Show downloaded content
        print(f"\n📁 Downloaded content in {LOCAL_DATA_DIR}:")
        for item in sorted(LOCAL_DATA_DIR.rglob("*")):
            if item.is_file() and item.name != ".gitkeep":
                size = item.stat().st_size
                size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
                print(f"   {item.relative_to(LOCAL_DATA_DIR)} ({size_str})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("   Make sure you have internet connection and huggingface_hub installed:")
        print("   pip install huggingface_hub")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download Chess GPT Evaluation Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                    # Download complete dataset
  python download_data.py --subset games    # Download only game files
  python download_data.py --subset analysis # Download only analysis files
  python download_data.py --force           # Overwrite existing files
        """
    )
    
    parser.add_argument(
        "--subset",
        choices=["games", "analysis", "all"],
        default="all",
        help="Download specific subset of the dataset"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Check if data already exists
    if LOCAL_DATA_DIR.exists() and any(LOCAL_DATA_DIR.iterdir()) and not args.force:
        existing_files = [f for f in LOCAL_DATA_DIR.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        if existing_files:
            print(f"⚠️  Data directory already contains {len(existing_files)} files.")
            print("   Use --force to overwrite or --subset to download specific parts.")
            print("   Existing files:")
            for f in existing_files[:5]:  # Show first 5 files
                print(f"     {f.relative_to(LOCAL_DATA_DIR)}")
            if len(existing_files) > 5:
                print(f"     ... and {len(existing_files) - 5} more files")
            
            response = input("\n   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("   Download cancelled.")
                return
    
    # Download the dataset
    success = download_subset(args.subset, args.force)
    
    if success:
        print(f"\n🎉 Dataset ready! You can now run:")
        print(f"   python src/visualization/interactive_dashboard.py")
        print(f"   python src/analysis/mass_stockfish_processor.py --help")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 