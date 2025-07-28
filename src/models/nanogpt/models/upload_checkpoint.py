#!/usr/bin/env python3
"""
Upload ChessGPT checkpoints to Hugging Face Hub.
Usage: python upload_checkpoint.py checkpoint_path model_name [repo_name]
Example: python upload_checkpoint.py out-stockfish-small-24/ckpt.pt small-24 chessgpt
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, upload_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def upload_checkpoint(checkpoint_path, model_name, repo_name="chessgpt", commit_message=None):
    """
    Upload a checkpoint to Hugging Face Hub
    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Name of the model (e.g., small-24, small-36)
        repo_name: Name of the Hugging Face repository
        commit_message: Optional commit message
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        return False
        
    # Get file size for reporting
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Uploading checkpoint ({file_size_mb:.2f} MB) to Hugging Face Hub...")
    
    # Create repo name with username prefix if needed
    if "/" not in repo_name:
        # Use environment variable for username or default to a placeholder
        username = os.environ.get("HF_USERNAME", "jd0g")
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = repo_name

    # Get token from environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found. Please set it in your .env file.")
        return False

    # Generate commit message if not provided
    if not commit_message:
        commit_message = f"Upload {model_name} checkpoint at iteration " + \
                        f"{os.environ.get('ITERATION', 'unknown')}"
    
    try:
        # Create API client and ensure repo exists
        api = HfApi(token=token)
        
        # First check if you're logged in
        try:
            username = api.whoami()["name"]
            print(f"Logged in as: {username}")
        except Exception as e:
            print(f"Not logged in to Hugging Face Hub. Please check your token.")
            print(f"Error: {e}")
            return False
        
        # Path in repo includes model name
        path_in_repo = f"checkpoints/{model_name}/ckpt.pt"
        
        # Upload the file
        uploaded_file = api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=path_in_repo,
            repo_id=full_repo_name,
            token=token,
            commit_message=commit_message,
        )
        
        print(f"Checkpoint uploaded successfully!")
        print(f"URL: {uploaded_file}")
        return True
        
    except Exception as e:
        print(f"Error uploading checkpoint: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ChessGPT checkpoints to Hugging Face Hub")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    parser.add_argument("model_name", help="Name of the model (e.g., small-24, small-36)")
    parser.add_argument("--repo", default="chessgpt", help="Hugging Face repository name")
    parser.add_argument("--message", help="Commit message")
    
    args = parser.parse_args()
    
    # Call the upload function
    success = upload_checkpoint(
        args.checkpoint_path, 
        args.model_name, 
        args.repo, 
        args.message
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 