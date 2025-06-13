srun --partition=gpuA --gres=gpu:a100_80g:1 --time=12:00:00 --cpus-per-task=12 --mem=32G --pty bash
module load cuda/12.6.2 miniforge/24.11.2