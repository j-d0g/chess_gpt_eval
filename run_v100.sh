srun --partition=gpuV --gres=gpu:v100:1 --cpus-per-task=8 --mem=40G --time=12:00:00 --pty bash
module load cuda/12.6.2 miniforge/24.11.2