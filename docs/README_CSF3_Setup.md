# CSF3 Setup Guide - University of Manchester

This guide covers how to get started with the CSF3 (Computational Shared Facility) system at the University of Manchester, specifically for GPU-accelerated machine learning projects.

## System Overview

CSF3 is a high-performance computing cluster with the following GPU resources:
- **gpuV partition**: 68 NVIDIA V100 GPUs
- **gpuA partition**: 76 NVIDIA A100 80GB GPUs
- Maximum job time: 4 days
- Access via SSH: `ssh username@csf3.itservices.manchester.ac.uk`

## Getting GPU Access

### Interactive Sessions

For development and testing:

```bash
# Request 1-hour A100 GPU session
srun --partition=gpuA --gres=gpu:1 --time=1:00:00 --pty bash

# Request 12-hour A100 GPU session (for longer experiments)
srun --partition=gpuA --gres=gpu:1 --time=12:00:00 --pty bash

# Request V100 GPU (alternative)
srun --partition=gpuV --gres=gpu:1 --time=1:00:00 --pty bash
```

### Batch Jobs

Create a job script for unattended runs:

```bash
#!/bin/bash
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=my_experiment
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load required modules
module load cuda/12.6.2
module load miniforge/24.11.2

# Your commands here
cd /path/to/your/project
python your_script.py
```

Submit with: `sbatch your_job_script.sh`

## Environment Setup

### 1. Load Essential Modules

```bash
# CUDA for GPU support
module load cuda/12.6.2

# Python environment management
module load miniforge/24.11.2
```

### 2. Check GPU Availability

```bash
# Verify GPU access
nvidia-smi

# Check CUDA version
nvcc --version
```

### 3. Python Dependencies

```bash
# Upgrade pip first (important for some packages)
pip install --upgrade pip

# Install your requirements
pip install -r requirements.txt

# Or install individual packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Common Workflows

### Starting a New Session

```bash
# 1. SSH into CSF3
ssh your_username@csf3.itservices.manchester.ac.uk

# 2. Navigate to your scratch space (recommended for large files)
cd /mnt/iusers01/fse-ugpgt01/compsci01/your_username/scratch/

# 3. Request GPU
srun --partition=gpuA --gres=gpu:1 --time=12:00:00 --pty bash

# 4. Load modules
module load cuda/12.6.2 miniforge/24.11.2

# 5. Run your code
python your_script.py
```

### Monitoring Jobs

```bash
# Check your running jobs
squeue -u your_username

# Check detailed job info
scontrol show job JOBID

# Cancel a job
scancel JOBID
```

## File Management

### Storage Locations

- **Home directory**: `/home/your_username` (limited space, backed up)
- **Scratch space**: `/mnt/iusers01/fse-ugpgt01/compsci01/your_username/scratch/` (large space, not backed up)

### Best Practices

- Use scratch space for large datasets and model files
- Keep code in home directory (backed up)
- Clean up scratch space regularly
- Transfer results to local machine when done

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure you've loaded the correct modules in the right order
2. **CUDA errors**: Check that you're on a GPU node with `nvidia-smi`
3. **Out of memory**: Monitor GPU memory usage, consider smaller batch sizes
4. **Import errors**: Verify all dependencies are installed in the current environment
5. **File not found**: Check file paths - absolute paths often work better than relative

### Performance Tips

- **GPU Utilization**: Monitor with `nvidia-smi` to ensure GPU is being used
- **File I/O**: Use scratch space for better I/O performance
- **Dependencies**: Some packages (like tiktoken) may need Rust compiler - upgrade pip first
- **Batch Size**: Start small and increase gradually to find optimal GPU memory usage

### Debugging Commands

```bash
# Check current directory and files
pwd && ls -la

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check loaded modules
module list

# Check Python path and packages
which python
pip list | grep torch
```

## Example: Running Chess GPT Evaluation

This system was tested with a chess evaluation project. Key steps:

1. **Clone/download project** to scratch space
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Download external tools** (e.g., Stockfish engine)
4. **Configure paths** in your code to use absolute paths
5. **Load correct branch** if using git (`git checkout friends`)
6. **Verify model files** exist in expected locations
7. **Run with modules loaded**: `module load cuda/12.6.2 miniforge/24.11.2 && python main.py`

## Getting Help

- CSF3 Documentation: [IT Services website]
- Check job limits: `sinfo`
- System status: Check IT Services status page
- For technical issues: Contact IT Services helpdesk

## Notes

- Jobs are killed if they exceed time limits
- GPU nodes may have different architectures (check with `nvidia-smi`)
- Some software may need specific CUDA versions
- Always test with short interactive sessions before submitting long batch jobs 