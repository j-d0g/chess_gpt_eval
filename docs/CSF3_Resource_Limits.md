# CSF3 Resource Limits - Definitive Guide

## **Discovered Limits (Tested)**

### **GPU Allocation Limits**
- **Maximum GPUs per job**: 1 GPU only
- **Maximum CPUs per GPU**: 12 cores
- **Maximum memory per GPU**: ~128GB
- **Total GPU limit**: 76 GPUs across all jobs (plenty)

### **Optimal Single Job Configuration**
```bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=12 --mem=120G --time=4:00:00
```

### **QOS Constraints**
- `QOSMaxGRESPerUser`: Limits total concurrent GPU usage
- `gres/gpu=1`: Forces single GPU per job
- Memory scales with GPU count (128GB per GPU)

## **Maximum Throughput Strategy**

Since we can only get 1 GPU per job, the optimal strategy is **massive parallelization**:

### **Strategy: Array Jobs (RECOMMENDED)**
```bash
#!/bin/bash
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --array=1-20%5    # 20 jobs, max 5 running simultaneously
#SBATCH --job-name=chess_eval_array

# Each job runs independently
python chess_evaluation.py --job-id=$SLURM_ARRAY_TASK_ID --games=1000
```

### **Theoretical Maximum Throughput**
- **Per job**: 1 GPU + 12 CPUs + 120GB RAM
- **Concurrent jobs**: Limited by QOS (need to test, likely 2-5)
- **Total jobs**: 500 (your limit)

### **Performance Estimates**

#### **Conservative (2 concurrent jobs)**
- 2 × 1000 games = 2000 games per batch
- Time per batch: ~1 hour
- **Daily capacity**: ~48,000 games

#### **Optimistic (5 concurrent jobs)**
- 5 × 1000 games = 5000 games per batch  
- Time per batch: ~1 hour
- **Daily capacity**: ~120,000 games

#### **Maximum (if no concurrent limit)**
- Submit 20+ jobs in array
- Each completes 1000 games in ~1 hour
- **Potential**: 20,000+ games per hour

## **Practical Implementation**

### **1. Test Concurrent Limit**
```bash
# Submit multiple jobs to find concurrent limit
for i in {1..5}; do
    sbatch single_gpu_job.sh &
done
```

### **2. Optimize Within Single Job**
```python
# Use all 12 CPUs efficiently
import multiprocessing as mp

def run_stockfish_parallel(games_per_process):
    # Run multiple Stockfish instances
    pass

# 1 GPU for model inference
# 11 CPUs for parallel Stockfish evaluation
# 1 CPU for coordination
```

### **3. Array Job Template**
```bash
#!/bin/bash
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --array=1-50%10
#SBATCH --output=logs/chess_%A_%a.out
#SBATCH --error=logs/chess_%A_%a.err

# Load modules
module load cuda/12.6.2 miniforge/24.11.2

# Run evaluation with unique parameters per job
python main.py \
    --model="large-24-600K_iters.pt" \
    --games=1000 \
    --seed=$SLURM_ARRAY_TASK_ID \
    --output="logs/results_${SLURM_ARRAY_TASK_ID}.csv"
```

### **4. Multi-Model Evaluation**
```bash
# Evaluate multiple models in parallel
MODELS=("small-8-600k_iters.pt" "medium-16-600K_iters.pt" "large-24-600K_iters.pt")
STOCKFISH_LEVELS=(7 8 9)

for model in "${MODELS[@]}"; do
    for level in "${STOCKFISH_LEVELS[@]}"; do
        sbatch --export=MODEL=$model,LEVEL=$level evaluation_job.sh
    done
done
```

## **Resource Optimization Tips**

### **CPU Utilization**
- 1 CPU: Model inference coordination
- 1 CPU: File I/O and logging  
- 10 CPUs: Parallel Stockfish instances
- Use `multiprocessing` for CPU parallelization

### **Memory Management**
- Model: ~2-5GB (depending on size)
- Stockfish instances: ~100MB each
- Remaining: Game state caching, buffers
- 120GB is more than sufficient

### **GPU Optimization**
- Load model once, reuse for all games
- Use optimized inference (float16, compilation)
- Batch multiple positions if possible

## **Monitoring and Management**

### **Job Monitoring**
```bash
# Check array job status
squeue -u j74739jt --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Monitor specific array job
squeue -j 1234567 --array

# Check completed jobs
sacct -u j74739jt --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
```

### **Resource Usage**
```bash
# Check efficiency
seff $SLURM_JOB_ID

# Real-time monitoring
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveCPU,AvePages
```

## **Next Steps**

1. **Test concurrent job limit**: Submit 2-5 jobs simultaneously
2. **Benchmark single job performance**: Optimize CPU/GPU usage
3. **Scale with array jobs**: Submit large batches
4. **Monitor and adjust**: Based on actual performance

## **Expected Results**

With this strategy, you should achieve:
- **10-50x throughput** compared to single sequential jobs
- **Efficient resource utilization** (>80% CPU, >90% GPU)
- **Fault tolerance** (individual job failures don't affect others)
- **Scalable evaluation** (easily add more models/configurations) 