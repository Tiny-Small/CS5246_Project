#!/bin/bash
#SBATCH --job-name=Test_01
#SBATCH --output=logs/Test_%j.log
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1

# Record the start time
echo "Job started at: $(date)"
start_time=$(date +%s)

echo "Activating virtual environment..."
source source cs5246_project/bin/activate || { echo "Failed to source venv"; exit 1; }
echo "venv activated."

# Confirm environment is active
python -c "import sys; print('Python version:', sys.version)"
python -c "print('Environment loaded successfully')"

# Run your script
python -m span_aste.main --mode train --data_path data/train.json --epochs 5 --batch_size 16 --max_span_len 23 --checkpoint_path checkpoints --log_filepath logs/train.log

echo "Job completed."

# Record the end time
echo "Job ended at: $(date)"
end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Job Duration: $(($duration / 3600)) hours $(($duration / 60 % 60)) minutes $(($duration % 60)) seconds"
