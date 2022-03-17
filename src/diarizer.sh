#!/bin/bash
#SBATCH --job-name=diarizer
#SBATCH --output=outLogs/diarizer_voice_proximity_%j.out
#SBATCH --error=outLogs/diarizer_voice_proximity_%j.err
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task=48
#SBATCH --time=94:00:00
#SBATCH --partition=gablab
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1

conda activate diarizer
python diarizer.py /scratch/scratch/Sat/rb_iv_0320 /scratch/scratch/Sat/rb_diarized_0305