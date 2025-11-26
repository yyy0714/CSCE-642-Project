#!/bin/bash

#SBATCH --export=HOME,PATH                  # Export HOME and PATH env var
#SBATCH --job-name=main.%j                  # Set the job name
#SBATCH --time=96:00:00                     # Set the wall clock limit to 96 hours
#SBATCH --output=main-%j.log                # Direct stdout and stderr to log
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks-per-node=1                 # Request 1 task per node
#SBATCH --cpus-per-task=8                   # Request 8 CPU cores per task (number can range from 1 to 144)
#SBATCH --mem-per-cpu=8G                    # Request 8 GB memory per CPU core
#SBATCH --partition=gpu-research            # Request GPU partition
#SBATCH --gres=gpu:tesla:1                  # Request GPUs (number can range from 1 to 4 for Tesla GPU and from 1 to 2 for A100 GPU)
#SBATCH --qos=olympus-research-gpu          # Enable QoS

source .venv/bin/activate

python3 main.py

deactivate
