#!/bin/bash

#SBATCH --export=HOME,PATH                  # Export HOME and PATH env var
#SBATCH --job-name=vitis-9.%j               # Set the job name
#SBATCH --time=01:00:00                     # Set the wall clock limit to 1 hour
#SBATCH --output=vitis-9.log                # Direct stdout and stderr to log
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks-per-node=1                 # Request 1 task per node
#SBATCH --cpus-per-task=8                   # Request 8 CPU cores per task
#SBATCH --mem-per-cpu=8G                    # Request 8 GB memory per CPU core
#SBATCH --partition=cpu-research            # Request the CPU research partition/queue
#SBATCH --qos=olympus-cpu-research          # Request the CPU research QoS

source /opt/coe/Xilinx/Vitis_HLS/2023.1/settings64.sh

cd ./1-designs/design-9

vitis_hls vitis.tcl
