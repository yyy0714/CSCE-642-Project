# One time setup
Since the code base uses Python 3.9.21, execute the following commands to create an appropriate Python virtual environment

```shell
srun --job-name=csce642-project --cpus-per-task=4 --partition=gpu-research --gres=gpu:tesla:1 --pty bash -l

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

exit
```

# Getting Started

## Step 1. Populating Design Files
Copy all the `design-*` directories from `1-sources/<DESIGN_NAME>` to `2-designs` where `DESIGN_NAME` is the name of the design (`2mm` by default).

## Step 2. Running the Program
Execute the following command to run the program for training and evaluating PPO agent.

```shell
sbatch ./job-scripts/job.sh
```

# Adding New Benchmarks
Follow the file structure in `1-sources` directory.
