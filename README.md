# One time setup
Since the code base uses Python 3.9.21, execute the following commands to create an appropriate Python virtual environment

```shell
srun --job-name=csce642-project --cpus-per-task=4 --partition=gpu-research --gres=gpu:tesla:1 --pty bash -l

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements

exit
```

# Run the program
Execute the following command to run the program

```shell
sbatch ./job-scripts/job.sh
```

# Add New Benchmark
Follow the file structure in `1-sources` directory. The default design (used in `2-designs`) is `2mm`.
