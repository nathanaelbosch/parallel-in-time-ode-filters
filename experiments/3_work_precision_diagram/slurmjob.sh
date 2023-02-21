#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-24:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=nathanael.bosch@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

# insert your commands here
cd /mnt/qb/work/hennig/nbosch12/parallel-ode-filters
.venv/bin/python experiments/4_work_precision_diagram/collect_data.py $1 --save
