#!/bin/env fish

# First sync all the files from the local directory to the server
echo "Syncing files to server..."
./sync_slurm.sh
echo "Done"

# Then submit a bunch of jobs
echo ""
echo "Submitting jobs..."

set jobfile "\$WORK/parallel-ode-filters/experiments/3_work_precision_diagram/slurmjob.sh"
echo "jobfile: $jobfile"

# set ivps logistic fhn lotkavolterra vdp0 rigidbody
set ivps fhn lotkavolterra henonheiles

for ivp in $ivps
    echo gpu-v100 $ivp
    ssh slurm "cd ~; sbatch -p gpu-v100 --gres=gpu:1 --exclude \"slurm-v100-6\" $jobfile $ivp"
end

for ivp in $ivps
    echo gpu-2080ti $ivp
    ssh slurm "cd ~; sbatch -p gpu-2080ti --gres=gpu:1 --exclude \"slurm-v100-6\" $jobfile $ivp"
end

for ivp in $ivps
    echo cpu-long $ivp
    ssh slurm "cd ~; sbatch -p cpu-long --exclude \"slurm-v100-6\" $jobfile \"$ivp --gpu-nocheck\""
end
echo "Done"
