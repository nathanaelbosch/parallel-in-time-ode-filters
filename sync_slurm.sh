#!/usr/bin/env bash 
rsync -rv ./ slurm:/mnt/qb/work/hennig/nbosch12/parallel-ode-filters --exclude ".venv/" --exclude "oldstuff/" --exclude ".tox" --exclude "experiments/old.2_init_comparison" --exclude "__pycache__"
