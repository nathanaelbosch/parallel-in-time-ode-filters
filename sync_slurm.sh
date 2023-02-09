#!/usr/bin/env bash
rsync -rv ./ slurm:/mnt/qb/work/hennig/nbosch12/parallel-ode-filters \
      --exclude={".venv/","oldstuff/",".tox","experiments/old.2_init_comparison","__pycache__"} \
      --exclude={"*.csv","*.pdf","*.png"}
