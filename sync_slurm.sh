#!/usr/bin/env bash
rsync -r ./ slurm:/mnt/qb/work/hennig/nbosch12/parallel-ode-filters \
      --exclude={".venv/","oldstuff/",".tox","experiments/old.*","__pycache__","notebooks/.ipynb_checkpoints/",".git/",".pytest_cache"} \
      --exclude={"*.csv","*.pdf","*.png"}
