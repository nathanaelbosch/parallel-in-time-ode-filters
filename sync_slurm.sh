#!/usr/bin/env bash
rsync -rv ./ slurm:/mnt/qb/work/hennig/nbosch12/parallel-ode-filters \
      --exclude={".venv/","oldstuff/",".tox","experiments/old.*","__pycache__"} \
      --exclude={"*.csv","*.pdf","*.png"}
