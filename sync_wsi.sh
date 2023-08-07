#!/usr/bin/env bash
rsync -r ./ wsi:parallel-ode-filters \
      --exclude={".venv/","oldstuff/",".tox","experiments/old.*","__pycache__","notebooks/.ipynb_checkpoints/",".git/",".pytest_cache"} \
      --exclude={"*.csv","*.pdf","*.png"}
