# ODE Solver Benchmark

This is now the main experiment of the whole paper, and it might even be that all the plots can be generated from the data that this solver benchmark creates.

Structure is as follows:
- `./collect_data.py` runs the benchmark and saves the results in a `.csv` file.
  It includes many solvers (diffrax, EKS, parallel IEKS, sequential IEKS), a range of grid densities, and evaluates runtime, final error, and trajectory error.
  Run it from the root directory with
  ```sh
  python experiments/3_work_precisionaid_gram/collect-data.py [--save] [--gpu-nocheck] setupname
  ```
  where `setupname` refers to the setup, i.e. the IVP and the range of grid sizes, e.g. `fhn`, `logistic`, or `rigidbody`.
- Plotting code:
  The plotting is likely to change significantly now that the experiment got much more general.
  - `./plot_data.py` makes the old, quite informative but too verbose plots.
  - `./plot_wpds.py` makes a single work-precision diagram where IVPs are shown next to each other.
  - `./plot_workprecision.py`: **This is the current plotting script to make the paper figure!**
- `./slurmjob.sh`: The SLURM job file that is used to run the stuff on the cluster
- `./submit_jobs.sh`: Sync the local files with the cluster (using `../../sync_slurm.sh`) and then submit a range of jobs, covering multiple IVPs and multiple GPUs (and also a CPU setup), where each job essentially runs `./collect_data.py`.
