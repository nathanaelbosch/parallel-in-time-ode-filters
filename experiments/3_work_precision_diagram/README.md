# ODE Solver Benchmark

This is now the main experiment of the whole paper, and it might even be that all the plots can be generated from the data that this solver benchmark creates.

Structure is as follows:
- `./run_benchmark.py` runs the benchmark and saves the results in a `.csv` file.
  It includes many solvers (diffrax, EKS, parallel IEKS, sequential IEKS), a range of grid densities, and evaluates runtime, final error, and trajectory error.
  Run it from the root directory with
  ```sh
  python experiments/3_work_precisionaid_gram/run_benchmark.py [--save] [--gpu-nocheck] setupname
  ```
  where `setupname` refers to the setup, i.e. the IVP and the range of grid sizes, e.g. `fhn`, `logistic`, or `rigidbody`.
- Plotting code:
  - `./make_figure5.py` creates figure 5
  - `./make_figure6.py` creates figure 6
