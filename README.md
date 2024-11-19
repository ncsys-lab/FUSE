# FUSE: A tool for expressing (Encoded) Energy Functions
This is the implementation for the tool described in Efficient Optimization with Encoded Energy Functions (HPCA 2025, link pending). It allows one to define problems and various energy function mappings for those problems, along with a p-computing simulator.

## Installation
If using [Docker](https://www.docker.com), you can spin up a container with all the dependencies installed after cloning the repo (may need to `chmod +x` the files):
```
./build_image.sh
./run_image.sh
```
This will drop you in a bash instance ready to run FUSE.

Alternatively, you can also use a virtual environment and install dependencies via  `pip install -r requirements.txt`. We have tested FUSE on Python3.11.

## Replicating Experiments
You can use `scripts/run_{prob}_exps.sh` to run the encoded and conventional energy function experiments for a given problem. We report the runtimes for these experiments on a consumer-grade laptop CPU. We also report Quick runtimes, where simulations exit early if a solution equivalent to the approximation is found. This significantly speeds up runtimes for some encoded energy formulations, but potentially makes the solution quality worse (better solutions might be found if the simulation is allowed to run for longer).
|Name|Problem|Runtime (HH:MM)|Quick Runtime (HH:MM)|
|--|--|--|--|
|col|Graph Coloring|-|-|
|tsp|Traveling Salesman|-|-|
|iso|Graph Isomorphism|-|-|
|knp|Knapsack|-|-|
|stp|Steiner Tree|-|-|


## Usage
The main command is `solver.py`:
```
python3 solver.py [-h] [-t TRIALS] [-q] [-x THREADS] [-i ITERS] [-s SEED] [-f] [-bi BETA_INIT] [-be BETA_END] [-bl] [-lr LOG_RATE] [-o] {cut,col,tsp,iso,knp,stp} ...
```
There are many options, explained in detail later. To run a simple Traveling Salesman Problem example over 8 cities with a conventional quadratic energy function, run the following command:
```
python3 solver.py -o tsp -n 8
```
This should return the following output:
```
[generate] Generating energy function...
[generate] Calculating symbolic gradients...
[compile] Compile time was 0.19
[lower] Lowering to p-computer...
[lower] Used p-bits: 64
[lower] Depedent colors: 16
[lower] Lowering time was 0.76
[run] Beginning execution...
[run] Done! Runtime was 1.37
==== RUN STATS ====
CtS: 28573
Best Cycle: 52184
Sol qual(%): 0.27
```
FUSE prints out some data about the compilation and lowering process and then begins execution. We see that the p-computer was able to find the approximate solution in 28573 cycles, and that the best proposed solution was found at cycle 52184 and was 27% better than the approximate solution.

To run the same problem using an encoded energy function, add the `-f` flag:
```
python3 solver.py -o -f tsp -n 8
```
This returns the following output:
```
[compile] Compile time was 0.00
[lower] Lowering to p-computer...
[lower] Lowering time was 0.00
[run] Beginning execution...
[run] Done! Runtime was 1.51
==== RUN STATS ====
CtS: 25
Best Cycle: 1717
Sol qual(%): 0.27
```
The encoded energy function takes 25 cycles to find a solution as good as the approximation, and takes 1717 cycles to find its best solution, which is also 27% better than the approximate solution.

### Options for FUSE
```
options:
  -h, --help            show help message and exit
  -o, --overwrite       Overwrite existing log directory, if it exists (will error otherwise)
  -t TRIALS 			Number of trials (unique problems) to test. Prints stats for values > 1
  -x THREADS			Number of threads to use for multiple trials
  -q, --quick           Early exit simulation if approximate solution is found
  -i ITERS              (Maximum) Number of iterations run
  -s SEED, --seed SEED  Random seed used
  -f, --enc             Use Encoded Energy Function instead of conventional (default)
  -bi BETA_INIT 		Initial Beta value (default 0)
  -be BETA_END			Log_10 of ending Beta value (default 0 => 1)
  -bl, --beta_log       Use Logarithmic schedule to raise beta instead of linear (default)
  -lr LOG_RATE	 		Proportion of logs to keep (default 0.1%)
```
FUSE currently supports 6 problems (5 of which are described in the paper, with MaxCut having no encoded energy formulation). These are:

Each problem has some parameters, described in their respective `prob/{prob}.py` files.
