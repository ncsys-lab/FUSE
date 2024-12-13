# FUSE: A tool for optimizing (Encoded) Energy Functions
This is the implementation for the tool described in Efficient Optimization with Encoded Energy Functions (HPCA 2025, link pending). It allows one to define problems and various energy function mappings for those problems, along with a p-computing simulator.

## Installation
First, clone the repo:

```
git clone --recurse-submodules https://github.com/ncsys-lab/FUSE.git
cd FUSE
```
The supported way to use FUSE is through [Docker](https://www.docker.com). You can spin up a container with all the dependencies installed after cloning the repo:
```
# Will take about 5 mins to install of synthesis tools
./scripts/build_image.sh
./scripts/run_image.sh
```
This will drop you in a bash instance ready to run FUSE. The repo will be mounted in the container, so you can make changes to the source in your editor of choice and they will be reflected.

Alternatively, you can also use a virtual environment and install dependencies via  `pip install -r requirements.txt`. You will also need to build [OpenLane2](https://openlane2.readthedocs.io/en/latest/getting_started/common/nix_installation/index.html). We have tested FUSE on Python3.11.
## Kick-the-Tires Phase
To run a simple Traveling Salesman Problem example over 8 cities with a conventional quadratic energy function, run the following command:
```
python3 solver.py -be -1.5 -lr 0.1 tsp -n 8
```
This should return the following output:
```
[generate] Generating energy functions...
[compile] Compile time was 0.00
[lower] Lowering to p-computer...
[lower] Used p-bits: 64
[lower] Dependent colors: 16
[lower] Lowering time was 0.79
[run] Beginning execution...
[run] Done! Runtime was 0.89
==== RUN STATS ====
CtS: 416963
Best Cycle: 416963
Sol qual(%): 0.00
```
FUSE prints out some data about the compilation and lowering process and then begins execution. We see that the p-computer was able to find the approximate solution in 416 thousand cycles, and that this solution is equivalent in quality to the approximate solution.

To run the same problem using an encoded energy function, run the following command (note the `-f` flag for encoded energy functions):
```
python3 solver.py -be 2.5 -lr 0.1 -f tsp -n 8
```
This returns the following output:
```
[generate] Encocded Energy Function! Nothing to generate...
[compile] Compile time was 0.13
[lower] Lowering to p-computer...
[lower] Lowering time was 0.00
[run] Beginning execution...
[run] Done! Runtime was 0.33
==== RUN STATS ====
CtS: 461
Best Cycle: 461
Sol qual(%): 0.00
```
The encoded energy function takes 461 cycles to find a solution as good as the approximation.

We can compare the area and latency of these circuits by synthesizing them. To synthesize the conventional energy function, run the following command (note that the first synthesis run will take longer as the PDK must be downloaded):
```
python3 synth.py tsp -n 8
```
This will create a directory `synths/tsp_n8_conv`, copy relevant verilog files, and invoke OpenLane2 to run synthesis. This should yield the following output:
```
...(previous OpenLane output)
====SYNTHESIS RESULTS====
Area (um^2): 54709.97
Latency (ns): 18.37
```
Run synthesis of the encoded energy function to test verilog emmission:
```
python3 synth.py -f tsp -n 8
```
This produces the following output:
```
...(previous OpenLane output)
====SYNTHESIS RESULTS====
Area (um^2): 84562.35
Latency (ns): 20.09
```
If these tests pass, your FUSE installation should be good to go.

## Data Availability
Due to the long runtimes of certain experiments, we have opted to include raw log files (which include data about success rates, CTS, and solution quality while holding a trace of 0.1% of visited state energies by default), which can be analyzed by the following script to generate the entries in Tables IV, V, and VI.

To use the analysis script, first unzip the log files:
```
tar -xzvf exp_data.tar.gz
```
Then, one can output the statistics of a run using the following command (this is the 10 node graph coloring example from Table IV):
```
python3 analyze.py exp_data/table_iv/col/col_n10_conv_b0.00_0.00_lin_s42/*
```
For Table V experiments, please add `-i 5000000` to the command in order to reflect the higher number of iterations:
```
python3 analyze.py exp_data/table_v/stp_n100_enc_b0.00_3.75_lin_s42/* -i 5000000
```
## Setting Thread Count + OOM Errors
By default, FUSE is configured to use 10 threads. You should adjust the thread defaults to match your machine (file `solver.py`, line ~207):
```
parser.add_argument("-t", "--trials", type=int, help="Number of trials")
parser.add_argument(
    "-x", "--threads", type=int, default=10, help="Number of threads to use" # Modify this line
)
parser.add_argument(
    "-i",
```
Ensure that you have enough memory (ideally ~1.5GB per core). **Many issues with experiments hanging/ crashing are root-caused to the docker container running out of memory**. You will likely have to adjust the container's limit in the Docker Desktop app.
## Replicating Key Results
### Replicating Figures
After the Kick-the-Tires Phase, you can use `scripts/gen_plots.sh` to generate 4 plots in the `plots/` directory. These plots require that you have run the first two commands in the KtT phase to generate the relevant logs, so ensure you have done so before trying to run the script. The script will place the These plots are manually overlaid to create the figures 1A and 1B.

### Table IV
You can use `scripts/run_{prob}_exp.sh` to run the encoded and conventional energy function experiments detailed in Table IV. You can also use `scripts/run_t4_exps.sh` to queue up all the experiments. Unless noted otherwise, all reported runtimes are from a consumer-grade laptop CPU with 10 cores and 16 GB of RAM. The conventional Steiner tree benchmarks take especially long - we have included the estimated runtime on a consumer grade CPU, as well as the runtime on a larger 32 core machine. We have made the log files for these runs available in case it is impractical to run these benchmarks.
|Name|Problem|Runtime (HH:MM)|
|--|--|--|
|tsp|Traveling Salesman|00:13|
|iso|Graph Isomorphism|00:11|
|col|Graph Coloring|00:07|
|knp|Knapsack|00:08|
|stp|Steiner Tree|~27:00 (08:50 on 32-thread machine)|
|Total||27:40|

### Table V
You can use `scripts/run_t5_exps.sh` to run the encoded energy function scaling experiments detailed in Table V.
|Name|Runtime (HH:MM)|
|--|--|
|tsp|00:13|
|iso|00:40|
|col|00:22|
|knp|00:11|
|stp|01:29|
|Total|03:00|

### Table VI
You can use `scripts/run_t6_exps.sh` to run experiments comparing a size N selection network to a size Log(N) selection network detailed in Table VI.
|Name|Runtime (HH:MM)|
|--|--|
|col|00:05|

### Table VII
You can use `scripts/run_t7_exps.sh` to run experiments synthesizing a conventional TSP circuit with an encoded energy function. These scripts will print out the latency and total area of the modules.
|Name|Runtime (HH:MM)|
|--|--|
|synth|00:03|

Re-running the script will only print out results instead of re-running synthesis - this can make it easier to read the reports.

## General FUSE Usage
Note: No need to read further if you only want to replicate the above experiments. This section is intended for users who want to extend FUSE (e.g. by writing new problem definitions).
The main command is `solver.py`:
```
python3 solver.py [-h] [-t TRIALS] [-x THREADS] [-i ITERS] [-q QUALITERS] [-s SEED] [-f] [-bi BETA_INIT] [-be BETA_END] [-bl] [-lr LOG_RATE] [-o] {cut,col,tsp,iso,knp,stp} ...
```
The options for the solver script are described here:
```
-h, --help            show this help message and exit
-o, --overwrite       Overwrite existing directory, if it exists
-t TRIALS 		      Number of trials (unique problems) to test. Prints stats for values > 1
-x THREADS            THREADS Number of threads to use
-i ITERS              (Maximum) number of iterations to run
-q QUALITERS          Extra iterations to run after approx solution is found
-s SEED, --seed SEED  Random Seed used
-f, --enc             Use Encoded Energy Function
-bi BETA_INIT         Initial Beta value (default 0)
-be BETA_END          Log_10 of ending Beta value (default 0 => 1)
-bl, --beta_log       Use Logarithmic schedule to raise beta instead of linear (default)
-lr LOG_RATE          Proportion of logs to keep (default 0.1%)
```
FUSE currently supports 6 problems (5 of which are described in the paper and are listed above, with Max-Cut having no encoded energy formulation). Each problem has some parameters, described in their respective `problems/{problem}.py` files.

## Adding New Problems/ Energy Functions
This section is intended for users that want to extend FUSE to new problems. One can add a new problem in three steps.
### Create a Problem Definition and Register it
Problems inherit from the `Prob` class (`problems/prob.py`). Each problem is located in a file in `problems/{problem}.py` and must implement four methods: `gen_parser, __init__, gen_inst,` and `sol_inst`. We will use `knp` as an example. It's important to recognize that a Problem class is a template for generating random instances of a problem type, i.e. the Knp class gives methods to generate individual knapsack problems, but is not a knapsack problem instance in and of itself.

`gen_parser` is a staticmethod that creates a parser which encodes various parameters of the problem generation. In `knp`, these parameters include the number of unique elements, the range of weights and costs, and the capacity of the bag. Additionally, one should define a three letter code for the problem that is used to uniquely identify it in the parser. This string should also be returned.
```
@staticmethod
def gen_parser(subparser):
    parser = subparser.add_parser("knp", help="Knapsack Problem")
    parser.add_argument("-n", "--size", type=int, default=15)
    parser.add_argument("-w_minval", type=int, default=1)
    parser.add_argument("-w_maxval", type=int, default=10)
    parser.add_argument("-c_minval", type=int, default=1)
    parser.add_argument("-c_maxval", type=int, default=10)
    parser.add_argument("-f", "--rel_cap", type=float, default=0.33)
    return "knp"
```

The `init` method is a wrapper that converts parser arguments to variables of the Problem. Note that these are still parameters for problem generation, not the actual values of a particular (random) problem instance. This method also instantiates the Energy Function (encoded or conventional) described later.
```
def __init__(self, args):
    self.n = args.size
    self.w_maxval = args.w_maxval
    self.w_minval = args.w_minval
    self.c_maxval = args.c_maxval
    self.c_minval = args.c_minval
    self.rel_cap = args.rel_cap
    self.cap = self.n * self.w_maxval * self.rel_cap
    if args.enc:
        self.efn = KnpEncEfn(self.n, self.cap, self.c_maxval)
    else:
        self.efn = KnpConvEfn(self.n, self.cap, self.c_maxval)
```

The `gen_inst` method will consume a JAX PRNGKey and generate a new random problem instance. The return type should be a numpy array, or a tuple of numpy arrays (we mix JAX and numpy here because we want the stateful randomness of JAX, but the flexibility of numpy. We will manipulate these arrays to generate energy functions during compile-time, and this is much easier in numpy. During compilation, everything will be lowered to a JAX expression, enabling quick execution). In the Knapsack case, we will generate a set of items with random costs and weights, and will return a tuple of them.
```
def gen_inst(self, key):
    key, *keys = jax.random.split(key, num=3)
    weights = np.asarray(
        jax.random.randint(
            keys[0],
            shape=(self.n),
            minval=self.w_minval,
            maxval=self.w_maxval,
            dtype=int,
        )
    )
    costs = np.asarray(
        jax.random.randint(
            keys[1],
            shape=(self.n),
            minval=self.c_minval,
            maxval=self.c_maxval,
            dtype=int,
        )
    )
    return (weights, costs)
```

The `sol_inst` method will consume a problem instance (numpy array or tuple of numpy arrays), and return an energy that represents an approximate solution. In this case, we leverage the `knapsack` python package to find an exact solution to the problem using dynamic programming.
```
def sol_inst(self, prob_inst):
    weights, costs = prob_inst
    value, result = knap_solver.knapsack(size=weights, weight=costs).solve(
        self.cap
    )
    return -value
```
Here, we return the negative of the returned value, as we want the minimum of our energy function to be the maximum possible value.

Finally, we register this problem with `solver.py`, by adding an import statement and adding it to the list of problems in `parse`:
```
from problems.knp import Knp
...

def parse(inparser, subparser):
    probs = [Cut, Col, Tsp, Iso, Knp, Stp]
    prob_parsers = {prob.gen_parser(subparser): prob for prob in probs}
    ...
```
Our problem is registered, and we can now start writing our energy function(s).
### Create a Conventional Energy Function
Just as a Problem class is a template for generating problem instances, the Energy Function classes create methods to generate energy function instances for particular problem instances. Conventional Energy Functions inherit from the `ConvEfn` class in `problems/prob.py`, and must implement three methods, `__init__ and _gen_funcs`.

The `init` method is similar to the Problem `init` in that it sets parameters for the problem (such as the number of elements) rather than values for a specific instance. It is called during the Problem `init` method is called, when the energy function is instantiated.
```
def __init__(self, n, cap, c_maxval):
    self.n = n
    self.cap = cap
    self.c_maxval = c_maxval
    super().__init__()
```
Be sure to call `super().__init__()` at the end of initialization to begin function generation.

The `_gen_funcs` procedure is responsible for generating functions that compute the valid and cost portions of the energy fucntion, as well as determining the number of p-bits used. We will break down the `knp` example, line-by-line.

`knp` is unique in that a portion of the spins are used to select the chosen items, while the rest are used to claim a particular weight for those chosen items. In the first few lines, we determine the numerical values associated with the latter group of spins, and then count the total number of p-bits we have.
```
def _gen_funcs(self):
    cap = int(self.cap) + 1
    m = floor(log2(cap))
    vals = [1 << i for i in range(m)]
    vals.append(cap - (1 << m))
    vals = jnp.asarray(vals)

    n_spins = len(vals) + self.n
```

Here we define a JAX-jitted utility function to "dispatch" the spins - this breaks up the set of p-bits into a tuple two arrays depending on their purpose. This is useful for computing the energy.
```
    @jax.jit
    def dispatch_fn(spins):
        return jnp.split(spins, [self.n])
```

Next, we define the `valid_fn`, which takes in the state and the particular problem instance we are solving and returns a value representing the energy penalty associated with invalid solutions. This should be a JAX-jittable function as well. We use `dispatch_fn` to separate the p-bits into groups and then compute the actual weight (`weight_expr`) and the claimed weight (`cweight_expr`).
```
    def valid_fn(spins, inst):
        spins, w_spins = dispatch_fn(spins)
        weights, _ = inst
        weight_expr = jnp.dot(spins, weights)
        cweight_expr = jnp.dot(w_spins, vals)
```

An implementation detail is that, with the conventional energy functions, quadratic, single-variable terms in the energy function must be lowered to be linear (i.e. x_i^2 => x_i). Because the spins are binary, this does not affect the energy calculation, but ommitting this would make our gradient calculation incorrect. Thus, we must add a set of terms that will not change the energy but will "adjust" the gradients such that the terms appear linear. These terms usually take the form of `spin * (spin - 1)`.
FUSE will check the Hessian of the energy function and will error if the diagonal terms are non-zero. You should adjust the derivatives (while ensuring the energy function is the same) until this condition is satisfied.
```
        adjust_derivs = jnp.dot(weights * weights, spins * (spins - 1)) + jnp.dot(
            vals * vals, w_spins * (w_spins - 1)
        )
```

Then we can compute the final energy penalty by multiplying the squared difference of `weight_expr` and `cweight_expr` (minus the adjusted derivative) by a large coefficient.
```
        return (
            self.c_maxval
            * self.n
            * ((weight_expr - cweight_expr) ** 2 - adjust_derivs)
        )
```

The `cost_fn` is more straightforward:
```
    def cost_fn(spins, inst):
        spins, _ = dispatch_fn(spins)
        _, costs = inst
        return -jnp.dot(spins, costs)
```
In `knp`, we just return the negative value of the quantity we want to maximize, namely the sum of the selected items' cost.

Finally, we return the generated functions and the total number of p-bits:
```
    return valid_fn, cost_fn, n_spins
```

The superclass compile method will inject the problem instance into the functions, JIT-compile them, and compute their Hessians to create the `J` matrix. It also performs a graph coloring on this matrix to find sets of spins that can be updated in parallel. Note that an alternative compile flow, where the energy function expressions are defined at problem-compile time can be found in `col`.

Our energy function is fully defined and is ready to be used with the simulator. Execution flow goes as follows: 1. The main method will use the `parse` method to dispatch the arguments to the correct class:
```
prob, args = parse(parser, subparsers)
res = execute(prob, args)
```
2. The execute method will instantiate our Problem using the arguments. This will also instantiate our chosen energy function, which generates the expressions and computes symbolic gradients:
```
def execute(Prob, args):
    key = jax.random.key(args.seed)
    start_time = time.perf_counter()
    prob = Prob(args)
    runtime = time.perf_counter() - start_time
    print(f"[compile] Compile time was {runtime:0.2f}")
    efn = prob.efn
```
3. The `run` method is called, which generates a random problem instance, and compiles it to our energy function:
```
def run(key, quick, iters, prob, efn, beta_i, betafn):
    key, prob_key = jax.random.split(key)
    prob_inst = prob.gen_inst(prob_key)
    prob_sol = prob.sol_inst(prob_inst)

    print("[lower] Lowering to p-computer...")
    start_time = time.perf_counter()
    engradfn, masks = efn.compile(prob_inst)
    runtime = time.perf_counter() - start_time
    print(f"[lower] Lowering time was {runtime:0.2f}")
```
The compile function returns a jax function `engradfn` which computes the energy and gradients over a set of spin variables, and the masks for parallel updates. From here, execution can begin.

### Create an Encoded Energy Function
The process for creating Encoded energy functions is less involved. Encoded energy functions inherit from the `EncEfn` class, and must implement three methods: `__init__, circuit,` and `compile`.

The purpse of the `init` method is mostly unchanged - we want to set variables that determine encoding circuit generation, although there are two notable differences: we call `super().__init__()` first, and we often set the `self.spins` variable to be a number, as opposed to a numpy array of symbols.
```
def __init__(self, n, cap, c_maxval):
    super().__init__()
    self.n = n
    self.spins = n
    self.cap = cap
    self.c_maxval = c_maxval
```
The `circuit` method is a forward method that transforms an input `spins` (decision variables), along with some metadata, into problem variables. Note that this must be a boolean => boolean relation. Additionally, we employ the optimization described in section V.B.2 of the paper, where we linearize the energy function in the outputs of the encoding circuit. Thus, one must ensure that their energy function can be expressed as a linear combination of problem variables. In the knapsack case, we add another variable that is true if the selected items are larger than the capacity and zero otherwise. For maximum performance, this function should be "jit-able" and use JAX operations.
```
@staticmethod
@jax.jit
def circuit(spins, cap, weights):
    return jnp.append(spins, (spins @ weights > cap).astype(int))
```
Finally, we create a `compile` method. This method creates a set of weights, which are dotted with the outputs of the encoding circuit output to determine the energy (and gradients). Additionally, we wrap the circuit function in a lambda that passes the metadata in, such that the only argument is the decision variables. We pass this to the super `compile` method.
```
def compile(self, inst):
    weights, costs = inst
    e_weights = jnp.append(-costs, self.n * self.c_maxval)
    return super().compile(
        e_weights, lambda x: KnpEncEfn.circuit(x, self.cap, weights)
    )
```
The super `compile` method uses the differentation trick described in the paper to generate the `engradfn`. By default, the updates are all serial, although in some cases, it is possible to pass in a set of masks that define parallel updates for Encoded formulations. See `problems/col.py` for an example.
#### Using Non-JAX functions
We have written that any boolean input-output relation can be used as an encoding circuit. This extends to algorithms that may not be easily expressed in JAX. For this, we use a `jax.pure_callback` to call into other code. See `problems/stp.py` for an example using networkx's MST algorithm.
