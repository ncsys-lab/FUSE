# FUSE: A tool for optimizing (Encoded) Energy Functions
This is the implementation for the tool described in Efficient Optimization with Encoded Energy Functions (HPCA 2025, link pending). It allows one to define problems and various energy function mappings for those problems, along with a p-computing simulator.

## Installation
If using [Docker](https://www.docker.com), you can spin up a container with all the dependencies installed after cloning the repo (may need to `chmod +x` the files):
```
./build_image.sh
./run_image.sh
```
This will drop you in a bash instance ready to run FUSE.

Alternatively, you can also use a virtual environment and install dependencies via  `pip install -r requirements.txt`. We have tested FUSE on Python3.11.
## Kick-the-Tires Phase
To run a simple Traveling Salesman Problem example over 8 cities with a conventional quadratic energy function, run the following command:
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



## Replicating Experiments
### Table IV
You can use `scripts/run_{prob}_exp.sh` to run the encoded and conventional energy function experiments detailed in Table IV. You can also use `scripts/run_t4_exps.sh` to queue up all the experiments. We report the runtimes for each experiment on a consumer-grade laptop CPU.
|Name|Problem|Runtime (HH:MM:SS)|
|--|--|--|
|col|Graph Coloring|00:03:57|
|tsp|Traveling Salesman|00:06:37|
|iso|Graph Isomorphism|00:16:44|
|knp|Knapsack|00:07:37|
|stp|Steiner Tree|-|
|Total||-|

### Table V
You can use `scripts/run_{prob}_scale_quick.sh` to run the encoded energy function scaling experiments detailed in Table V (CtS and ESP numbers), or `scripts/run_t5_scale_quick.sh` to queue up all experiments. To get all the data (CtS, ESP, and solution quality metrics), run `scripts/run_{prob}_scale_long.sh` (or `scripts/run_t5_scale_long.sh` to queue up all experiments). These experiments do not exit early and thus runtimes can be long.
|Name|Quick Runtime (HH:MM)|Long Runtime (HH:MM)|
|--|--|--|
|col|-|-|
|tsp|-|-|
|iso|-|-|
|knp|-|-|
|stp|-|-|
|Total||-|

### Table VI
You can use `scripts/run_t6_exps.sh` to run experiments comparing a size N selection network to a size Log(N) selection network detailed in Table VI.
|Name|Runtime (HH:MM:SS)|
|--|--|
|col|00:03:59|


### Table VII
You can use `scripts/run_t7_exps.sh` to run experiments synthesizing a conventional TSP circuit with an encoded energy function
|Name|Runtime (HH:MM:SS)|
|--|--|
||00:03:59|

## Replicating Figures
You can use `scripts/run_t7_exps.sh` to run experiments synthesizing a conventional TSP circuit with an encoded energy function

## General FUSE Usage
The main command is `solver.py`:
```
python3 solver.py [-h] [-t TRIALS] [-l] [-x THREADS] [-i ITERS] [-s SEED] [-f] [-bi BETA_INIT] [-be BETA_END] [-bl] [-lr LOG_RATE] [-o] {cut,col,tsp,iso,knp,stp} ...
```
There are many options, explained in detail later. ### Options for FUSE
```
-h, --help            show help message and exit
-o, --overwrite       Overwrite existing log directory, if it exists (will error otherwise)
-t TRIALS             Number of trials (unique problems) to test. Prints stats for values > 1
-x THREADS            Number of threads to use for multiple trials
-l, --long            Disable early exiting to find best solution
-i ITERS              (Maximum) Number of iterations run
-s SEED,              Random seed used
-f, --enc             Use Encoded Energy Function instead of conventional (default)
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
Just as a Problem class is a template for generating problem instances, the Energy Function classes create methods to generate energy function instances for particular problem instances. Conventional Energy Functions inherit from the `ConvEfn` class in `problems/prob.py`, and must implement three methods, `__init__, _gen_exprs`, and `compile`.

The `init` method is similar to the Problem `init` in that it sets parameters for the problem (such as the number of elements) rather than values for a specific instance. It is called during the Problem `init` method is called, when the energy function is instantiated.
```
def __init__(self, n, cap, c_maxval):
    self.n = n
    self.cap = cap
    self.c_maxval = c_maxval
    super().__init__()
```
Be sure to call `super().__init__()` at the end of initialization to begin expression generation.

For conventional energy functions, it is straightforward to generate a closed-form algebraic expression for the energy function over the spin variables. The `_gen_exprs` function is responsible for creating these expressions. We will break down the `knp` example, line-by-line.
```
def _gen_exprs(self):
    spins = np.array(sympy.symbols(f"s:{self.n}"))
    costs = np.array(sympy.symbols(f"c:{self.n}"))
    weights = np.array(sympy.symbols(f"w:{self.n}"))
```
We begin by using sympy to create symbolic variables for the spins. For a particular problem instance, the final energy function should only be a function of the spins, but we do not yet have a problem instance when this method is called at initialization time. Thus, we will also create symbolic variables for the problem instance variables, which will be substituted in at compile-time. We package these as numpy arrays in order to use vector operations for faster manipulation.
```
    cap = int(self.cap) + 1
    m = floor(log2(cap))
    vals = [1 << i for i in range(m)]
    vals.append(cap - (1 << m))

    w_spins = np.array(sympy.symbols(f"ws:{len(vals)}"))
    vals = np.array(vals)
```
For Knapack, we have to add some additional variables for keeping track of the claimed weight of selected items. Now we have four symbolic arrays, `spins, w_spins, costs,` and `weights`. The former two are problem variables, and the latter two are instance variables.
```
    cost_expr = -np.dot(spins, costs)
    weight_expr = np.dot(spins, weights)
    cweight_expr = np.dot(w_spins, vals)

    invalid_expr = self.c_maxval * self.n * (weight_expr - cweight_expr) ** 2
    energy_expr = invalid_expr + cost_expr
```
Now we create the algebraic expressions for the energy. We start by computing the cost of selected items, and negate the sum as a higher cost will lower our energy. We also compute the actual weight and the claimed weight. We create an `invalid_expr`, which raises the energy of invalid weight combinations by a large constant. Finally, we create the full `energy_expr` by summing the cost and invalid expressions.
```
    self.costs = costs
    self.weights = weights
```
We add the instance variable arrays as variables of the class. This is because we will have to reference the symbols later in order to make subsitutions at compile time.
```
    out_spins = np.hstack((spins, w_spins))
    return energy_expr, out_spins
```
Finally, we flatten the spin array into one list, and return the energy expression and the list of spins. During initalization, the `ConvEfn` class will call `_gen_exprs` and compute their symbolic derivatives w.r.t the spin variables, leaving the instance variables symbolic. It also reduces squared variables to linear ones (as all values are boolean).

The `compile` method is responsible for substituting in instance variables before calling the superclass `compile()`. It consumes a problem instance.
```
def compile(self, inst):
    weights, costs = inst
    weight_dict = {weight: inst_w for weight, inst_w in zip(self.weights, weights)}
    cost_dict = {cost: inst_c for cost, inst_c in zip(self.costs, costs)}
    sub_dict = {**weight_dict, **cost_dict}
    return super().compile(sub_dict)
```
We get the numpy arrays from the problem instance tuple, and create dictionaries that map instance variables to their value (this is why we needed to save the symbolic variables during `_gen_exprs`). For convenience, we create two separate dictionaries for the two sets of instance variables, and merge them. We then pass this substitution dictionary to the superclass `compile()` method.

The superclass compile method will perform the requisite substitutions and create a set of matrices `J, h` that can be used to efficiently compute values and gradients of conventional energy functions. It also performs a graph coloring to find sets of spins that can be updated in parallel.

Our energy function is fully defined and is ready to be used with the simulator. Execution flow goes as follows:
1. The main method will use the `parse` method to dispatch the arguments to the correct class:
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
