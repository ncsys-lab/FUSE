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
-h, --help            show help message and exit
-o, --overwrite       Overwrite existing log directory, if it exists (will error otherwise)
-t TRIALS             Number of trials (unique problems) to test. Prints stats for values > 1
-x THREADS            Number of threads to use for multiple trials
-q, --quick           Early exit simulation if approximate solution is found
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
This section is intended for users that want to extend FUSE to new problems. One can add a new problem in three steps:
### Create Problem Definition and Register it
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
### Create Conventional Energy Function
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
We begin by using sympy to create symbolic variables for the spins. For a particular problem instance, the final energy function should only be a function of the spins, but at initialization time, when this method is called, we do not have a problem instance yet. Thus, we will also create symbolic variables for the problem instance variables, which will be substituted in at compile-time. We package these as numpy arrays in order to use vector operations for faster manipulation.
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
Now we create the algebraic expressions for the energy. We start by computing the cost of selected items, and negate the sum as a higher cost will lower our energy. We also compute the actual weight and the claimed weight. Finally, we create an `invalid_expr`, which raises the energy of invalid weight combinations by a large constant. Finally, we create the full `energy_expr` by summing the cost and invalid expressions.
```
    self.costs = costs
    self.weights = weights
```
We add the instance variable arrays as variables of the class. This is because we will have to reference the symbols later in order to make subsitutions at compile time.
```
    out_spins = np.hstack((spins, w_spins))
    return energy_expr, out_spins
```
Finally, we flatten the spin array into one list, and return the energy expression and the list of spins.
### Create Encoded Energy Function
#### Using Circuit functions

#### Using Non-JAX functions
