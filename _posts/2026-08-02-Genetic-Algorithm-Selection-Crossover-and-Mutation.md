---
title: Genetic Algorithm - Selection, Crossover, and Mutation
author: fanghao
date: 2026-08-02 10:00:00 +0200
categories: [Optimization]
tags: [optimization, genetic-algorithm, evolutionary-computation, python]
math: true
---

A Genetic Algorithm (GA) treats candidate solutions as a population that evolves. Selection gives better candidates more opportunities to reproduce, crossover combines information from parents, and mutation introduces new variation.

This article is part of the [heuristic optimization series]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}). The code is in [`optimization_algorithms/ga.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/ga.py).

## The evolutionary loop

The implementation uses real-valued vectors rather than binary chromosomes. One generation consists of:

1. preserve a small number of elite candidates;
2. select parents with tournament selection;
3. create children with arithmetic crossover;
4. mutate selected coordinates with Gaussian noise;
5. clip children to the variable bounds;
6. evaluate the new population.

Tournament selection samples several candidates and returns the best among them. Increasing the tournament size raises selection pressure: useful solutions spread faster, but diversity disappears sooner.

For parents \(x^{(1)}\) and \(x^{(2)}\), arithmetic crossover uses a random vector \(\alpha\):

\[
y^{(1)}=\alpha x^{(1)}+(1-\alpha)x^{(2)},
\]

\[
y^{(2)}=\alpha x^{(2)}+(1-\alpha)x^{(1)}.
\]

Mutation then perturbs individual coordinates. When `mutation_rate` is omitted, the implementation uses \(1/d\), giving approximately one mutated coordinate per child in \(d\) dimensions.

## Run GA on Rosenbrock

```python
from optimization_algorithms import GeneticAlgorithm
from optimization_algorithms.benchmarks import rosenbrock

optimizer = GeneticAlgorithm(
    population_size=80,
    generations=250,
    crossover_rate=0.9,
    mutation_scale=0.08,
    tournament_size=3,
    elitism=2,
    seed=42,
)

result = optimizer.minimize(rosenbrock, [(-3.0, 3.0)] * 5)
print(result.fun)
print(result.x)
```

Rosenbrock has a narrow curved valley. It is a useful test of whether selection and variation can both locate and follow a difficult basin.

## What the parameters control

| Parameter | Trade-off |
|---|---|
| `population_size` | Search coverage versus evaluation cost |
| `crossover_rate` | Recombination of known material |
| `mutation_rate` | How many coordinates receive new variation |
| `mutation_scale` | Size of mutation steps relative to each bound range |
| `tournament_size` | Selection pressure |
| `elitism` | Protection from losing the best candidates |

Too much elitism or selection pressure can make the population nearly identical. Too much mutation turns the method into an inefficient random search. The right balance depends on representation and landscape.

GA is particularly valuable when custom representations and domain-specific crossover or repair operators matter. The current implementation deliberately focuses on bounded real-valued problems so its operators remain easy to read.

Previous: [Particle Swarm Optimization]({% post_url 2026-08-02-Particle-Swarm-Optimization-From-Velocity-to-Convergence %}). Next: [Differential Evolution]({% post_url 2026-08-02-Differential-Evolution-Simple-Operators-Strong-Search %}).
