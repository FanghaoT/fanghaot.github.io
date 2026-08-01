---
title: From PSO to NSGA-II - Building an Optimization Algorithm Toolkit
author: fanghao
date: 2026-08-01 15:30:00 +0200
categories: [Optimization]
tags: [optimization, pso, genetic-algorithm, differential-evolution, simulated-annealing, nsga-ii, python]
---

My [OptimizationAlgorithm](https://github.com/FanghaoT/OptimizationAlgorithm) repository started as a small Particle Swarm Optimization experiment. I have now rebuilt it as a compact Python toolkit covering both single-objective and multi-objective optimization.

The goal is educational: keep the implementation readable enough to study, while providing consistent interfaces, deterministic experiments, useful examples, and tests.

## What is included?

| Family | Algorithm | Best suited to |
|---|---|---|
| Swarm intelligence | Particle Swarm Optimization (PSO) | Smooth or moderately multimodal continuous problems |
| Evolutionary | Genetic Algorithm (GA) | Flexible population search and custom operators |
| Evolutionary | Differential Evolution (DE) | Robust continuous parameter optimization |
| Trajectory-based | Simulated Annealing (SA) | Small problems where occasional uphill moves help escape local optima |
| Multi-objective | NSGA-II | Approximating a diverse Pareto front |

All implementations use minimization semantics and explicit bounds. A fixed random seed makes comparisons reproducible.

## A common single-objective interface

An objective is simply a Python function that maps a decision vector to one scalar:

```python
from optimization_algorithms import ParticleSwarmOptimizer
from optimization_algorithms.benchmarks import rastrigin

optimizer = ParticleSwarmOptimizer(
    population_size=60,
    iterations=200,
    seed=42,
)

result = optimizer.minimize(
    rastrigin,
    bounds=[(-5.12, 5.12)] * 5,
)

print(result.x)       # best decision vector
print(result.fun)     # best objective value
print(result.history) # convergence history
```

The same objective and bounds can be passed to GA, DE, or SA. This makes it straightforward to compare methods without rewriting the optimization problem.

## Why add multiple algorithms?

No optimizer is universally best. Their search behavior differs:

- **PSO** shares information through personal and global best positions. It is intuitive and often converges quickly, but population diversity can collapse.
- **GA** combines selection, crossover, and mutation. Its operator-based structure is highly adaptable, especially when a problem needs a custom representation.
- **DE** creates trial vectors from differences between population members. It is simple and particularly effective for many continuous problems.
- **SA** maintains one solution and sometimes accepts a worse move. This controlled randomness can help it cross local barriers with little memory overhead.

The repository includes Sphere, Rastrigin, and Rosenbrock benchmark functions so the algorithms can be tested on problems with different landscapes.

## Moving from one objective to many

Engineering design rarely has only one goal. A converter design, for example, may need to balance efficiency, power density, cost, thermal stress, and reliability. These objectives often conflict, so there is no single solution that is best in every respect.

A solution is **Pareto dominant** when it is no worse in every objective and strictly better in at least one. The set of solutions not dominated by any other solution forms the **Pareto front**. Instead of returning one point, a multi-objective optimizer returns a set of trade-off candidates for the designer to examine.

## NSGA-II in the toolkit

The NSGA-II implementation follows four central ideas:

1. Rank candidates using non-dominated sorting.
2. Estimate local solution density with crowding distance.
3. Prefer low-rank and less-crowded candidates in tournament selection.
4. Combine parents and offspring before selecting the next generation, preserving strong solutions.

For continuous variables, the implementation uses simulated binary crossover and polynomial mutation.

```python
from optimization_algorithms import NSGA2
from optimization_algorithms.benchmarks import zdt1

optimizer = NSGA2(
    population_size=100,
    generations=200,
    seed=42,
)

result = optimizer.minimize(
    zdt1,
    bounds=[(0.0, 1.0)] * 30,
)

print(result.x.shape)   # non-dominated decision vectors
print(result.fun.shape) # corresponding objective values
```

ZDT1 is included as a first validation problem because its two-objective Pareto front is easy to visualize and compare.

## Reliability improvements

The expanded repository is more than a collection of scripts. It now includes:

- input and bound validation;
- dedicated result objects;
- reproducible NumPy random generators;
- convergence histories;
- runnable examples;
- unit tests for convergence, deterministic runs, invalid inputs, and NSGA-II non-domination;
- packaging through `pyproject.toml`.

These details matter when an algorithm moves from a notebook experiment into a reusable engineering workflow.

## What comes next?

The next steps are constraint-handling strategies, mixed discrete-continuous variables, Pareto-front quality indicators such as hypervolume and IGD, and additional algorithms such as CMA-ES, MOEA/D, and Multi-Objective PSO.

The complete source, examples, and installation instructions are available in the [OptimizationAlgorithm repository](https://github.com/FanghaoT/OptimizationAlgorithm).
