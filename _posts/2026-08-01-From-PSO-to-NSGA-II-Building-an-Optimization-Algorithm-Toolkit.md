---
title: Heuristic Optimization Algorithms - A Code-First Learning Series
author: fanghao
date: 2026-08-01 15:30:00 +0200
categories: [Optimization]
tags: [optimization, heuristic-algorithms, python]
---

Heuristic optimization is not one algorithm. It is a family of search strategies with different assumptions, operators, strengths, and failure modes. This series therefore gives each algorithm its own article and its own readable Python implementation instead of compressing everything into a single survey.

All code lives in the [OptimizationAlgorithm repository](https://github.com/FanghaoT/OptimizationAlgorithm). The implementations share explicit bounds, minimization semantics, seeded NumPy random generators, result objects, and automated tests, so the algorithms can be studied and compared without unrelated interface differences.

## Series map

| # | Algorithm | Central idea | Article | Implementation |
|---|---|---|---|---|
| 1 | Particle Swarm Optimization (PSO) | Learn from personal and swarm experience | [Read the PSO article]({% post_url 2026-08-02-Particle-Swarm-Optimization-From-Velocity-to-Convergence %}) | [`pso.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/pso.py) |
| 2 | Genetic Algorithm (GA) | Evolve a population through selection and variation | [Read the GA article]({% post_url 2026-08-02-Genetic-Algorithm-Selection-Crossover-and-Mutation %}) | [`ga.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/ga.py) |
| 3 | Differential Evolution (DE) | Use population differences as search directions | [Read the DE article]({% post_url 2026-08-02-Differential-Evolution-Simple-Operators-Strong-Search %}) | [`de.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/de.py) |
| 4 | Simulated Annealing (SA) | Escape local optima by accepting controlled uphill moves | [Read the SA article]({% post_url 2026-08-02-Simulated-Annealing-Escaping-Local-Optima %}) | [`sa.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/sa.py) |
| 5 | NSGA-II | Preserve a diverse set of Pareto-optimal trade-offs | [Read the NSGA-II article]({% post_url 2026-08-02-NSGA-II-From-One-Best-Solution-to-a-Pareto-Front %}) | [`nsga2.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/nsga2.py) |

## A shared experimental interface

The four single-objective algorithms accept the same objective and bounds:

```python
from optimization_algorithms import ParticleSwarmOptimizer
from optimization_algorithms.benchmarks import rastrigin

optimizer = ParticleSwarmOptimizer(seed=42)
result = optimizer.minimize(rastrigin, [(-5.12, 5.12)] * 5)

print(result.x)
print(result.fun)
print(result.history)
```

Replace `ParticleSwarmOptimizer` with `GeneticAlgorithm`, `DifferentialEvolution`, or `SimulatedAnnealing` to keep the problem fixed while changing the search strategy. NSGA-II uses the same bounds convention but returns decision vectors and objective vectors for a non-dominated set.

## What each article contains

Each article follows the same structure:

1. the intuition behind the algorithm;
2. the update rules or genetic operators;
3. a guided reading of the repository implementation;
4. a runnable benchmark example;
5. parameter choices, common failure modes, and suitable use cases.

The goal is not to declare a universal winner. It is to make the behavior of each method understandable enough that an optimizer can be chosen, tuned, and debugged deliberately.

Future additions will extend this index with CMA-ES, MOPSO, MOEA/D, constraint handling, and mixed discrete-continuous optimization.
