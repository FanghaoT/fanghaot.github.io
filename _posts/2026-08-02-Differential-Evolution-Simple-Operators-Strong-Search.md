---
title: Differential Evolution - Simple Operators, Strong Search
author: fanghao
date: 2026-08-02 11:00:00 +0200
categories: [Optimization]
tags: [optimization, differential-evolution, evolutionary-computation, python]
math: true
---

Differential Evolution (DE) is a population method for continuous optimization. Its distinctive idea is to build a search step from the difference between two existing candidates. The population therefore supplies both candidate solutions and an adaptive notion of scale.

This article is part of the [heuristic optimization series]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}). The implementation is in [`optimization_algorithms/de.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/de.py).

## DE/rand/1/bin

For each target vector \(x_i\), choose three distinct population members \(x_a,x_b,x_c\). The mutation step creates

\[
v=x_a+F(x_b-x_c),
\]

where \(F\) is the differential weight. Binomial crossover then constructs a trial vector:

\[
u_j=\begin{cases}
v_j, & r_j<CR,\\
x_{i,j}, & \text{otherwise}.
\end{cases}
\]

At least one coordinate is forced to come from the mutant. Finally, greedy selection keeps the trial only if it improves the target.

This `rand/1/bin` label means a random base vector, one difference vector, and binomial crossover.

## Why the difference vector matters

When the population is widely dispersed, differences are large and exploration steps are naturally broad. As the population contracts, the same operation produces smaller refinement steps. This self-scaling behavior is one reason DE performs well on many continuous numerical problems.

The implementation clips mutant vectors to explicit bounds. Bound handling is not a minor detail: clipping, reflection, resampling, and repair can create measurably different behavior near constraints.

## Run DE on Rastrigin

```python
from optimization_algorithms import DifferentialEvolution
from optimization_algorithms.benchmarks import rastrigin

optimizer = DifferentialEvolution(
    population_size=60,
    generations=200,
    differential_weight=0.8,
    crossover_rate=0.9,
    seed=42,
)

result = optimizer.minimize(rastrigin, [(-5.12, 5.12)] * 5)
print(result.fun)
print(result.x)
```

`differential_weight` controls step amplitude. `crossover_rate` controls how much of the mutant enters the trial. Very small populations provide few distinct directions; overly large weights can repeatedly push candidates against the bounds.

## When to use DE

DE is an excellent baseline for bounded continuous optimization because it has few operators and relatively few parameters. It is less direct for categorical variables or specialized combinatorial structures, where a representation-aware GA may be easier to adapt.

Previous: [Genetic Algorithm]({% post_url 2026-08-02-Genetic-Algorithm-Selection-Crossover-and-Mutation %}). Next: [Simulated Annealing]({% post_url 2026-08-02-Simulated-Annealing-Escaping-Local-Optima %}).
