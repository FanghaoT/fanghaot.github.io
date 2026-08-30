---
title: NSGA-II - From One Best Solution to a Pareto Front
author: fanghao
date: 2026-08-02 13:00:00 +0200
categories: [Optimization]
tags: [optimization, nsga-ii, multi-objective-optimization, pareto-front, python]
math: true
---

Multi-objective problems do not usually have one best solution. Improving efficiency may increase cost; reducing mass may increase stress. NSGA-II searches for a diverse set of non-dominated trade-offs rather than collapsing those objectives into one arbitrary weighted sum.

This article is part of the [heuristic optimization series]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}). The complete implementation is in [`optimization_algorithms/nsga2.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/nsga2.py), with a runnable [`ZDT1 example`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/examples/nsga2_zdt1.py).

## Dominance and Pareto optimality

For minimization, solution \(a\) dominates solution \(b\) when \(a\) is no worse in every objective and strictly better in at least one. A non-dominated solution has no known competitor that dominates it. The objective vectors of all such solutions approximate the Pareto front.

This changes the result shape. Single-objective optimizers return one decision vector `x` and one value `fun`. NSGA-II returns a matrix of decision vectors and the corresponding matrix of objective vectors.

## The four mechanisms in NSGA-II

1. **Fast non-dominated sorting** assigns each candidate a rank. Rank zero is the current non-dominated front.
2. **Crowding distance** estimates local sparsity in objective space. Boundary points receive infinite distance so the extremes are preserved.
3. **Crowded tournament selection** prefers lower rank, then greater crowding distance.
4. **Elitist environmental selection** combines parents and children before filling the next population front by front.

The continuous implementation creates offspring with simulated binary crossover (SBX) and polynomial mutation. Their distribution indices control how closely children and mutations remain around existing solutions.

## Run NSGA-II on ZDT1

```python
from optimization_algorithms import NSGA2
from optimization_algorithms.benchmarks import zdt1

optimizer = NSGA2(
    population_size=100,
    generations=200,
    crossover_rate=0.9,
    crossover_eta=15.0,
    mutation_eta=20.0,
    seed=42,
)

result = optimizer.minimize(zdt1, [(0.0, 1.0)] * 30)
print(result.x.shape)
print(result.fun.shape)
```

ZDT1 has two objectives and a known front, making it suitable for a first visual validation. The example script plots `result.fun[:, 0]` against `result.fun[:, 1]`.

## Reading the result correctly

A large rank-zero set is not automatically good. A useful approximation should also converge toward the true front and cover it evenly. Visual inspection is a start, but repeated runs and indicators such as hypervolume, generational distance, and IGD provide stronger evidence.

Population size matters more visibly here than in single-objective search because it limits how many trade-offs can be represented at once. Constraint handling is also crucial in engineering applications: non-dominated but infeasible solutions are not useful design choices.

NSGA-II is a strong general baseline for continuous multi-objective problems. It becomes expensive as population size, objective count, or evaluation cost grows, and many-objective problems often require different diversity mechanisms.

Previous: [Simulated Annealing]({% post_url 2026-08-02-Simulated-Annealing-Escaping-Local-Optima %}). Return to the [series index]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}).
