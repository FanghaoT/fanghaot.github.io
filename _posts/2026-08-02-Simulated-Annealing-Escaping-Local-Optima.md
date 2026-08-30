---
title: Simulated Annealing - Escaping Local Optima
author: fanghao
date: 2026-08-02 12:00:00 +0200
categories: [Optimization]
tags: [optimization, simulated-annealing, stochastic-search, python]
math: true
---

Simulated Annealing (SA) searches with one current solution. Unlike a greedy local search, it can accept a worse candidate. Early in the run these uphill moves help the search cross local barriers; later, cooling makes the algorithm increasingly selective.

This article is part of the [heuristic optimization series]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}). The implementation is in [`optimization_algorithms/sa.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/sa.py).

## Metropolis acceptance

Let \(\Delta=f(x')-f(x)\) for a minimization problem. A candidate is accepted when it improves the current point. If it is worse, it is accepted with probability

\[
P(\text{accept})=\exp(-\Delta/T).
\]

At high temperature \(T\), even a noticeable deterioration may be accepted. At low temperature, the same move is unlikely. The implementation uses geometric cooling:

\[
T_{k+1}=\alpha T_k,
\]

where `cooling_rate` is \(\alpha\).

## Current solution versus best solution

SA must track two different states:

- the **current** solution, which may become worse after an accepted uphill move;
- the **best** solution found at any point in the run.

Returning the current solution would make the final result depend unnecessarily on the last few random transitions. The repository returns the best solution and records its best-so-far history.

## Run SA on Rastrigin

```python
from optimization_algorithms import SimulatedAnnealing
from optimization_algorithms.benchmarks import rastrigin

optimizer = SimulatedAnnealing(
    iterations=10_000,
    initial_temperature=2.0,
    cooling_rate=0.999,
    step_scale=0.08,
    seed=42,
)

result = optimizer.minimize(rastrigin, [(-5.12, 5.12)] * 2)
print(result.fun)
print(result.x)
```

The implementation proposes a Gaussian perturbation scaled by each variable's bound range, then clips the candidate to the feasible box.

## Tuning the schedule

| Symptom | Likely adjustment |
|---|---|
| Search freezes early | Increase initial temperature or cooling rate |
| Search remains random | Cool faster or lower the initial temperature |
| Candidates barely move | Increase `step_scale` |
| Candidates repeatedly hit bounds | Decrease `step_scale` or improve bound handling |

SA uses little memory and adapts naturally to custom neighborhood operators. Its sequential search can require many iterations, and one poorly chosen temperature schedule can dominate the result.

Previous: [Differential Evolution]({% post_url 2026-08-02-Differential-Evolution-Simple-Operators-Strong-Search %}). Next: [NSGA-II]({% post_url 2026-08-02-NSGA-II-From-One-Best-Solution-to-a-Pareto-Front %}).
