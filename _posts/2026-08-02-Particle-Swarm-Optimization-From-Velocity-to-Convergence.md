---
title: Particle Swarm Optimization - From Velocity to Convergence
author: fanghao
date: 2026-08-02 09:00:00 +0200
categories: [Optimization]
tags: [optimization, pso, swarm-intelligence, python]
math: true
---

Particle Swarm Optimization (PSO) searches with a population of moving points. Each particle remembers the best position it has found, while the swarm shares the best position found by any particle. The balance between memory, social learning, and momentum gives PSO its characteristic behavior.

This is the first article in the [heuristic optimization series]({% post_url 2026-08-01-From-PSO-to-NSGA-II-Building-an-Optimization-Algorithm-Toolkit %}). The complete implementation is in [`optimization_algorithms/pso.py`](https://github.com/FanghaoT/OptimizationAlgorithm/blob/main/optimization_algorithms/pso.py).

## The update rule

For particle \(i\), PSO updates velocity and position as

\[
v_i^{t+1}=w v_i^t+c_1r_1(p_i-x_i^t)+c_2r_2(g-x_i^t),
\]

\[
x_i^{t+1}=x_i^t+v_i^{t+1}.
\]

Here, \(p_i\) is the particle's personal best, \(g\) is the global best, and \(r_1,r_2\) are random vectors. The three velocity terms have distinct roles:

- inertia \(w v_i\) preserves motion and encourages exploration;
- the cognitive term pulls a particle toward its own experience;
- the social term pulls it toward information shared by the swarm.

Large acceleration coefficients can produce unstable motion. Too little inertia or diversity can make the swarm collapse early around a local minimum.

## Implementation flow

The repository implementation uses continuous bounded variables and minimization semantics:

1. sample positions uniformly inside the bounds;
2. initialize small random velocities;
3. evaluate all particles and store personal and global bests;
4. update velocity and position;
5. clip both velocity and position to safe ranges;
6. update the memories only when an objective value improves.

Keeping `personal_positions` separate from the current `positions` is essential. If the same array is reused accidentally, the cognitive term no longer points to a remembered best.

## Run PSO on Rastrigin

```python
from optimization_algorithms import ParticleSwarmOptimizer
from optimization_algorithms.benchmarks import rastrigin

optimizer = ParticleSwarmOptimizer(
    population_size=60,
    iterations=200,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    seed=42,
)

result = optimizer.minimize(rastrigin, [(-5.12, 5.12)] * 5)
print(f"best value: {result.fun:.6g}")
print(f"best point: {result.x}")
```

Rastrigin is multimodal, so it reveals premature convergence more clearly than the smooth Sphere benchmark. The `history` field stores the best-so-far value and can be plotted to diagnose stagnation.

## Parameter guide

| Parameter | Effect |
|---|---|
| `population_size` | More coverage, but more objective evaluations per iteration |
| `iterations` | Longer search budget |
| `inertia` | Higher values retain momentum; lower values emphasize local refinement |
| `cognitive` | Strength of personal memory |
| `social` | Strength of swarm-wide information |

PSO is a useful first choice for bounded continuous problems when gradients are missing or unreliable. It is less natural for strongly discrete representations, strict constraints, or problems where every objective evaluation is extremely expensive.

Next: [Genetic Algorithm - Selection, Crossover, and Mutation]({% post_url 2026-08-02-Genetic-Algorithm-Selection-Crossover-and-Mutation %}).
