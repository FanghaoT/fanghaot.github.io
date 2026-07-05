---
title: SARSA Lambda - Adding Memory with Eligibility Traces
author: fanghao
date: 2024-01-26 10:00:00 +0200
categories: [AI, Reinforcement Learning]
tags: [sarsa-lambda, eligibility-traces, temporal-credit, multi-step]
---

# SARSA Lambda: Adding Memory with Eligibility Traces

Standard SARSA updates one state-action pair at a time. This is simple, but it can be slow when a reward comes much later than the useful action.

SARSA Lambda adds a short-term memory called an eligibility trace. It lets the agent give credit to several recent decisions at once.

## The Problem

Imagine an agent moves through a maze. It makes a good turn early, walks several steps, and then reaches the goal.

Standard SARSA mostly updates the last step before the goal. The earlier good turn only receives credit slowly over many episodes.

Eligibility traces solve this by remembering recently visited state-action pairs.

## The Core Idea

Each state-action pair has a trace value.

- Recently visited pairs have high trace values.
- Older pairs fade over time.
- When reward arrives, all pairs with active traces are updated.

A simple way to think about it:

```text
recent action -> strong credit
older action  -> weaker credit
forgotten     -> no credit
```

## The Update

SARSA Lambda still uses the SARSA TD error:

```text
delta = R + gamma Q(S', A') - Q(S, A)
```

But instead of updating only `Q(S,A)`, it updates all state-action pairs according to their traces:

```text
Q(s,a) <- Q(s,a) + alpha * delta * e(s,a)
e(s,a) <- gamma * lambda * e(s,a)
```

Here `lambda` controls how quickly the trace fades.

## Understanding Lambda

`lambda` is between 0 and 1.

- `lambda = 0`: only the most recent step matters. This is close to normal SARSA.
- `lambda = 1`: credit travels far back through the episode.
- A value such as `0.8` or `0.9`: a practical middle ground.

Higher lambda can learn faster in sparse-reward tasks, but it can also make updates noisier.

## A Small Code Sketch

```python
error = q_target - q_predict

# Mark the current state-action pair as recently visited
eligibility_trace.loc[s, a] += 1

# Update all Q-values using the trace
q_table += learning_rate * error * eligibility_trace

# Fade old traces
eligibility_trace *= gamma * trace_decay
```

The main difference from SARSA is that the update is spread across the recent path.

## When It Helps

SARSA Lambda is useful when:

- Rewards are delayed.
- A full path matters more than the last action.
- The state space is still small enough for a table.
- You want faster learning than one-step SARSA.

It is less useful when the reward is immediate or the state space is too large to store traces easily.

## Key Takeaways

- SARSA Lambda adds memory to SARSA.
- Eligibility traces remember recently visited state-action pairs.
- Rewards can update several past actions at once.
- `lambda` controls how far back credit is assigned.
- It is a bridge between one-step TD learning and full-episode learning.

---

*Explore SARSA Lambda implementation and trace visualizations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
