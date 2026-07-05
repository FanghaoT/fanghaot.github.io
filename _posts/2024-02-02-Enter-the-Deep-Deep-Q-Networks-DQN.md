---
title: Enter the Deep - Deep Q-Networks (DQN)
author: fanghao
date: 2024-02-02 10:00:00 +0200
categories: [AI, Reinforcement Learning, Deep Learning]
tags: [dqn, deep-q-networks, function-approximation, experience-replay]
---

# Enter the Deep: Deep Q-Networks (DQN)

Q-Learning works well when the state space is small. In the maze example, we can store a value for every state-action pair in a table.

But many problems are too large for a table. DQN replaces the Q-table with a neural network.

## Why Q-Tables Stop Working

A Q-table needs one row for every state. That is fine for a small grid, but not for images, sensor readings, or continuous variables.

For example, a game screen is not just one state label. It is thousands of pixel values. We need a model that can generalize from one state to similar states.

This is where a neural network helps.

## From Q-Table to Q-Network

In tabular Q-Learning:

```text
Q(state, action) -> table lookup
```

In DQN:

```text
Q(state, action) -> neural network prediction
```

The network receives a state and outputs one Q-value for each possible action.

```text
input state -> neural network -> [Q(left), Q(right), Q(up), Q(down)]
```

The agent then chooses the action with the highest predicted Q-value, with some exploration.

## Two Important Stabilizers

DQN is not just "Q-Learning plus a neural network". Two tricks make it much more stable.

## Experience Replay

RL data comes in sequence. Consecutive samples are usually very similar, which is bad for neural network training.

DQN stores experiences in a replay buffer:

```text
(state, action, reward, next_state)
```

During learning, it samples a random batch from this buffer. This makes training data less correlated and allows the agent to learn from old experiences more than once.

## Target Network

DQN uses one network to predict current Q-values and another network to produce more stable targets.

```python
q_target = reward + gamma * max(target_network(next_state))
```

The target network is copied from the main network only from time to time. This prevents the target from changing too quickly.

## DQN vs Tabular Q-Learning

| Topic | Q-Learning | DQN |
|---|---|---|
| State space | Small and discrete | Large or continuous |
| Value storage | Table | Neural network |
| Generalization | No | Yes |
| Training data | One transition | Mini-batches |
| Stability tools | Usually simple | Replay buffer and target network |

## When to Use DQN

Use DQN when:

- The action space is discrete.
- The state space is too large for a table.
- You can collect many interactions in simulation.
- Similar states should share learned information.

DQN is not a good fit when the action itself is continuous. Methods such as DDPG or PPO are usually better for that.

## Key Takeaways

- DQN replaces the Q-table with a neural network.
- The network predicts one Q-value per action.
- Experience replay makes training data less correlated.
- A target network makes the learning target more stable.
- DQN is mainly for large state spaces with discrete actions.

---

*Explore DQN implementation and training dynamics in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
