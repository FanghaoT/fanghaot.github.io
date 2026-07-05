---
title: OpenAI Gym - Your RL Playground
author: fanghao
date: 2024-02-16 10:00:00 +0200
categories: [AI, Reinforcement Learning, Environments]
tags: [openai-gym, cartpole, mountaincar, environments, classic-control]
---

# OpenAI Gym: Your RL Playground

Reinforcement learning needs environments. An environment gives the agent observations, receives actions, and returns rewards.

OpenAI Gym became popular because it gave many RL tasks the same simple interface.

## The Basic Loop

Most Gym examples follow this pattern:

```python
import gym

env = gym.make("CartPole-v1")
observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
```

The important methods are:

- `reset()`: start a new episode.
- `step(action)`: apply one action.
- `observation_space`: what the agent can observe.
- `action_space`: what the agent can do.

This common interface makes it easier to test different algorithms on different tasks.

## CartPole

CartPole is a common first environment.

The goal is to keep a pole balanced on a moving cart. The agent can push the cart left or right.

The observation contains four values:

- Cart position.
- Cart velocity.
- Pole angle.
- Pole angular velocity.

The reward is simple: the agent gets a positive reward for each time step the pole stays upright.

CartPole is useful because it is easy to understand, but still requires feedback control.

## MountainCar

MountainCar looks simple, but it is harder than CartPole.

The car starts in a valley and must reach a flag on the hill. Its engine is too weak to drive straight up, so it has to move back and forth to build momentum.

The observation contains:

- Car position.
- Car velocity.

The actions are:

- Push left.
- Do nothing.
- Push right.

MountainCar is useful for learning about sparse rewards. Many actions look bad at first, but they are needed to build momentum later.

## Why Gym Helps

Gym separates the algorithm from the environment.

This means the same DQN code can often be tested on CartPole, MountainCar, and other tasks with small changes.

It also makes debugging easier. If an algorithm fails on a simple environment, the problem is probably in the algorithm or hyperparameters, not in a complex custom environment.

## A Good Workflow

When testing a new RL algorithm, I would usually start with:

1. Run a random agent.
2. Print observations, actions, and rewards.
3. Train on a simple environment such as CartPole.
4. Move to a harder environment such as MountainCar.
5. Only then try a custom environment.

This keeps the learning process manageable.

## Key Takeaways

- Gym provides a standard interface for RL environments.
- `reset()` starts an episode and `step()` advances it.
- CartPole is good for basic control experiments.
- MountainCar is good for sparse-reward experiments.
- A standard environment helps separate algorithm problems from environment problems.

---

*Try CartPole and MountainCar implementations with various DQN variants in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
