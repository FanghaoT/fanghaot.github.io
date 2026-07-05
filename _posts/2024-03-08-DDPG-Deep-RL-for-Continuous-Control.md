---
title: DDPG - Deep RL for Continuous Control
author: fanghao
date: 2024-03-08 10:00:00 +0200
categories: [AI, Reinforcement Learning, Continuous Control]
tags: [ddpg, continuous-control, deterministic-policy, robotics]
---

# DDPG: Deep RL for Continuous Control

DQN works with discrete actions. For example, an agent may choose left or right.

Many control problems need continuous actions instead. A robot joint may need a torque value such as `0.37`, not just action `0` or action `1`.

DDPG is an Actor-Critic method designed for this kind of continuous control.

## The Continuous Action Problem

If the action is continuous, we cannot simply compute:

```text
max Q(state, action)
```

There are infinitely many possible actions.

DDPG solves this by using an actor network to output the action directly.

```text
state -> actor -> continuous action
```

The critic then evaluates that action:

```text
state + action -> critic -> Q-value
```

## The Two Networks

DDPG has two main networks:

- **Actor**: chooses a continuous action.
- **Critic**: estimates how good that action is.

The actor is deterministic. For the same state, it gives the same action unless we add exploration noise.

```python
action = actor(state)
q_value = critic(state, action)
```

The actor is trained to choose actions that the critic scores highly.

## Replay Buffer and Target Networks

DDPG borrows two ideas from DQN:

- Experience replay.
- Target networks.

Experience replay stores transitions:

```text
(state, action, reward, next_state)
```

Target networks make the learning target more stable. DDPG usually updates target networks slowly:

```text
target <- tau * online + (1 - tau) * target
```

This is called a soft update.

## Exploration

Because the actor is deterministic, exploration does not happen automatically.

DDPG usually adds noise to the action:

```python
action = actor(state) + noise
```

At the beginning, the noise is larger. Later, it can be reduced as the policy improves.

## Example: Pendulum

In the Pendulum environment, the agent controls torque. The goal is to swing the pendulum upright and keep it balanced.

This is a natural DDPG problem because the action is continuous:

```text
action = torque value
```

A discrete method would need to split torque into bins. DDPG can output a smooth torque value directly.

## When to Use DDPG

Use DDPG when:

- Actions are continuous.
- You need precise control.
- You can train in simulation.
- A deterministic policy is acceptable.

Avoid DDPG when the action space is small and discrete. DQN is usually simpler there.

## Key Takeaways

- DDPG is for continuous action spaces.
- It uses an actor to output actions and a critic to score them.
- It borrows replay buffers and target networks from DQN.
- Exploration is added as noise on top of the actor's action.
- It is useful for control tasks such as pendulum balancing or robotics simulation.

---

*Master continuous control with DDPG implementations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
