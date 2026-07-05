---
title: DQN Improvements - Double, Dueling, and Prioritized Experience
author: fanghao
date: 2024-02-09 10:00:00 +0200
categories: [AI, Reinforcement Learning, Deep Learning]
tags: [double-dqn, dueling-dqn, prioritized-replay, dqn-improvements]
---

# DQN Improvements: Double, Dueling, and Prioritized Experience

DQN is a useful starting point for deep reinforcement learning, but it has some common weaknesses. Three popular improvements are Double DQN, Dueling DQN, and Prioritized Experience Replay.

Each one fixes a different part of the original DQN.

## Double DQN: Reduce Overestimation

DQN uses a max operation when building the target:

```python
q_target = r + gamma * max(Q(next_state))
```

This can overestimate action values. The network may choose an action because of noise, then also use that noisy value as the target.

Double DQN separates action selection from action evaluation:

```python
best_action = argmax(online_network(next_state))
q_target = r + gamma * target_network(next_state)[best_action]
```

The online network chooses the action. The target network evaluates it.

This small change usually gives more realistic Q-values.

## Dueling DQN: Separate State Value and Action Advantage

Sometimes the value of a state matters more than the exact action. For example, if the agent is already in a safe and useful position, several actions may be similarly good.

Dueling DQN splits the network into two parts:

- `V(s)`: how good the state is.
- `A(s,a)`: how much better one action is than the others.

Then it combines them into Q-values:

```text
Q(s,a) = V(s) + A(s,a)
```

In practice, the advantage values are normalized before combining. The main idea is simple: learn the value of the state and the value of each action separately.

## Prioritized Experience Replay: Learn More from Useful Transitions

Standard replay samples experiences randomly. But not every experience is equally useful.

Prioritized replay samples important experiences more often. A common signal is TD error:

```text
priority = abs(target - prediction)
```

If the network was very wrong about a transition, that transition may be worth revisiting.

Because prioritized sampling changes the data distribution, implementations usually add importance-sampling weights to reduce bias.

## How They Fit Together

These improvements can be combined:

- Double DQN gives better targets.
- Dueling DQN gives a better network structure.
- Prioritized replay gives better training samples.

Together, they make DQN more stable and sample efficient.

## When Each One Helps

Use Double DQN when Q-values seem too optimistic.

Use Dueling DQN when many actions have similar effects in some states.

Use Prioritized Experience Replay when useful experiences are rare or the reward is sparse.

## Key Takeaways

- Double DQN reduces overestimation.
- Dueling DQN separates state value from action advantage.
- Prioritized replay samples more informative transitions more often.
- These methods improve DQN without changing the basic RL problem.
- They are useful steps before moving to more advanced deep RL algorithms.

---

*Compare all DQN variants and their performance in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
