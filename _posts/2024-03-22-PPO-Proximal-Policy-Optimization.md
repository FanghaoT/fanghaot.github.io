---
title: PPO - Proximal Policy Optimization
author: fanghao
date: 2024-03-22 10:00:00 +0200
categories: [AI, Reinforcement Learning, Policy Optimization]
tags: [ppo, policy-optimization, clipped-surrogate, modern-rl]
---

# PPO: Proximal Policy Optimization

Policy gradient methods update the policy directly. This is flexible, but large updates can damage a policy very quickly.

PPO tries to solve this with a simple rule: improve the policy, but do not move too far in one update.

## The Problem

Suppose the current policy is working reasonably well. If one update changes it too much, the agent may suddenly perform much worse.

So we want a conservative policy update:

```text
large enough to learn
small enough to stay stable
```

PPO is built around this balance.

## The Probability Ratio

PPO compares the new policy with the old policy:

```text
ratio = new_probability(action) / old_probability(action)
```

If the ratio is greater than 1, the new policy makes that action more likely.

If the ratio is less than 1, the new policy makes that action less likely.

## The Clipping Idea

PPO clips the ratio so the policy does not change too much:

```text
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
```

If `epsilon = 0.2`, then the ratio is limited to the range:

```text
0.8 to 1.2
```

This does not freeze the policy. It simply discourages overly large updates.

## Actor-Critic Structure

PPO is usually implemented as an Actor-Critic method.

- The actor is the policy.
- The critic estimates value.
- The advantage tells the actor whether an action was better or worse than expected.

The actor update uses the clipped objective. The critic learns to predict returns.

## A Simple Training Outline

```python
for iteration in range(num_iterations):
    trajectories = collect_experience(policy)
    advantages = estimate_advantages(trajectories)

    for epoch in range(update_epochs):
        update_actor_with_clipped_objective(trajectories, advantages)
        update_critic(trajectories)
```

PPO often reuses the same collected batch for several update epochs. The clipping keeps these repeated updates from moving the policy too far.

## Why PPO Is Practical

PPO is popular because it is relatively simple compared with older trust-region methods. It works with discrete or continuous actions and is usually easier to tune than many earlier policy-gradient algorithms.

It is not magic. It still needs enough data, reasonable rewards, and careful monitoring. But it is a strong general-purpose baseline.

## When to Use PPO

Use PPO when:

- You want a stable policy-gradient method.
- The action space may be discrete or continuous.
- You can collect batches of experience.
- You want a reliable baseline before trying more complex methods.

For small tabular problems, PPO is unnecessary. For simple discrete tasks, DQN may be easier.

## Key Takeaways

- PPO updates the policy directly.
- It uses clipping to avoid overly large policy changes.
- It is usually implemented with an actor and a critic.
- It can reuse collected data for multiple update epochs.
- It is a practical general-purpose RL algorithm.

---

*Master PPO implementation and tuning in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
