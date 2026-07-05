---
title: Actor-Critic - Best of Both Worlds
author: fanghao
date: 2024-03-01 10:00:00 +0200
categories: [AI, Reinforcement Learning, Actor-Critic]
tags: [actor-critic, advantage, policy-gradients, value-functions]
---

# Actor-Critic: Best of Both Worlds

Actor-Critic methods combine two ideas. The actor decides what to do. The critic evaluates how good that decision was.

This is useful because learning a policy directly can be noisy, while learning only values does not directly optimize the action policy.

## Actor and Critic

The **actor** is the policy. It takes a state and chooses an action.

```text
state -> actor -> action
```

The **critic** estimates value. It tells us whether the current state or action looks good.

```text
state -> critic -> value
```

Together:

1. The actor chooses an action.
2. The environment returns a reward and next state.
3. The critic computes a learning signal.
4. The actor updates its policy using that signal.

## The TD Error

A common critic signal is the TD error:

```text
delta = reward + gamma * V(next_state) - V(state)
```

If `delta` is positive, the outcome was better than expected. The actor should make that action more likely.

If `delta` is negative, the outcome was worse than expected. The actor should make that action less likely.

## A Simple Learning Step

```python
action = actor.choose_action(state)
next_state, reward, done, info = env.step(action)

td_error = critic.learn(state, reward, next_state)
actor.learn(state, action, td_error)
```

This shows the division of work clearly. The critic learns to evaluate. The actor learns to act.

## Why Not Use Only One?

Pure policy gradient methods can work, but their updates often have high variance.

Value-based methods such as DQN can be efficient, but they are mainly natural for discrete actions.

Actor-Critic sits between them. It can learn a policy directly while using value estimates to make updates less noisy.

## A CartPole Example

In CartPole, the actor chooses left or right. The critic estimates whether the current cart and pole state is promising.

At first, both are poor. The actor chooses weak actions, and the critic gives inaccurate values.

After training, the critic becomes better at predicting which states are stable. The actor uses that signal to choose better balancing actions.

## Key Takeaways

- The actor chooses actions.
- The critic evaluates states or actions.
- The TD error can guide the actor update.
- Actor-Critic methods reduce some of the noise in policy learning.
- Many modern RL algorithms are built from this idea.

---

*Implement and experiment with Actor-Critic variants in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
