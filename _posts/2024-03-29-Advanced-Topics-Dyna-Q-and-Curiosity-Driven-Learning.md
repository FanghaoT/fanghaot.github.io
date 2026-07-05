---
title: Advanced Topics - Dyna-Q and Curiosity-Driven Learning
author: fanghao
date: 2024-03-29 10:00:00 +0200
categories: [AI, Reinforcement Learning, Advanced Topics]
tags: [dyna-q, model-based-rl, curiosity, intrinsic-motivation, exploration]
---

# Advanced Topics: Dyna-Q and Curiosity-Driven Learning

Most of the previous methods learn from real experience. The agent acts, observes the result, and updates its policy or value function.

This post introduces two ideas that go beyond that basic loop: planning with a learned model, and exploration driven by curiosity.

## Dyna-Q: Learning and Planning Together

Q-Learning only updates from real experience:

```text
state, action, reward, next_state
```

Dyna-Q does one extra thing. It learns a simple model of the environment:

```text
(state, action) -> (next_state, reward)
```

Then it uses this model to create simulated experience for planning.

## The Dyna-Q Loop

```text
1. Take a real action.
2. Update Q-values from the real transition.
3. Store the transition in the model.
4. Sample old transitions from the model.
5. Update Q-values again using simulated transitions.
```

The benefit is sample efficiency. One real experience can lead to several learning updates.

## When Dyna-Q Helps

Dyna-Q is useful when:

- The environment is simple enough to model.
- Real experience is expensive.
- Planning time is available.
- Past transitions are still useful.

It can struggle if the learned model is wrong. Planning with a bad model can reinforce bad assumptions.

## Curiosity-Driven Learning

Some environments have sparse rewards. The agent may wander for a long time without receiving useful feedback.

Curiosity adds an internal reward for discovering something new or surprising.

The basic idea:

```text
external reward + curiosity reward = total reward
```

This encourages the agent to explore even when the environment reward is rare.

## Prediction Error as Curiosity

One common method is to train a model to predict what happens next.

If the prediction is bad, the transition is surprising. The agent receives a curiosity reward.

```text
curiosity reward = prediction error
```

As the model learns familiar states, those states become less interesting. The agent is pushed toward unfamiliar parts of the environment.

## A Simple Example

In a maze, the external reward may only appear at the goal.

Without curiosity, the agent may never find the goal.

With curiosity, unexplored corridors become temporarily rewarding. This can help the agent discover the goal sooner.

## Dyna-Q vs Curiosity

| Method | Main Idea | Best Use |
|---|---|---|
| Dyna-Q | Learn a model and plan with it | Reusing experience |
| Curiosity | Reward novelty or surprise | Sparse-reward exploration |

They address different problems. Dyna-Q asks, "Can I learn more from what I already saw?" Curiosity asks, "What should I explore next?"

## Key Takeaways

- Dyna-Q combines real learning with planning.
- A learned model can generate simulated experience.
- Curiosity gives the agent an internal reason to explore.
- Prediction error is one way to measure novelty.
- These ideas are useful when experience or rewards are limited.

---

*Experiment with advanced RL techniques in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
