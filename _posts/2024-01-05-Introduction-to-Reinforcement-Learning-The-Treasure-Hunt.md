---
title: Introduction to Reinforcement Learning - The Treasure Hunt
author: fanghao
date: 2024-01-05 10:00:00 +0200
categories: [Reinforcement Learning]
tags: [reinforcement-learning, q-learning, tutorial]
---

# Introduction to Reinforcement Learning: The Treasure Hunt

Welcome to my Reinforcement Learning tutorial! Today, we'll dive into the fascinating world of RL using a simple yet powerful example: an agent searching for treasure in a 1D world.

## What is Reinforcement Learning?

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, there's no teacher providing correct answers. Instead, the agent learns from rewards and punishments, much like how we learn in real life.

### Key Concepts

- **Agent**: The learner or decision maker
- **Environment**: The world the agent interacts with  
- **Actions**: What the agent can do
- **States**: Situations the agent can be in
- **Rewards**: Feedback from the environment
- **Policy**: The agent's strategy for choosing actions

## The Treasure Hunt Example

Our first example features an agent `o` on the left side of a 1-dimensional world, with treasure `T` on the rightmost location:

```
[ o |   |   |   |   | T ]
  0   1   2   3   4   5
```

The agent can move `left` or `right`, and receives a reward of +1 when it reaches the treasure.

## Q-Learning: Our First Algorithm

Q-Learning is a model-free algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It uses a Q-table to store the expected future rewards for each state-action pair.

### The Q-Learning Algorithm

The formal Q-learning algorithm follows these steps:

```
Initialize Q(s,a) arbitrarily
Repeat (for each episode):
    Initialize s
    Repeat (for each step of episode):
        Choose a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
    until s is terminal
```

### The Q-Value Update Rule

The heart of Q-learning is the update equation:

**Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]**

Let's break this down:
- **Q(s,a)**: Current Q-value for state s and action a
- **α**: Learning rate (how much we update based on new information)
- **r**: Immediate reward received after taking action a
- **γ**: Discount factor (how much we value future rewards)
- **max_a' Q(s',a')**: Maximum Q-value for the next state s'
- **[r + γ max_a' Q(s',a') - Q(s,a)]**: This is the "temporal difference error"

### How the Update Works

1. **Temporal Difference Target**: `r + γ max_a' Q(s',a')` represents our estimate of the total future reward
2. **Current Estimate**: `Q(s,a)` is what we currently think this state-action pair is worth
3. **Error**: The difference between our target and current estimate shows how wrong we were
4. **Learning**: We adjust our Q-value by a fraction (α) of this error

### Action Selection: ε-Greedy Policy

The agent chooses actions using an ε-greedy policy:
- With probability (1-ε): Choose the **best** action (exploit)
- With probability ε: Choose a **random** action (explore)

This balances learning about the environment (exploration) with using current knowledge (exploitation).

### Implementation Highlights

Our implementation uses these key parameters:
- `EPSILON = 0.9`: Exploration vs exploitation balance (90% exploitation, 10% exploration)
- `ALPHA = 0.1`: Learning rate (how quickly we update our beliefs)
- `GAMMA = 0.9`: Discount factor (future rewards are worth 90% of immediate rewards)

## Step-by-Step Example: Q-Value Updates in Action

Let's walk through a concrete example to see how Q-values are updated during the treasure hunt. We'll use our parameters: α = 0.1, γ = 0.9.

### Initial Setup

Our Q-table starts with all zeros:

| State | Left | Right |
|-------|------|-------|
| 0     | 0.0  | 0.0   |
| 1     | 0.0  | 0.0   |
| 2     | 0.0  | 0.0   |
| 3     | 0.0  | 0.0   |
| 4     | 0.0  | 0.0   |
| 5     | 0.0  | 0.0   |

### Episode 1: First Discovery

**Step 1**: Agent at state 0, chooses action "right"
- Current state: s = 0
- Action: a = right  
- Next state: s' = 1
- Reward: r = 0 (no treasure yet)

**Q-Update Calculation**:
```
Q(0,right) ← Q(0,right) + α[r + γ max Q(1,a') - Q(0,right)]
Q(0,right) ← 0.0 + 0.1[0 + 0.9 × max(0.0, 0.0) - 0.0]
Q(0,right) ← 0.0 + 0.1[0 + 0.9 × 0.0 - 0.0]
Q(0,right) ← 0.0 + 0.1[0]
Q(0,right) ← 0.0
```

No change yet because we haven't found any rewards to propagate.

**Step 2**: Agent at state 1, chooses action "right"
- s = 1, a = right, s' = 2, r = 0
- Q(1,right) ← 0.0 (same calculation, still no reward)

**Steps 3-5**: Continue moving right...
- State 2 → 3 → 4 → 5

**Step 6**: Agent reaches treasure at state 5!
- s = 4, a = right, s' = 5, r = 1 (treasure found!)

**Q-Update Calculation**:
```
Q(4,right) ← Q(4,right) + α[r + γ max Q(5,a') - Q(4,right)]
Q(4,right) ← 0.0 + 0.1[1 + 0.9 × max(0.0, 0.0) - 0.0]
Q(4,right) ← 0.0 + 0.1[1 + 0]
Q(4,right) ← 0.1
```

**Updated Q-table after Episode 1**:

| State | Left | Right |
|-------|------|-------|
| 0     | 0.0  | 0.0   |
| 1     | 0.0  | 0.0   |
| 2     | 0.0  | 0.0   |
| 3     | 0.0  | 0.0   |
| 4     | 0.0  | **0.1** |
| 5     | 0.0  | 0.0   |

### Episode 2: Value Propagation

The agent takes the same path again. Now when it reaches state 3:

**Step**: Agent at state 3, chooses action "right"
- s = 3, a = right, s' = 4, r = 0

**Q-Update Calculation**:
```
Q(3,right) ← Q(3,right) + α[r + γ max Q(4,a') - Q(3,right)]
Q(3,right) ← 0.0 + 0.1[0 + 0.9 × max(0.0, 0.1) - 0.0]
Q(3,right) ← 0.0 + 0.1[0 + 0.9 × 0.1]
Q(3,right) ← 0.0 + 0.1[0.09]
Q(3,right) ← 0.009
```

When it reaches state 4 again:
```
Q(4,right) ← 0.1 + 0.1[1 + 0.9 × 0.0 - 0.1]
Q(4,right) ← 0.1 + 0.1[0.9]
Q(4,right) ← 0.19
```

**Updated Q-table after Episode 2**:

| State | Left | Right |
|-------|------|-------|
| 0     | 0.0  | 0.0   |
| 1     | 0.0  | 0.0   |
| 2     | 0.0  | 0.0   |
| 3     | 0.0  | **0.009** |
| 4     | 0.0  | **0.19** |
| 5     | 0.0  | 0.0   |

### Key Observations

1. **Reward Propagation**: The reward from finding treasure (state 5) propagates backward through the Q-table
2. **Discounting Effect**: Values decrease as we move away from the reward source (0.1 → 0.009)
3. **Cumulative Learning**: Q(4,right) increases from 0.1 to 0.19 with more experience
4. **Convergence Direction**: Values will eventually converge to optimal estimates

After many episodes, the Q-table will show:
- **Right actions**: Higher values (leading toward treasure)
- **Left actions**: Lower values (leading away from treasure)
- **Gradient effect**: Values increase as we get closer to the treasure

This demonstrates how the Q-learning algorithm automatically discovers which actions lead to rewards, even with delayed feedback!

## Key Takeaways

- RL agents learn through trial and error
- Q-Learning builds a table of state-action values
- The ε-greedy policy balances exploration and exploitation
- The agent gradually improves its strategy over episodes

---

*Code for this tutorial is available in the [Reinforcement-Learning-PyTorch](https://github.com/FanghaoT/Reinforcement-Learning-PyTorch)* 