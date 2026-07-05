---
title: SARSA - On-Policy Learning in Action
author: fanghao
date: 2024-01-19 10:00:00 +0200
categories: [AI, Reinforcement Learning]
tags: [sarsa, on-policy, off-policy, temporal-difference]
---

# SARSA: On-Policy Learning in Action

SARSA is very close to Q-Learning, but it updates values in a slightly different way. Q-Learning looks at the best possible next action. SARSA looks at the action the agent actually takes next.

This small change makes SARSA useful when the behavior during learning matters.

## Q-Learning vs SARSA: The Key Difference

Both methods learn a Q-table. The difference is the target used in the update.

- **Q-Learning** uses the best next action.
- **SARSA** uses the actual next action chosen by the current policy.

So Q-Learning asks: "What is the best thing I could do next?"

SARSA asks: "What did I actually decide to do next?"

## The SARSA Algorithm

SARSA gets its name from the sequence:

**State -> Action -> Reward -> Next State -> Next Action**

The update rule is:

**Q(S,A) <- Q(S,A) + alpha [R + gamma Q(S',A') - Q(S,A)]**

1. Initialize the Q-table.
2. Choose action A from state S using the current policy.
3. Take action A, observe reward R and next state S'.
4. Choose next action A' from S' using the same policy.
5. Update Q(S,A) using R, S', and A'.
6. Move to S' and A'.
7. Repeat until the episode ends.

The important part is step 4. SARSA chooses the next action before updating the current Q-value.

## The Code Difference

```python
# Q-Learning: use the best next action
q_target = r + self.gamma * self.q_table.loc[s_, :].max()

# SARSA: use the action actually chosen next
q_target = r + self.gamma * self.q_table.loc[s_, a_]
```

This is the whole practical difference. The rest of the algorithm can look almost the same.

## A Simple Example: Cliff Walking

Imagine this small world:

```text
[S][  ][  ][  ][G]
[C][C][C][C][C]
```

`S` is the start, `G` is the goal, and `C` is a cliff with a large negative reward.

Q-Learning tends to learn the shortest path near the cliff, because it evaluates the best possible next action. If exploration is still happening, the agent may sometimes step into the cliff.

SARSA is more conservative. Since it updates using the action actually taken during exploration, risky exploration affects the Q-values. The learned path may be longer, but safer.

## When to Choose SARSA vs Q-Learning

Use SARSA when:

- The agent is learning while acting in the real environment.
- Exploration can cause bad outcomes.
- You want the learned policy to include the effect of exploration.
- A safer policy is more important than the shortest possible path.

Use Q-Learning when:

- You want to estimate the optimal policy.
- Exploration mistakes are acceptable.
- You are learning in simulation or from logged experience.
- The final performance matters more than behavior during learning.

## Key Takeaways

- SARSA is an on-policy method.
- It updates Q-values using the next action the agent actually chooses.
- Q-Learning uses the best possible next action instead.
- SARSA often learns more cautious behavior.
- The difference is small in code, but important in behavior.

---

*Compare SARSA and Q-Learning implementations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)*
