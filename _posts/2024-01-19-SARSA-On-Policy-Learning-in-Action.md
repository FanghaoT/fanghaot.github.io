---
title: SARSA - On-Policy Learning in Action  
author: fanghao
date: 2024-01-19 10:00:00 +0200
categories: [AI, Reinforcement Learning]
tags: [sarsa, on-policy, off-policy, temporal-difference]
---

# SARSA: On-Policy Learning in Action

Welcome back to our RL journey! Today we explore SARSA (State-Action-Reward-State-Action), a fascinating alternative to Q-Learning that highlights one of RL's most important distinctions: on-policy vs off-policy learning.

## Q-Learning vs SARSA: The Key Difference

While both are temporal difference methods, they differ in a crucial way:

- **Q-Learning (Off-Policy)**: Learns the optimal policy regardless of the policy being followed
- **SARSA (On-Policy)**: Learns the value of the policy being followed

This seemingly small difference has profound implications for behavior and applications.

## The SARSA Algorithm

SARSA gets its name from the sequence: **S**tate → **A**ction → **R**eward → **S**tate → **A**ction

### Algorithm Steps:
1. Initialize Q(s,a) arbitrarily
2. Choose action A from state S using policy derived from Q (e.g., ε-greedy)
3. Take action A, observe reward R and next state S'
4. Choose next action A' from S' using same policy
5. Update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
6. S ← S', A ← A'
7. Repeat until S is terminal

### Key Implementation Difference

```python
# Q-Learning Update (Off-Policy)
q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # Uses max action

# SARSA Update (On-Policy)  
q_target = r + self.gamma * self.q_table.loc[s_, a_]  # Uses actual next action
```

## Behavioral Differences in the Maze

### The Cliff Walking Problem

Consider a maze with a dangerous cliff:
```
[S][  ][  ][  ][G]
[C][C][C][C][C]
```
Where S=Start, G=Goal, C=Cliff (-100 reward)

**Q-Learning behavior:**
- Learns the optimal path (top route)
- During learning, may fall off cliff due to exploration
- Converges to risky but optimal policy

**SARSA behavior:**  
- Learns a safer path (avoiding cliff area)
- Takes exploration into account during learning
- Converges to safer suboptimal policy

### Risk-Sensitive Learning

SARSA naturally becomes more conservative because:
- It considers the exploration policy during updates
- Bad experiences during exploration are reflected in value estimates
- Results in policies that account for uncertainty

## Implementation Insights

### Action Selection Consistency
SARSA requires the same policy for both:
- Choosing actions during episodes
- Selecting the next action A' for updates

### Learning Stability
- SARSA typically converges more smoothly
- Less variance in Q-value updates
- More predictable learning curves

### Environment Interaction
SARSA learns from actual experience sequences, making it:
- More suitable for online learning
- Better for safety-critical applications
- More aligned with the actual policy being executed

## When to Choose SARSA vs Q-Learning

### Use SARSA when:
- Safety matters (avoid dangerous actions during learning)
- Learning online with real consequences
- The exploration policy should influence the learned policy
- You want more conservative behavior

### Use Q-Learning when:
- Seeking the truly optimal policy
- Learning offline from recorded data
- Exploration consequences are manageable
- Maximum performance is the goal

## Practical Example: Maze Navigation

In our maze environment, observe how:

**SARSA agents:**
- Take wider turns around obstacles
- Learn more cautious routes
- Show consistent, predictable paths

**Q-Learning agents:**
- Take tighter, riskier paths
- May occasionally hit walls during execution
- Achieve shorter optimal paths once converged

## The Exploration-Exploitation Trade-off

SARSA makes the exploration-exploitation dilemma explicit:
- High ε: More exploration → More conservative learned policy
- Low ε: Less exploration → Policy closer to optimal

This creates an interesting feedback loop where the exploration strategy directly shapes the final policy.

## What's Next?

Our next post introduces eligibility traces with SARSA(λ), a powerful extension that accelerates learning by updating multiple state-action pairs simultaneously. We'll see how this addresses the temporal credit assignment problem more efficiently.

Coming up: "SARSA(λ): Adding Memory with Eligibility Traces"!

---

*Compare SARSA and Q-Learning implementations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 