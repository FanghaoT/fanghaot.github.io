---
title: SARSA(λ) - Adding Memory with Eligibility Traces
author: fanghao
date: 2024-01-26 10:00:00 +0200
categories: [AI, Reinforcement Learning]
tags: [sarsa-lambda, eligibility-traces, temporal-credit, multi-step]
---

# SARSA(λ): Adding Memory with Eligibility Traces

Welcome to one of RL's most elegant solutions to the temporal credit assignment problem! Today we explore SARSA(λ), which extends our familiar SARSA algorithm with eligibility traces – a mechanism that bridges the gap between temporal difference and Monte Carlo methods.

## The Credit Assignment Challenge

Consider this scenario: An agent makes a great move early in an episode, followed by several neutral moves, then receives a reward. Traditional SARSA only updates the last state-action pair immediately. How do we credit that brilliant early move?

**The problem:** Information flows backward one step at a time, creating slow learning for actions distant from rewards.

**The solution:** Eligibility traces maintain a "memory" of recently visited state-action pairs and update them all when rewards arrive.

## Introducing Eligibility Traces

Think of eligibility traces as breadcrumbs marking your path:
- Each state-action pair gets an "eligibility" score
- Recent pairs have high eligibility, older ones decay
- When reward arrives, all eligible pairs get updated

### Mathematical Foundation

For each state-action pair, we maintain an eligibility trace e(s,a):

```
e(s,a) = γλe(s,a) + 1  (if s,a visited this step)
e(s,a) = γλe(s,a)      (otherwise)
```

Where:
- **λ** (lambda): Trace decay parameter (0 ≤ λ ≤ 1)
- **γ** (gamma): Discount factor

### The Update Rule

SARSA(λ) updates ALL state-action pairs simultaneously:

```
δ = r + γQ(s',a') - Q(s,a)  (TD error)
For all s,a:
    Q(s,a) += αδe(s,a)
    e(s,a) *= γλ
```

## Understanding Lambda (λ)

The λ parameter controls the trace decay rate:

- **λ = 0**: Pure TD learning (standard SARSA)
- **λ = 1**: Monte Carlo-like updates (no decay)
- **0 < λ < 1**: Blend of TD and MC approaches

### Lambda's Effect on Learning

**High λ (0.8-0.95):**
- Stronger credit to distant actions
- Faster learning but more variance
- Better for sparse reward environments

**Low λ (0.1-0.3):**
- Focus on recent actions
- More stable but slower learning
- Better for dense reward environments

## Implementation Deep Dive

### Eligibility Trace Management

```python
class SarsaLambdaTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, 
                 e_greedy=0.9, trace_decay=0.9):
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()
        
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, a_] if s_ != 'terminal' else r
        error = q_target - q_predict
        
        # Update eligibility trace for current state-action
        self.eligibility_trace.loc[s, a] += 1
        
        # Update all Q-values and decay traces
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_
```

### Trace Reset Strategies

**Replacing traces:** Set e(s,a) = 1 when visiting (s,a)
**Accumulating traces:** Add 1 to e(s,a) when visiting (s,a)

Most implementations use replacing traces for better empirical performance.

## Behavioral Advantages

### Faster Learning
- Multi-step updates accelerate value propagation
- Good early moves get credit immediately when rewards arrive
- Particularly effective in sparse reward environments

### Better Exploration Credit
- Actions that lead to informative states get reinforced
- Reduces the need for extensive repeated experiences
- More sample efficient than standard SARSA

### Temporal Smoothing
- Smooths out noise in individual updates
- Creates more stable learning trajectories
- Reduces sensitivity to individual bad experiences

## Maze Navigation with SARSA(λ)

In our maze environment, SARSA(λ) demonstrates:

### Episode-Level Learning
- Finding the goal reinforces the entire successful path
- Previous dead-ends get negatively reinforced across multiple states
- Faster convergence to good policies

### Path Optimization
- Earlier path segments get updated based on later success
- More efficient exploration of the state space
- Better handling of delayed rewards

### Trace Visualization
Watch eligibility traces during learning:
1. Traces build up along the current path
2. Successful goal reaching updates the entire trail
3. Traces decay over time, maintaining recency bias

## Computational Considerations

### Memory Requirements
- Need to store eligibility trace for every state-action pair
- Memory grows with state space size
- Trade-off: performance vs. memory usage

### Update Efficiency
- Updates all traced states simultaneously
- Can be computationally expensive for large state spaces
- Often worth the cost for the learning speedup

## Practical Tips

### Lambda Selection
- Start with λ = 0.9 for most problems
- Reduce λ if learning is too unstable
- Increase λ for very sparse reward problems

### Trace Management
- Clear traces at episode boundaries (for episodic tasks)
- Consider trace cutoff thresholds for efficiency
- Monitor trace magnitudes during debugging

## What's Next?

We've covered the foundations of tabular RL methods. Our next post marks a major transition as we enter the world of deep reinforcement learning with Deep Q-Networks (DQN), where neural networks replace our Q-tables to handle vastly larger state spaces.

Coming up: "Enter the Deep: Deep Q-Networks (DQN)"!

---

*Explore SARSA(λ) implementation and trace visualizations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 