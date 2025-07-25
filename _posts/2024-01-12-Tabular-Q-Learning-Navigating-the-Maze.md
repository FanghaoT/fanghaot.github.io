---
title: Tabular Q-Learning - Navigating the Maze
author: fanghao
date: 2024-01-12 10:00:00 +0200
categories: [Reinforcement Learning]
tags: [q-learning, maze, tabular-methods]
---

# Tabular Q-Learning: Navigating the Maze

Building on our treasure hunt introduction, today we'll explore Q-Learning in a more complex environment: a maze! This is where RL truly starts to shine, as our agent must learn to navigate obstacles and find optimal paths.

## From Linear to Grid World

While our 1D treasure hunt was instructive, real-world problems are rarely so simple. In our maze environment, the agent faces:

- **Multiple paths** to the goal
- **Obstacles** to avoid (walls)
- **Larger state space** requiring dynamic Q-table expansion
- **Spatial reasoning** challenges

## The Maze Environment

Our maze is a grid world where:
- 🔴 Red rectangle: Our agent
- ⚫ Black squares: Walls/obstacles  
- 🟡 Yellow oval: The goal
- ⬜ White squares: Valid paths

The agent can move in four directions: up, down, left, right.

## Q-Learning Implementation Deep Dive

### Dynamic Q-Table Construction

Unlike our fixed 1D world, the maze requires a dynamic Q-table that grows as the agent discovers new states:

```python
def check_state_exist(self, state):
    if state not in self.q_table.index:
        # Append new state to q table
        self.q_table = self.q_table.append(
            pd.Series([0]*len(self.actions), 
                     index=self.q_table.columns, 
                     name=state)
        )
```

### Action Selection Strategy

The ε-greedy policy becomes more interesting in complex environments:

```python
def choose_action(self, observation):
    if np.random.uniform() < self.epsilon:
        # Exploit: choose best known action
        state_action = self.q_table.loc[observation, :]
        action = np.random.choice(
            state_action[state_action == np.max(state_action)].index
        )
    else:
        # Explore: try random action
        action = np.random.choice(self.actions)
```

### Learning Update

The core Q-Learning update remains the same, but now handles more complex state transitions:

```python
def learn(self, s, a, r, s_):
    q_predict = self.q_table.loc[s, a]
    if s_ != 'terminal':
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()
    else:
        q_target = r
    self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
```

## Key Observations

### Exploration Patterns
- Early episodes: Random wandering, hitting walls
- Middle episodes: Discovering viable paths
- Later episodes: Optimizing known routes

### Convergence Behavior
- Q-values stabilize as optimal policy emerges
- Agent learns to avoid walls without explicit programming
- Shorter paths are naturally preferred due to discounting

### The Credit Assignment Problem
The maze demonstrates how Q-Learning solves temporal credit assignment:
- Actions early in a path get credit for eventual success
- The discount factor (γ) ensures immediate rewards matter more

## Practical Insights

**When to use tabular Q-Learning:**
- Discrete, manageable state spaces
- Simple action spaces
- When interpretability matters (you can inspect the Q-table)

**Limitations:**
- Scales poorly with state space size
- No generalization between similar states
- Requires extensive exploration for large spaces

## Interactive Visualization

Watch how the agent's behavior evolves:
1. **Episode 1-10**: Chaotic exploration, frequent wall collisions
2. **Episode 11-50**: Path discovery, reducing random moves
3. **Episode 51+**: Consistent optimal or near-optimal paths


*Complete maze implementation available in the [Reinforcement-Learning-PyTorch](https://github.com/FanghaoT/Reinforcement-Learning-PyTorch)* 