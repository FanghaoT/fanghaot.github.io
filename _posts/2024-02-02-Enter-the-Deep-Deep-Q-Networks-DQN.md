---
title: Enter the Deep - Deep Q-Networks (DQN)
author: fanghao
date: 2024-02-02 10:00:00 +0200
categories: [AI, Reinforcement Learning, Deep Learning]
tags: [dqn, deep-q-networks, function-approximation, experience-replay]
---

# Enter the Deep: Deep Q-Networks (DQN)

Welcome to a pivotal moment in our RL journey! Today we make the leap from tabular methods to function approximation with Deep Q-Networks (DQN). This breakthrough, introduced by DeepMind in 2015, revolutionized RL by enabling agents to tackle problems with massive state spaces.

## The Tabular Limitation

Our maze adventures were educational, but consider these real-world challenges:
- **Atari games**: 210×160×3 pixel screens = ~100,000 possible states per frame
- **Go**: ~10^170 possible board positions  
- **Autonomous driving**: Continuous sensor inputs

Tabular methods simply cannot scale to these problems. We need **function approximation**.

## From Tables to Neural Networks

Instead of storing Q-values in a table, DQN uses a neural network to approximate the Q-function:

```
Q-Table: Q(s,a) → lookup value
DQN: Q(s,a;θ) → neural network with parameters θ
```

### The Deep Q-Network Architecture

```python
def _build_net(self):
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
    self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
    
    # Evaluation network
    with tf.variable_scope('eval_net'):
        e1 = tf.layers.dense(self.s, 20, tf.nn.relu, name='e1')
        self.q_eval = tf.layers.dense(e1, self.n_actions, name='q')
    
    # Target network  
    with tf.variable_scope('target_net'):
        t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, name='t1')
        self.q_next = tf.layers.dense(t1, self.n_actions, name='t')
```

## Key DQN Innovations

### 1. Experience Replay

**Problem**: Sequential RL data is highly correlated, leading to inefficient learning.

**Solution**: Store experiences in a replay buffer and sample randomly for training.

```python
# Store experience
self.memory[self.memory_counter % self.memory_size, :] = s, a, r, s_

# Sample random batch for training  
batch_memory = self.memory[sample_index, :]
q_next = self.sess.run(self.q_next, {self.s_: batch_memory[:, -self.n_features:]})
q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
```

**Benefits:**
- Breaks correlation between consecutive samples
- Enables multiple updates from single experiences
- Improves sample efficiency dramatically

### 2. Fixed Target Networks

**Problem**: Using the same network for both current Q-values and targets creates instability.

**Solution**: Maintain separate target network, updated periodically.

```python
# Periodic target network update
if self.learn_step_counter % self.replace_target_iter == 0:
    self.sess.run(self.target_replace_op)
```

**Benefits:**
- Stabilizes learning by providing consistent targets
- Reduces harmful correlations between updates
- Enables more stable convergence

### 3. Clipped Rewards

For Atari games, rewards are clipped to [-1, +1] to:
- Normalize learning across different games
- Prevent large rewards from dominating gradient updates
- Improve training stability

## DQN Training Process

### 1. Experience Collection
```python
def choose_action(self, observation):
    if np.random.uniform() < self.epsilon:
        # Forward pass through network
        actions_value = self.sess.run(self.q_eval, {self.s: observation[np.newaxis, :]})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, self.n_actions)
    return action
```

### 2. Memory Storage
Experiences (s, a, r, s') are stored in circular buffer.

### 3. Batch Learning
```python
def learn(self):
    # Sample batch from memory
    batch_memory = self.memory[sample_index, :]
    
    # Compute targets using target network
    q_next = self.sess.run(self.q_next, feed_dict)
    q_target = batch_memory[:, self.n_features + 1] + self.gamma * np.max(q_next, axis=1)
    
    # Train evaluation network
    self.sess.run(self._train_op, feed_dict)
```

## Handling Continuous State Spaces

DQN excels with continuous or high-dimensional states:

### State Preprocessing
- **Normalization**: Scale inputs to reasonable ranges
- **Feature extraction**: Convert raw pixels to meaningful representations
- **Frame stacking**: Use sequence of frames for temporal information

### Network Design Considerations
- **Input layer**: Match state dimensionality
- **Hidden layers**: Sufficient capacity for complex functions
- **Output layer**: One neuron per discrete action

## DQN vs Q-Learning Comparison

| Aspect | Q-Learning | DQN |
|--------|------------|-----|
| State space | Small, discrete | Large, continuous |
| Storage | Q-table | Neural network |
| Updates | Single (s,a) pair | Batch of experiences |
| Generalization | None | Across similar states |
| Sample efficiency | Low | Higher (with replay) |

## Common Challenges and Solutions

### 1. Overestimation Bias
Neural networks tend to overestimate Q-values, leading to suboptimal policies.

### 2. Training Instability
Deep networks can be unstable with RL updates. Target networks help, but careful hyperparameter tuning is crucial.

### 3. Sample Efficiency
Despite improvements, DQN still requires many samples. Advanced techniques (prioritized replay, etc.) help.

### 4. Exploration
ε-greedy works but isn't optimal for complex environments. Advanced exploration strategies improve performance.

## Practical Implementation Tips

### Hyperparameter Choices
- **Learning rate**: Start with 0.001, adjust based on stability
- **Batch size**: 32-64 for most problems
- **Target update frequency**: Every 300-1000 steps
- **Memory size**: 10,000-1,000,000 depending on problem

### Network Architecture
- Start simple (2-3 hidden layers)
- Use ReLU activations
- Consider state-specific preprocessing

### Training Monitoring
- Track average Q-values over time
- Monitor loss convergence
- Watch for sudden jumps (indicates instability)

## What's Next?

DQN opened the floodgates for deep RL, but it has limitations. Our next post explores three major DQN improvements: Double DQN (addressing overestimation), Dueling DQN (better value estimation), and Prioritized Experience Replay (smarter sampling).

Coming up: "DQN Improvements: Double, Dueling, and Prioritized Experience"!

---

*Explore DQN implementation and training dynamics in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 