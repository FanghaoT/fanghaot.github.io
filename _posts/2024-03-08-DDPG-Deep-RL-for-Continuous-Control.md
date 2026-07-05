---
title: DDPG - Deep RL for Continuous Control
author: fanghao
date: 2024-03-08 10:00:00 +0200
categories: [AI, Reinforcement Learning, Continuous Control]
tags: [ddpg, continuous-control, deterministic-policy, robotics]
---

# DDPG: Deep RL for Continuous Control

Welcome to the world of continuous control! Today we explore Deep Deterministic Policy Gradient (DDPG), a breakthrough algorithm that extends Actor-Critic methods to continuous action spaces. DDPG enables RL agents to perform precise motor control tasks like robotic manipulation, autonomous driving, and flight control.

## The Continuous Control Challenge

### Why Continuous Actions Matter

Most real-world control tasks involve continuous actions:
- **Robotics**: Joint angles, force magnitudes, velocities  
- **Autonomous vehicles**: Steering angle, acceleration, braking force
- **Finance**: Portfolio allocation percentages
- **Game playing**: Movement directions, aim angles

Traditional methods struggle with continuous spaces because:
- **Infinite action spaces**: Can't enumerate all possible actions
- **Function approximation**: Need smooth policy representations
- **Exploration**: How do you explore infinite spaces effectively?

### Previous Approaches and Limitations

**Discretization**: Break continuous space into discrete bins
- Loses precision and smoothness
- Curse of dimensionality with multi-dimensional actions
- Suboptimal performance

**Policy Gradients**: Work but suffer from high variance
- Sample inefficient
- Slow convergence
- Require careful hyperparameter tuning

## Introducing DDPG

DDPG (Lillicrap et al., 2015) combines the best ideas from DQN and Actor-Critic:

### Key Innovations

1. **Deterministic Policy**: π(s) → a (not probability distribution)
2. **Actor-Critic Architecture**: Separate networks for policy and value
3. **Experience Replay**: Reuse past experiences for stable learning
4. **Target Networks**: Separate target networks for stability
5. **Noise-based Exploration**: Add noise to deterministic actions

### Deterministic Policy Gradient Theorem

For deterministic policies, the policy gradient simplifies to:

```
∇_θ J ≈ E[∇_a Q(s,a)|_{a=π(s)} ∇_θ π(s)]
```

This means we can compute gradients by:
1. Getting actions from current policy: a = π(s; θ)
2. Computing Q-function gradient w.r.t. actions: ∇_a Q(s,a)
3. Computing policy gradient w.r.t. parameters: ∇_θ π(s; θ)
4. Chaining them together using chain rule

## DDPG Architecture

### Actor Network (Policy)

```python
def build_actor(self, s, scope, trainable):
    with tf.variable_scope(scope):
        net = tf.layers.dense(s, 400, activation=tf.nn.relu, trainable=trainable)
        net = tf.layers.dense(net, 300, activation=tf.nn.relu, trainable=trainable)
        a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, trainable=trainable)
        return tf.multiply(a, self.a_bound, name='scaled_a')  # Scale to action bounds
```

**Key features**:
- Final layer uses tanh activation (outputs [-1, 1])
- Actions scaled to environment bounds
- Deterministic output (no sampling)

### Critic Network (Q-function)

```python
def build_critic(self, s, a, scope, trainable):
    with tf.variable_scope(scope):
        n_l1 = 400
        w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
        w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
        b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
        net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
        
        net = tf.layers.dense(net, 300, activation=tf.nn.relu, trainable=trainable)
        q = tf.layers.dense(net, 1, trainable=trainable)
        return q
```

**Key features**:
- Takes both state and action as input
- Actions introduced in first hidden layer (not concatenated at input)
- Outputs single Q-value

### Target Networks

```python
# Soft target updates
self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                    for t, e in zip(self.at_params + self.ct_params, 
                                   self.ae_params + self.ce_params)]
```

Unlike DQN's hard updates, DDPG uses **soft updates**:
- θ' ← τθ + (1-τ)θ' where τ << 1 (e.g., 0.001)
- More stable than periodic hard copies
- Provides gradual target network evolution

## The DDPG Learning Algorithm

### 1. Experience Collection

```python
def choose_action(self, s):
    a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
    a = np.clip(np.random.normal(a, self.var), -self.a_bound, self.a_bound)
    return a
```

**Exploration Strategy**: Add Gaussian noise to deterministic actions
- Ornstein-Uhlenbeck process (temporally correlated noise)
- Gradually decay noise variance during training
- More sophisticated than ε-greedy for continuous spaces

### 2. Experience Replay

Same as DQN but with continuous actions:
- Store (s, a, r, s') transitions
- Sample random batches for training
- Breaks correlation in sequential data

### 3. Critic Update

```python
# Compute target Q-values
a_ = self.sess.run(self.a_, {self.S_: batch_memory[:, -self.s_dim:]})
q_ = self.sess.run(self.q_, {self.S_: batch_memory[:, -self.s_dim:], 
                            self.A_: a_})
q_target = batch_memory[:, self.s_dim + self.a_dim] + self.gamma * q_

# Train critic to minimize TD error
self.sess.run(self.ctrain, {self.S: batch_memory[:, :self.s_dim],
                           self.A: batch_memory[:, self.s_dim:self.s_dim+self.a_dim],
                           self.R: q_target})
```

Standard temporal difference learning for continuous actions.

### 4. Actor Update

```python
# Policy gradient: maximize Q(s, π(s))
self.sess.run(self.atrain, {self.S: batch_memory[:, :self.s_dim]})
```

The actor maximizes the Q-value of its chosen actions.

### 5. Target Network Updates

```python
# Soft update of target networks
self.sess.run(self.soft_replace)
```

After each learning step, gently update target networks.

## Exploration in Continuous Spaces

### Ornstein-Uhlenbeck Process

Better than simple Gaussian noise for control tasks:

```python
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

**Properties**:
- **Mean reverting**: Tends to return to mean (μ)
- **Temporally correlated**: Smoother than independent noise
- **Configurable**: θ controls correlation, σ controls magnitude

### Exploration Decay

```python
self.var *= 0.9995  # Decay exploration noise
```

Start with high exploration, gradually reduce as policy improves.

## DDPG vs Other Methods

### Comparison Table

| Aspect | DDPG | Actor-Critic | DQN | Policy Gradients |
|--------|------|--------------|-----|------------------|
| Action Space | Continuous | Both | Discrete | Both |
| Policy Type | Deterministic | Stochastic | Deterministic | Stochastic |
| Sample Efficiency | High | Medium | High | Low |
| Stability | Medium | Medium | High | Low |
| Exploration | Noise-based | Natural | ε-greedy | Natural |
| Off-policy | Yes | No | Yes | No |

### When to Use DDPG

**Best for**:
- Continuous control tasks
- Robotics and physical simulation
- Tasks requiring precise actions
- When sample efficiency matters

**Avoid when**:
- Discrete action spaces (use DQN)
- Very high-dimensional actions
- Stochastic optimal policies exist
- Computational resources are limited

## Practical Implementation

### Hyperparameter Guidelines

```python
# Network architecture
ACTOR_LR = 0.0001      # Lower than critic
CRITIC_LR = 0.001      # Higher learning rate
TAU = 0.001           # Soft update rate
GAMMA = 0.99          # Discount factor

# Experience replay  
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

# Exploration
NOISE_VARIANCE = 3.0   # Initial exploration
NOISE_DECAY = 0.9995   # Decay rate
```

### Common Issues and Solutions

**1. Training Instability**
- Reduce learning rates
- Increase batch size
- Use batch normalization
- Clip gradients

**2. Poor Exploration**
- Increase initial noise variance
- Use different noise processes
- Add curiosity-driven exploration
- Parameter space noise

**3. Slow Convergence**
- Check action/state scaling
- Normalize observations
- Tune target update rate (τ)
- Verify reward scaling

## DDPG in Action: Pendulum Control

### Environment Setup

```python
env = gym.make('Pendulum-v0')
# State: [cos(θ), sin(θ), θ_dot]  
# Action: torque ∈ [-2, 2]
# Goal: Balance pendulum upright
```

### Learning Progression

**Episodes 1-100**: Random swinging, learning basic dynamics
**Episodes 101-300**: Occasional balance attempts, improving stability  
**Episodes 301-500**: Consistent balancing, fine-tuning control
**Episodes 501+**: Near-optimal control, minimal oscillation

### Performance Metrics

- **Reward progression**: From -1500 to -200
- **Action smoothness**: Less jittery over time
- **Convergence speed**: 2-3x faster than policy gradients

## Advanced DDPG Variants

### Twin Delayed DDPG (TD3)

Improvements over DDPG:
- Two critic networks (reduce overestimation)
- Delayed policy updates (reduce variance)
- Target policy smoothing (regularization)

### Soft Actor-Critic (SAC)

- Maximum entropy framework
- Better exploration and robustness
- State-of-the-art continuous control performance

## What's Next?

DDPG introduced us to deep continuous control, but modern RL has evolved further. Our next post explores Asynchronous Advantage Actor-Critic (A3C), which parallelizes learning across multiple environments for faster, more robust training.

Coming up: "A3C: Asynchronous Advantage Actor-Critic"!

---

*Master continuous control with DDPG implementations in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 