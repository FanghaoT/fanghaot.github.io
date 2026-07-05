---
title: Actor-Critic - Best of Both Worlds
author: fanghao
date: 2024-03-01 10:00:00 +0200
categories: [AI, Reinforcement Learning, Actor-Critic]
tags: [actor-critic, advantage, policy-gradients, value-functions]
---

# Actor-Critic: Best of Both Worlds

Today we unite two powerful paradigms: the direct policy optimization of policy gradients with the sample efficiency of value-based methods. Actor-Critic methods represent a major breakthrough in RL, combining the strengths of both approaches while mitigating their individual weaknesses.

## The Marriage of Policy and Value

### Why Combine Them?

**Policy Gradients** are great but suffer from:
- High variance in gradient estimates
- Poor sample efficiency  
- Slow convergence

**Value-Based Methods** excel at:
- Low variance estimates
- Sample efficiency
- Stable learning

**Actor-Critic** combines both by using:
- **Actor**: Policy network π(a|s; θ) that selects actions
- **Critic**: Value network V(s; φ) that evaluates states

## The Actor-Critic Architecture

### Dual Network Design

```python
class ActorCritic:
    def __init__(self, n_features, n_actions, lr_a=0.001, lr_c=0.01):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr_a = lr_a  # Actor learning rate
        self.lr_c = lr_c  # Critic learning rate
        
        # Build both networks
        self._build_actor()
        self._build_critic()
        
    def _build_actor(self):
        with tf.name_scope('actor'):
            self.s = tf.placeholder(tf.float32, [1, self.n_features], name="state")
            self.a = tf.placeholder(tf.int32, None, name="act")
            self.td_error = tf.placeholder(tf.float32, None, name="td_error")
            
            l1 = tf.layers.dense(self.s, 20, tf.nn.relu)
            acts_prob = tf.layers.dense(l1, self.n_actions, tf.nn.softmax)
            
            # Policy loss weighted by TD error
            log_prob = tf.log(acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)
            self.train_op_a = tf.train.AdamOptimizer(self.lr_a).minimize(-self.exp_v)
            
    def _build_critic(self):  
        with tf.name_scope('critic'):
            self.r = tf.placeholder(tf.float32, None, name='r')
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            
            l1 = tf.layers.dense(self.s, 20, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            
            # TD target and error
            self.td_target = self.r + GAMMA * self.v_
            self.td_error_out = self.td_target - self.v
            self.loss_c = tf.square(self.td_error_out)
            self.train_op_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_c)
```

### The Learning Process

1. **Actor** selects action based on current policy
2. **Critic** evaluates the current state  
3. Environment provides reward and next state
4. **TD Error** computed: δ = r + γV(s') - V(s)
5. **Critic** updated to minimize TD error
6. **Actor** updated using TD error as advantage estimate

## Understanding the Advantage Function

### From Returns to Advantages

Instead of using full returns G_t, Actor-Critic uses the **advantage function**:

```
A(s,a) = Q(s,a) - V(s)
```

The advantage tells us "how much better is action a compared to the average action in state s?"

### TD Error as Advantage Estimate

The brilliant insight: TD error approximates the advantage!

```
δ_t = r_t + γV(s_{t+1}) - V(s_t) ≈ A(s_t, a_t)
```

This gives us a low-variance, unbiased estimate of advantage without needing to store Q-values.

### Actor Update with Advantage

```python
def learn(self, s, a, r, s_):
    # Critic learns from TD error
    v_ = self.sess.run(self.v, {self.s: s_})
    td_error, _ = self.sess.run([self.td_error_out, self.train_op_c],
                               {self.s: s, self.v_: v_, self.r: r})
    
    # Actor learns from advantage (TD error)
    self.sess.run(self.train_op_a, {self.s: s, self.a: a, self.td_error: td_error})
```

## Types of Actor-Critic Methods

### 1. One-Step Actor-Critic

Updates after every step using single-step TD error:
- Fastest updates
- Higher variance
- Good for online learning

### 2. Advantage Actor-Critic (A2C)

Uses n-step returns or GAE (Generalized Advantage Estimation):
- Lower variance than one-step
- Better bias-variance tradeoff
- More stable learning

### 3. Asynchronous Advantage Actor-Critic (A3C)

Runs multiple actors in parallel:
- Faster training through parallelization  
- More diverse experience collection
- Improved exploration

## Implementation Details

### Action Selection (Actor)

```python
def choose_action(self, s):
    probs = self.sess.run(self.acts_prob, {self.s: s})
    return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
```

### Value Estimation (Critic)

```python
def get_value(self, s):
    if s is None:
        return 0
    return self.sess.run(self.v, {self.s: s})[0, 0]
```

### Complete Learning Loop

```python
def train_episode(self):
    s = env.reset()
    track_r = []
    
    while True:
        a = ac.choose_action(s)
        s_, r, done, info = env.step(a)
        
        if done:
            r = -20  # Terminal state penalty
            
        track_r.append(r)
        
        # TD learning
        ac.learn(s, a, r, s_)
        
        s = s_
        
        if done:
            # Episode finished
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                render = True  # Rendering
            break
```

## Advantages of Actor-Critic

### 1. Reduced Variance
- TD error has lower variance than Monte Carlo returns
- Bootstrapping provides more stable learning signals
- Faster convergence than pure policy gradients

### 2. Online Learning
- Updates after every step, not every episode
- More sample efficient than REINFORCE
- Suitable for continuing tasks

### 3. Theoretical Properties
- Unbiased policy gradient estimates
- Convergence guarantees under certain conditions
- Natural handling of continuous actions

### 4. Flexibility
- Works with both discrete and continuous action spaces
- Can incorporate different advantage estimation methods
- Easily extended with additional techniques

## Common Challenges and Solutions

### 1. Learning Rate Balance

**Problem**: Actor and critic learning rates must be carefully balanced.

**Solution**: 
- Usually lr_critic > lr_actor (e.g., 0.01 vs 0.001)
- Monitor both losses during training
- Use adaptive learning rates if needed

### 2. Bootstrapping Bias

**Problem**: Critic errors propagate to actor updates.

**Solutions**:
- Use target networks (like DQN)
- Multiple critics with ensemble averaging
- Regularization techniques

### 3. Exploration vs Exploitation

**Problem**: Policy may converge prematurely to suboptimal solutions.

**Solutions**:
- Entropy regularization: add H(π) to loss
- Exploration bonuses
- Parameter noise injection

## Entropy Regularization

Adding entropy encourages exploration:

```python
# Entropy bonus in actor loss
entropy = -tf.reduce_sum(acts_prob * tf.log(acts_prob + 1e-8))
actor_loss = -self.exp_v - 0.01 * entropy  # 0.01 is entropy coefficient
```

Benefits:
- Prevents premature convergence
- Maintains exploration throughout training
- Particularly important in sparse reward environments

## CartPole with Actor-Critic

### Performance Comparison

| Method | Episodes to Solve | Sample Efficiency | Stability |
|--------|------------------|-------------------|-----------|
| REINFORCE | 800-1200 | Low | High variance |
| DQN | 300-500 | Medium | Stable |
| Actor-Critic | 200-400 | High | Medium |

### Learning Characteristics

**Early Episodes (1-50)**:
- Erratic behavior as both networks learn
- Critic provides increasingly accurate value estimates
- Actor gradually improves action selection

**Middle Episodes (51-150)**:
- More consistent performance
- Critic stabilizes, providing better advantage estimates  
- Actor policy becomes more deterministic around good actions

**Late Episodes (151+)**:
- Near-optimal performance
- Stable learning with low variance updates
- Policy converges to high-performance solution

## Debugging Actor-Critic

### Key Metrics to Monitor

1. **TD Error**: Should decrease over time as critic improves
2. **Policy Entropy**: Should gradually decrease but not hit zero
3. **Value Function**: Should approximate true returns
4. **Actor Loss**: Should generally decrease (with some noise)

### Common Issues

**Critic not learning**: Check learning rate, network capacity
**Actor not exploring**: Add entropy regularization, check temperature
**Unstable training**: Balance learning rates, add target networks
**Poor performance**: Check reward scaling, advantage normalization

## What's Next?

Actor-Critic opened the door to modern deep RL. Our next post explores Deep Deterministic Policy Gradient (DDPG), which extends Actor-Critic to continuous control tasks with deterministic policies - perfect for robotics and control applications.

Coming up: "DDPG: Deep RL for Continuous Control"!

---

*Implement and experiment with Actor-Critic variants in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 