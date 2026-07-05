---
title: Advanced Topics - Dyna-Q and Curiosity-Driven Learning
author: fanghao
date: 2024-03-29 10:00:00 +0200
categories: [AI, Reinforcement Learning, Advanced Topics]
tags: [dyna-q, model-based-rl, curiosity, intrinsic-motivation, exploration]
---

# Advanced Topics: Dyna-Q and Curiosity-Driven Learning

Welcome to the cutting edge of reinforcement learning! Today we explore two fascinating advanced topics that push RL beyond traditional boundaries: Dyna-Q (which bridges model-free and model-based RL) and Curiosity-Driven Learning (which gives agents intrinsic motivation to explore). These methods represent different approaches to making RL more sample-efficient and capable.

## Part 1: Dyna-Q - Planning Meets Learning

### The Model-Free vs Model-Based Divide

**Model-Free RL** (Q-learning, PPO):
- Learn directly from experience
- No assumptions about environment dynamics
- Sample inefficient but robust

**Model-Based RL**:  
- Learn a model of the environment
- Use model for planning
- Sample efficient but model errors can be problematic

### Dyna-Q: Best of Both Worlds

Dyna-Q combines:
- **Direct learning**: Update Q-values from real experience
- **Planning**: Use learned model to generate simulated experience
- **Model learning**: Continuously improve environment model

### The Dyna-Q Algorithm

```python
class DynaQ:
    def __init__(self, n_states, n_actions, planning_steps=50):
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}  # (state, action) -> (next_state, reward)
        self.planning_steps = planning_steps
        self.visited_states = set()
        
    def learn(self, state, action, reward, next_state):
        # 1. Direct RL update (Q-learning)
        self.q_update(state, action, reward, next_state)
        
        # 2. Model learning
        self.model[(state, action)] = (next_state, reward)
        self.visited_states.add(state)
        
        # 3. Planning (simulated experience)
        for _ in range(self.planning_steps):
            self.planning_update()
    
    def q_update(self, s, a, r, s_next):
        """Standard Q-learning update"""
        best_next_q = np.max(self.q_table[s_next])
        td_target = r + self.gamma * best_next_q
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error
    
    def planning_update(self):
        """Update using simulated experience"""
        # Sample random state that has been visited
        s = np.random.choice(list(self.visited_states))
        
        # Sample random action that has been tried from this state
        actions_tried = [a for (state, a) in self.model.keys() if state == s]
        if not actions_tried:
            return
            
        a = np.random.choice(actions_tried)
        s_next, r = self.model[(s, a)]
        
        # Update Q-value using simulated experience
        self.q_update(s, a, r, s_next)
```

### Benefits of Dyna-Q

**Sample Efficiency**: 
- Each real experience generates multiple learning updates
- Planning leverages past experience repeatedly
- Faster convergence than pure model-free methods

**Robustness**:
- Still learns from real experience (handles model errors)
- Model improves over time
- Degrades gracefully when model is poor

**Flexibility**:
- Planning steps can be adjusted based on computational budget
- Works with any model-free algorithm as base
- Easy to implement and understand

### Dyna-Q in Maze Navigation

In maze environments, Dyna-Q excels because:

1. **Deterministic dynamics**: Easy to model accurately
2. **Reusable experience**: Previous explorations help plan new routes
3. **Credit assignment**: Planning helps propagate rewards backward

```python
def maze_dyna_experiment():
    env = MazeEnvironment()
    agent = DynaQ(n_states=100, n_actions=4, planning_steps=50)
    
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Learn from real experience + planning
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode}: {total_reward} steps")
```

**Results**: Dyna-Q typically solves mazes 5-10x faster than Q-learning alone.

## Part 2: Curiosity-Driven Learning

### The Exploration Problem

Traditional RL relies on external rewards, but many environments have:
- **Sparse rewards**: Long periods without feedback
- **Exploration challenges**: Random actions may never find rewards
- **Local optima**: Agents get stuck in suboptimal behaviors

### Intrinsic Motivation

**Key insight**: Give agents internal drive to explore novel situations.

**Curiosity**: Reward the agent for encountering surprising states or transitions.

### Intrinsic Curiosity Module (ICM)

The ICM framework has three components:

1. **Forward Model**: Predicts next state given current state and action
2. **Inverse Model**: Predicts action given current and next state  
3. **Curiosity Reward**: Based on forward model prediction error

```python
class ICM:
    def __init__(self, state_dim, action_dim, feature_dim=256):
        self.feature_net = self.build_feature_network(state_dim, feature_dim)
        self.forward_model = self.build_forward_model(feature_dim, action_dim)
        self.inverse_model = self.build_inverse_model(feature_dim, action_dim)
        
    def build_feature_network(self, state_dim, feature_dim):
        """Extract features from raw states"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(state_dim,)),
            Dense(256, activation='relu'),
            Dense(feature_dim, activation='relu')
        ])
        return model
    
    def build_forward_model(self, feature_dim, action_dim):
        """Predict next state features from current features and action"""
        state_input = Input(shape=(feature_dim,))
        action_input = Input(shape=(action_dim,))
        
        x = Concatenate()([state_input, action_input])
        x = Dense(256, activation='relu')(x)
        next_state_pred = Dense(feature_dim)(x)
        
        return Model([state_input, action_input], next_state_pred)
    
    def build_inverse_model(self, feature_dim, action_dim):
        """Predict action from current and next state features"""
        current_input = Input(shape=(feature_dim,))
        next_input = Input(shape=(feature_dim,))
        
        x = Concatenate()([current_input, next_input])
        x = Dense(256, activation='relu')(x)
        action_pred = Dense(action_dim, activation='softmax')(x)
        
        return Model([current_input, next_input], action_pred)
```

### Computing Curiosity Rewards

```python
def compute_curiosity_reward(self, state, action, next_state):
    # Extract features
    state_features = self.feature_net(state)
    next_state_features = self.feature_net(next_state)
    
    # Forward model prediction
    predicted_next_features = self.forward_model([state_features, action])
    
    # Curiosity reward = prediction error
    curiosity_reward = tf.reduce_mean(tf.square(
        next_state_features - predicted_next_features
    ), axis=1)
    
    return curiosity_reward

def train_icm(self, states, actions, next_states):
    """Train the ICM components"""
    state_features = self.feature_net(states)
    next_state_features = self.feature_net(next_states)
    
    # Forward model loss
    predicted_next_features = self.forward_model([state_features, actions])
    forward_loss = tf.reduce_mean(tf.square(
        next_state_features - predicted_next_features
    ))
    
    # Inverse model loss  
    predicted_actions = self.inverse_model([state_features, next_state_features])
    inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=predicted_actions
    ))
    
    # Combined loss
    total_loss = forward_loss + inverse_loss
    
    return total_loss
```

### PPO + Curiosity Integration

```python
class CuriousPPO:
    def __init__(self, state_dim, action_dim):
        self.ppo = PPO(state_dim, action_dim)
        self.icm = ICM(state_dim, action_dim)
        self.curiosity_coef = 0.01
        
    def compute_total_reward(self, states, actions, next_states, ext_rewards):
        """Combine external and intrinsic rewards"""
        curiosity_rewards = self.icm.compute_curiosity_reward(
            states, actions, next_states
        )
        
        total_rewards = ext_rewards + self.curiosity_coef * curiosity_rewards
        return total_rewards
    
    def update(self, trajectory):
        states, actions, ext_rewards, next_states = trajectory
        
        # Compute total rewards (external + curiosity)
        total_rewards = self.compute_total_reward(
            states, actions, next_states, ext_rewards
        )
        
        # Update PPO with augmented rewards
        self.ppo.update(states, actions, total_rewards, ...)
        
        # Update ICM
        icm_loss = self.icm.train_icm(states, actions, next_states)
```

## Random Network Distillation (RND)

An alternative curiosity approach:

### Core Idea
- Train predictor network to match random target network
- Prediction error indicates state novelty
- Simpler than ICM, often more effective

```python
class RND:
    def __init__(self, state_dim, feature_dim=512):
        # Fixed random target network
        self.target_net = self.build_network(state_dim, feature_dim)
        self.target_net.trainable = False
        
        # Trainable predictor network
        self.predictor_net = self.build_network(state_dim, feature_dim)
        
    def build_network(self, state_dim, feature_dim):
        return Sequential([
            Dense(512, activation='relu', input_shape=(state_dim,)),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(feature_dim)
        ])
    
    def compute_curiosity_reward(self, states):
        target_features = self.target_net(states)
        predicted_features = self.predictor_net(states)
        
        curiosity_reward = tf.reduce_mean(tf.square(
            target_features - predicted_features
        ), axis=1)
        
        return curiosity_reward
```

### RND Advantages
- **Simpler**: No inverse model needed
- **More stable**: No issues with environment stochasticity
- **Better exploration**: Often finds rare states more effectively

## Practical Considerations

### Hyperparameter Guidelines

**Dyna-Q**:
```python
PLANNING_STEPS = 10-200    # More planning = better but slower
ALPHA = 0.1               # Q-learning rate
EPSILON = 0.1             # Exploration rate
```

**Curiosity Learning**:
```python
CURIOSITY_COEF = 0.01-0.1  # Balance external vs intrinsic rewards
FEATURE_DIM = 256-512      # Feature representation size
ICM_LR = 1e-3             # ICM learning rate
```

### When to Use Each Method

**Dyna-Q**:
- Deterministic environments
- When computational resources allow planning
- Sparse but learnable dynamics

**Curiosity**:  
- Sparse reward environments
- When exploration is critical
- Complex, high-dimensional state spaces

### Common Pitfalls

**Dyna-Q**:
- Model errors compound during planning
- Computational overhead of planning
- May overfit to early incorrect models

**Curiosity**:
- Noisy TV problem (fascination with randomness)
- Reward scale balancing issues
- May ignore external rewards if curiosity is too strong

## Combining Advanced Techniques

### Dyna + Curiosity

```python
class CuriousDyna:
    def __init__(self):
        self.dyna_agent = DynaQ(...)
        self.curiosity = ICM(...)
        
    def learn(self, state, action, reward, next_state):
        # Compute curiosity reward
        curiosity_reward = self.curiosity.compute_reward(state, action, next_state)
        total_reward = reward + 0.01 * curiosity_reward
        
        # Dyna learning with augmented reward
        self.dyna_agent.learn(state, action, total_reward, next_state)
        
        # Update curiosity model
        self.curiosity.update(state, action, next_state)
```

This combination can be particularly powerful for complex exploration tasks.

## Real-World Applications

### Game Playing
- **Montezuma's Revenge**: Curiosity helps navigate sparse rewards
- **Minecraft**: Intrinsic motivation for open-ended exploration
- **Go/Chess**: Dyna-Q style planning in tree search

### Robotics
- **Manipulation**: Curiosity for discovering object interactions
- **Navigation**: Dyna-Q for path planning with learned maps
- **Skill Discovery**: Intrinsic motivation for learning diverse behaviors

### Scientific Discovery
- **Drug Discovery**: Curiosity for exploring chemical space
- **Materials Science**: Model-based planning for property optimization

## What's Next?

We've covered the fundamentals and many advanced topics in reinforcement learning! Our final post will provide a comprehensive summary of the entire series, discuss the current state of the field, and point toward exciting future directions in RL research and applications.

Coming up: "The Future of Reinforcement Learning - Summary and Outlook"!

---

*Experiment with advanced RL techniques in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 