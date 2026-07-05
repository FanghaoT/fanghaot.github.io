---
title: OpenAI Gym - Your RL Playground
author: fanghao
date: 2024-02-16 10:00:00 +0200
categories: [AI, Reinforcement Learning, Environments]
tags: [openai-gym, cartpole, mountaincar, environments, classic-control]
---

# OpenAI Gym: Your RL Playground

Welcome to the standardized world of RL environments! OpenAI Gym provides a consistent interface for hundreds of RL tasks, from classic control problems to Atari games. Today we'll explore this essential toolkit and apply our DQN knowledge to two iconic challenges: CartPole and MountainCar.

## Why Standardized Environments Matter

Before Gym, RL researchers faced:
- **Inconsistent interfaces** across different problems
- **Irreproducible results** due to environment differences  
- **Wasted effort** reimplementing basic environments
- **Limited comparability** between algorithms

Gym solved these issues with a unified API that became the industry standard.

## The Gym Interface

### Core Components

```python
import gym

# Create environment
env = gym.make('CartPole-v1')

# Reset environment  
observation = env.reset()

# Take action
next_observation, reward, done, info = env.step(action)

# Render (optional)
env.render()
```

### Key Concepts

**Observation Space**: What the agent sees
```python
env.observation_space  # Box(4,) for CartPole
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

**Action Space**: What the agent can do
```python  
env.action_space  # Discrete(2) for CartPole
# 0: Push cart left, 1: Push cart right
```

**Episode Structure**: Tasks are divided into episodes with clear start/end points.

## CartPole: The "Hello World" of RL

### Problem Description

**Goal**: Balance a pole on a moving cart by applying left/right forces.

**Observation**: 4D continuous vector
- Cart position: [-4.8, 4.8]
- Cart velocity: [-∞, ∞]  
- Pole angle: [-0.418, 0.418] radians (~24°)
- Pole angular velocity: [-∞, ∞]

**Actions**: 2 discrete choices
- 0: Push cart to the left
- 1: Push cart to the right

**Rewards**: +1 for each timestep the pole remains upright

**Termination**: 
- Pole angle > 15°
- Cart position > 2.4 units from center
- Episode length > 500 steps

### Why CartPole is Perfect for Learning

1. **Simple but non-trivial**: Easy to understand, challenging to solve
2. **Fast episodes**: Quick feedback for algorithm development
3. **Continuous observations**: Tests function approximation
4. **Clear success metric**: Episode length indicates performance

### DQN Solution for CartPole

```python
class CartPoleDQN:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.dqn = DeepQNetwork(
            n_actions=2,
            n_features=4,
            learning_rate=0.01,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
        )
    
    def train(self):
        for episode in range(300):
            observation = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.dqn.choose_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.dqn.store_transition(observation, action, reward, next_observation)
                
                # Learn from experience
                if self.dqn.memory_counter > 200:
                    self.dqn.learn()
                
                observation = next_observation
                total_reward += reward
                
                if done:
                    break
            
            print(f'Episode {episode}: {total_reward} steps')
```

### Learning Progression

**Episodes 1-50**: Random behavior, 10-30 steps average
**Episodes 51-150**: Gradual improvement, 50-200 steps  
**Episodes 151-300**: Consistent performance, 400-500 steps

The agent learns to:
1. Keep the pole centered initially
2. Make corrective movements when pole tilts
3. Anticipate pole motion and act preemptively

## MountainCar: The Sparse Reward Challenge

### Problem Description

**Goal**: Drive an underpowered car up a steep hill to reach the flag.

**The Challenge**: The car's engine isn't strong enough to drive straight up the hill. The agent must learn to build momentum by rocking back and forth.

**Observation**: 2D continuous vector
- Car position: [-1.2, 0.6] (flag at 0.5)
- Car velocity: [-0.07, 0.07]

**Actions**: 3 discrete choices
- 0: Push left (negative direction)
- 1: No push (neutral)  
- 2: Push right (positive direction)

**Rewards**: -1 for each timestep (encouraging speed)

**Termination**:
- Reach the flag (position ≥ 0.5)
- Maximum 200 steps per episode

### Why MountainCar is Challenging

1. **Sparse rewards**: Only -1 per step, no immediate feedback for good actions
2. **Counterintuitive strategy**: Must go backward to go forward
3. **Long episodes**: Takes many steps to see results
4. **Exploration challenge**: Random actions rarely succeed

### DQN Adaptations for MountainCar

```python
class MountainCarDQN:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.dqn = DeepQNetwork(
            n_actions=3,
            n_features=2,
            learning_rate=0.001,  # Lower for stability
            e_greedy=0.95,        # Higher exploration needed
            replace_target_iter=200,
            memory_size=10000,    # Larger memory for sparse rewards
            e_greedy_increment=0.0002,  # Gradual exploration decay
        )
```

### Key Modifications for Sparse Rewards

**Reward Shaping** (optional):
```python
def shape_reward(position, velocity):
    # Encourage rightward movement and higher positions
    return position + abs(velocity) * 0.1
```

**Extended Exploration**:
- Higher initial ε value
- Slower ε decay
- Larger replay buffer to store rare successful experiences

### Learning Progression

**Episodes 1-500**: Consistent failure, -200 reward (max steps)
**Episodes 501-1000**: Occasional success, breakthrough moments
**Episodes 1001+**: Increasing success rate, faster solutions

The agent learns to:
1. Oscillate to build momentum
2. Time the final push to the right
3. Optimize the momentum-building strategy

## Environment Variations and Customization

### Modifying Existing Environments

```python
# Wrapper to modify rewards
class RewardShapingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # Modify reward function
        return modified_reward

# Wrapper to change observations  
class ObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        # Transform observations
        return transformed_obs

# Usage
env = RewardShapingWrapper(gym.make('MountainCar-v0'))
```

### Creating Custom Environments

```python
class CustomEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    
    def step(self, action):
        # Implement environment dynamics
        return observation, reward, done, info
        
    def reset(self):
        # Reset environment to initial state
        return initial_observation
        
    def render(self, mode='human'):
        # Visualization (optional)
        pass
```

## Practical Tips for Gym Environments

### Performance Optimization
- **Vectorized environments**: Run multiple environments in parallel
- **Action repeat**: Skip frames for faster learning in some domains
- **Observation preprocessing**: Normalize or stack observations as needed

### Debugging Strategies
- **Random baseline**: Test with random actions first
- **Environment inspection**: Understand observation/action spaces
- **Reward analysis**: Plot reward distributions and episode lengths
- **Action frequency**: Monitor which actions are being selected

### Common Pitfalls
- **Reward scale**: Some environments have vastly different reward scales
- **Episode termination**: Understand when and why episodes end
- **Determinism**: Use seeds for reproducible results
- **Version differences**: Gym versions may have different behavior

## Environment Categories in Gym

### Classic Control
- CartPole, MountainCar, Acrobot, Pendulum
- Perfect for learning and testing algorithms

### Atari
- Pong, Breakout, Space Invaders, etc.
- Pixel-based observations, great for CNN-based agents

### Box2D
- LunarLander, BipedalWalker, CarRacing
- Physics simulation environments

### MuJoCo
- Humanoid, Ant, HalfCheetah (requires license)
- High-fidelity continuous control tasks

## What's Next?

We've mastered value-based methods (DQN family) in standard environments. Our next post introduces a fundamentally different approach: Policy Gradients. Instead of learning action values, we'll learn to directly optimize the policy itself using gradient ascent.

Coming up: "Policy Gradients: Learning Actions Directly"!

---

*Try CartPole and MountainCar implementations with various DQN variants in the [RL-Tutorial-Series repository](https://github.com/fanghaot/RL-Tutorial-Series)* 