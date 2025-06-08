# Reinforcement Learning

## Background and Introduction
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. It combines elements of behavioral psychology, control theory, and machine learning to create systems that can learn optimal behavior through trial and error.

## What is Reinforcement Learning?
RL is characterized by:
1. Agent-environment interaction
2. State-action-reward framework
3. Policy learning
4. Value function estimation
5. Exploration vs. exploitation trade-off

## Why Reinforcement Learning?
1. **Autonomous Decision Making**: Learn optimal policies
2. **Adaptive Behavior**: Respond to changing environments
3. **Complex Problem Solving**: Handle sequential decision-making
4. **Real-world Applications**: Robotics, gaming, control systems
5. **Continuous Learning**: Improve over time

## How to Implement Reinforcement Learning?

### 1. Q-Learning
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import gym

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-learning update
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Update Q-value
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value
        
        # Decay exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay
    
    def save(self, filename):
        np.save(filename, self.q_table)
    
    def load(self, filename):
        self.q_table = np.load(filename)

def train_q_learning(env, agent, episodes=1000):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    return rewards_history
```

### 2. Deep Q-Network (DQN)
```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

def train_dqn(env, agent, episodes=1000, batch_size=32):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(batch_size)
        
        if episode % 10 == 0:
            agent.update_target_model()
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    return rewards_history
```

### 3. Policy Gradient Methods
```python
class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs = self.model.predict(state)[0]
        return np.random.choice(self.action_size, p=probs)
    
    def train(self, states, actions, rewards):
        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / \
                           (np.std(discounted_rewards) + 1e-8)
        
        # Convert actions to one-hot encoding
        actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_size)
        
        # Train the model
        self.model.fit(states, actions_one_hot, sample_weight=discounted_rewards,
                      epochs=1, verbose=0)
    
    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)

def train_policy_gradient(env, agent, episodes=1000):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            total_reward += reward
        
        # Train the agent
        agent.train(np.array(states), np.array(actions), np.array(rewards))
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    return rewards_history
```

## Model Evaluation

### 1. Performance Metrics
```python
def evaluate_agent(env, agent, episodes=100):
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'min_reward': np.min(total_rewards)
    }

def plot_training_history(rewards_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
```

## Common Interview Questions

1. **Q: What is the difference between value-based and policy-based methods?**
   - A: Value-based methods (e.g., Q-learning) learn the value of state-action pairs and select actions based on these values. Policy-based methods (e.g., Policy Gradient) directly learn the policy that maps states to actions. Value-based methods are more sample-efficient but policy-based methods can handle continuous action spaces better.

2. **Q: What is the exploration vs. exploitation trade-off?**
   - A: The trade-off involves:
     - Exploration: Trying new actions to discover their effects
     - Exploitation: Using known good actions to maximize rewards
     - Need to balance both for optimal learning
     - Various strategies (ε-greedy, softmax, etc.)
     - Important for avoiding local optima

3. **Q: How do you handle the credit assignment problem in RL?**
   - A: Solutions include:
     - Temporal difference learning
     - Eligibility traces
     - Reward shaping
     - Hierarchical reinforcement learning
     - Proper discount factor selection

## Hands-on Task: CartPole Balancing

### Project: CartPole Control
```python
def cartpole_control_project():
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Train agent
    rewards_history = train_dqn(env, agent)
    
    # Plot training history
    plot_training_history(rewards_history)
    
    # Evaluate agent
    evaluation_results = evaluate_agent(env, agent)
    print("\nEvaluation Results:")
    print(f"Mean Reward: {evaluation_results['mean_reward']:.2f}")
    print(f"Std Reward: {evaluation_results['std_reward']:.2f}")
    print(f"Max Reward: {evaluation_results['max_reward']:.2f}")
    print(f"Min Reward: {evaluation_results['min_reward']:.2f}")
    
    # Save model
    agent.save('cartpole_dqn.h5')
    
    return {
        'agent': agent,
        'history': rewards_history,
        'evaluation': evaluation_results
    }
```

## Next Steps
1. Learn about advanced RL algorithms
2. Study multi-agent reinforcement learning
3. Explore inverse reinforcement learning
4. Practice with real-world applications
5. Learn about model optimization

## Resources
- [OpenAI Gym](https://gym.openai.com/)
- [Deep Reinforcement Learning](https://www.deepmind.com/learning-resources/-introduction-to-reinforcement-learning-with-david-silver)
- [RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [TensorFlow Agents](https://www.tensorflow.org/agents) 