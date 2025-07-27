import numpy as np
import random
import math
from collections import deque


class PPOAgent:
    """PPO Agent for GBICR routing decisions"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Experience buffer
        self.memory = PPOMemory()
        
        # For offline training mode
        self.is_training = False
        self.model_loaded = False
        
        # Simple policy network simulation (in real implementation, use PyTorch/TensorFlow)
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.value_weights = np.random.randn(state_dim, 1) * 0.1
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def select_action(self, state, available_actions, exploration=True):
        """
        Select action using PPO policy
        
        Args:
            state: current state vector
            available_actions: list of available neighbor IDs
            exploration: whether to use exploration
            
        Returns:
            selected_action: chosen neighbor ID
            action_prob: probability of the selected action
        """
        if len(available_actions) == 0:
            return None, 0.0
        
        if len(available_actions) == 1:
            return available_actions[0], 1.0
        
        # Simple policy evaluation (placeholder for neural network)
        action_logits = self._evaluate_policy(state, available_actions)
        
        if exploration and not self.model_loaded:
            # Add exploration noise during training
            action_logits += np.random.normal(0, 0.1, len(action_logits))
        
        # Softmax to get probabilities
        action_probs = self._softmax(action_logits)
        
        # Sample action
        if exploration:
            action_idx = np.random.choice(len(available_actions), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)
        
        selected_action = available_actions[action_idx]
        action_prob = action_probs[action_idx]
        
        return selected_action, action_prob
    
    def _evaluate_policy(self, state, available_actions):
        """Evaluate policy for available actions (simplified)"""
        # This is a simplified version - in practice, use neural networks
        action_values = []
        
        for action in available_actions:
            # Simple linear combination (placeholder)
            value = np.dot(state, self.policy_weights[:, action % self.action_dim])
            action_values.append(value)
        
        return np.array(action_values)
    
    def _evaluate_value(self, state):
        """Evaluate state value (simplified)"""
        return np.dot(state, self.value_weights.flatten())
    
    def _softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def store_transition(self, state, action, reward, next_state, done, action_prob):
        """Store transition in memory"""
        self.memory.store(state, action, reward, next_state, done, action_prob)
    
    def update(self):
        """Update PPO policy (simplified version)"""
        if not self.is_training or len(self.memory.states) < 32:
            return
        
        # Get batch data
        states = np.array(self.memory.states)
        actions = np.array(self.memory.actions)
        rewards = np.array(self.memory.rewards)
        old_probs = np.array(self.memory.action_probs)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # PPO update (simplified)
        for _ in range(self.k_epochs):
            # Calculate current action probabilities
            current_probs = []
            values = []
            
            for i, state in enumerate(states):
                action_logits = self._evaluate_policy(state, [actions[i]])
                prob = self._softmax(action_logits)[0]
                current_probs.append(prob)
                values.append(self._evaluate_value(state))
            
            current_probs = np.array(current_probs)
            values = np.array(values)
            
            # Calculate advantages
            advantages = discounted_rewards - values
            
            # Calculate ratio
            ratios = current_probs / (old_probs + 1e-8)
            
            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = np.clip(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -np.mean(np.minimum(surr1, surr2))
            
            # Calculate value loss
            value_loss = np.mean((values - discounted_rewards) ** 2)
            
            # Simple gradient update (placeholder)
            self._update_weights(states, actions, policy_loss, value_loss)
        
        # Clear memory
        self.memory.clear()
    
    def _calculate_discounted_rewards(self, rewards):
        """Calculate discounted rewards"""
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        
        return discounted
    
    def _update_weights(self, states, actions, policy_loss, value_loss):
        """Update network weights (simplified)"""
        # This is a placeholder - in practice, use proper gradient descent
        learning_rate = self.lr
        
        # Simple weight updates
        policy_gradient = np.random.randn(*self.policy_weights.shape) * 0.001
        value_gradient = np.random.randn(*self.value_weights.shape) * 0.001
        
        self.policy_weights -= learning_rate * policy_gradient
        self.value_weights -= learning_rate * value_gradient
    
    def save_model(self, filepath):
        """Save model weights"""
        model_data = {
            'policy_weights': self.policy_weights,
            'value_weights': self.value_weights,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        np.save(filepath, model_data)
    
    def load_model(self, filepath):
        """Load model weights"""
        try:
            model_data = np.load(filepath, allow_pickle=True).item()
            self.policy_weights = model_data['policy_weights']
            self.value_weights = model_data['value_weights']
            self.model_loaded = True
            return True
        except:
            return False
    
    def set_training_mode(self, training=True):
        """Set training mode"""
        self.is_training = training
    
    def get_training_stats(self):
        """Get training statistics"""
        if len(self.episode_rewards) == 0:
            return {'avg_reward': 0, 'avg_length': 0}
        
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'episodes': len(self.episode_rewards)
        }


class PPOMemory:
    """Memory buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
    
    def store(self, state, action, reward, next_state, done, action_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.action_probs.append(action_prob)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.action_probs.clear()


class GbicrIntelligentAgent:
    """Intelligent agent wrapper for GBICR routing"""
    
    def __init__(self, state_extractor, max_neighbors=10, pretrained_model_path=None):
        self.state_extractor = state_extractor
        self.max_neighbors = max_neighbors
        
        # Initialize PPO agent
        state_dim = state_extractor.get_state_dimension()
        action_dim = max_neighbors  # maximum possible actions
        
        self.ppo_agent = PPOAgent(state_dim, action_dim)
        
        # Load pretrained model if available
        if pretrained_model_path:
            self.load_pretrained_model(pretrained_model_path)
        
        # Decision history for learning
        self.decision_history = []
        self.current_episode_reward = 0
    
    def select_next_hop(self, my_drone, dst_drone, neighbor_table, q_table=None):
        """
        Select next hop using PPO agent
        
        Args:
            my_drone: current drone
            dst_drone: destination drone
            neighbor_table: available neighbors
            q_table: Q-values (optional)
            
        Returns:
            selected_neighbor_id: chosen next hop
        """
        # Extract state features
        state = self.state_extractor.extract_state(my_drone, dst_drone, neighbor_table, q_table)
        
        # Get available actions (neighbor IDs)
        available_actions = list(neighbor_table.keys())
        
        if len(available_actions) == 0:
            return my_drone.identifier  # no neighbors available
        
        # Use PPO to select action
        selected_action, action_prob = self.ppo_agent.select_action(
            state, available_actions, exploration=self.ppo_agent.is_training
        )
        
        # Store decision for potential learning
        if self.ppo_agent.is_training:
            self.decision_history.append({
                'state': state,
                'action': selected_action,
                'action_prob': action_prob,
                'timestamp': my_drone.simulator.env.now
            })
        
        return selected_action if selected_action is not None else my_drone.identifier
    
    def update_reward(self, reward, done=False):
        """Update reward for the last decision"""
        if len(self.decision_history) > 0 and self.ppo_agent.is_training:
            last_decision = self.decision_history[-1]
            
            # Store transition in PPO memory
            self.ppo_agent.store_transition(
                last_decision['state'],
                last_decision['action'],
                reward,
                None,  # next_state will be filled later
                done,
                last_decision['action_prob']
            )
            
            self.current_episode_reward += reward
            
            if done:
                self.ppo_agent.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                self.decision_history.clear()
                
                # Update PPO policy
                self.ppo_agent.update()
    
    def load_pretrained_model(self, model_path):
        """Load pretrained PPO model"""
        success = self.ppo_agent.load_model(model_path)
        if success:
            self.ppo_agent.set_training_mode(False)  # Set to inference mode
        return success
    
    def save_model(self, model_path):
        """Save current PPO model"""
        self.ppo_agent.save_model(model_path)
    
    def set_training_mode(self, training=True):
        """Set training mode"""
        self.ppo_agent.set_training_mode(training)
    
    def get_training_stats(self):
        """Get training statistics"""
        return self.ppo_agent.get_training_stats()