#!/usr/bin/env python3
"""
GBICR Training Script

This script demonstrates how to train the GBICR PPO agent in different scenarios.
It includes offline training with various network topologies and mobility patterns.

Usage:
    python train_gbicr.py --episodes 1000 --save_path ./models/gbicr_trained.npy
"""

import argparse
import logging
import numpy as np
import random
import time
from collections import deque

# Import GBICR components
from .gbicr_config import get_gbicr_config, get_training_config, get_training_env_config
from .gbicr_state import GbicrStateExtractor
from .gbicr_agent import GbicrIntelligentAgent, PPOAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gbicr_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GbicrTrainingEnvironment:
    """Simplified training environment for GBICR"""
    
    def __init__(self, config):
        self.config = config
        self.map_size = config['map_size']
        self.reset()
    
    def reset(self, num_drones=None, mobility_model=None):
        """Reset environment for new episode"""
        self.num_drones = num_drones or random.choice(self.config['num_drones'])
        self.mobility_model = mobility_model or random.choice(self.config['mobility_models'])
        
        # Initialize drone positions randomly
        self.drone_positions = []
        self.drone_velocities = []
        
        for i in range(self.num_drones):
            pos = [
                random.uniform(0, self.map_size[0]),
                random.uniform(0, self.map_size[1]),
                random.uniform(10, self.map_size[2])
            ]
            vel = [
                random.uniform(-20, 20),  # m/s
                random.uniform(-20, 20),
                random.uniform(-5, 5)
            ]
            self.drone_positions.append(pos)
            self.drone_velocities.append(vel)
        
        self.current_step = 0
        self.max_steps = 100
        
        return self._get_state()
    
    def step(self, action, current_drone_id, destination_id):
        """Execute one step in the environment"""
        # Update drone positions based on mobility model
        self._update_positions()
        
        # Calculate reward based on action
        reward = self._calculate_reward(action, current_drone_id, destination_id)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        """Get current state representation"""
        # Simplified state: positions and velocities of all drones
        state = []
        for pos, vel in zip(self.drone_positions, self.drone_velocities):
            state.extend(pos + vel)
        
        # Pad or truncate to fixed size
        max_drones = 20
        while len(state) < max_drones * 6:
            state.append(0.0)
        
        return np.array(state[:max_drones * 6], dtype=np.float32)
    
    def _update_positions(self):
        """Update drone positions based on mobility model"""
        dt = 0.1  # time step in seconds
        
        for i in range(self.num_drones):
            # Update position
            for j in range(3):
                self.drone_positions[i][j] += self.drone_velocities[i][j] * dt
                
                # Boundary conditions
                if self.drone_positions[i][j] < 0:
                    self.drone_positions[i][j] = 0
                    self.drone_velocities[i][j] *= -1
                elif self.drone_positions[i][j] > self.map_size[j]:
                    self.drone_positions[i][j] = self.map_size[j]
                    self.drone_velocities[i][j] *= -1
            
            # Add mobility model variations
            if self.mobility_model == 'random_walk':
                for j in range(3):
                    self.drone_velocities[i][j] += random.uniform(-2, 2)
                    self.drone_velocities[i][j] = max(-30, min(30, self.drone_velocities[i][j]))
            
            elif self.mobility_model == 'gauss_markov':
                alpha = 0.5  # memory factor
                for j in range(3):
                    self.drone_velocities[i][j] = (
                        alpha * self.drone_velocities[i][j] + 
                        (1 - alpha) * random.uniform(-20, 20)
                    )
    
    def _calculate_reward(self, action, current_drone_id, destination_id):
        """Calculate reward for the action"""
        if action >= self.num_drones or action == current_drone_id:
            return -1.0  # Invalid action
        
        # Distance-based reward
        current_pos = self.drone_positions[current_drone_id]
        next_hop_pos = self.drone_positions[action]
        dest_pos = self.drone_positions[destination_id]
        
        # Geographic progress
        current_to_dest = np.linalg.norm(np.array(current_pos) - np.array(dest_pos))
        next_hop_to_dest = np.linalg.norm(np.array(next_hop_pos) - np.array(dest_pos))
        
        progress = (current_to_dest - next_hop_to_dest) / 1000.0  # normalize
        
        # Link quality (distance-based)
        link_distance = np.linalg.norm(np.array(current_pos) - np.array(next_hop_pos))
        link_quality = max(0.0, 1.0 - link_distance / 500.0)  # 500m max range
        
        reward = progress + 0.3 * link_quality
        
        return reward


class GbicrTrainer:
    """GBICR training manager"""
    
    def __init__(self, config):
        self.config = config
        self.env = GbicrTrainingEnvironment(get_training_env_config())
        
        # Initialize state extractor and agent
        self.state_extractor = GbicrStateExtractor(max_neighbors=config['max_neighbors'])
        state_dim = 120  # Simplified state dimension
        action_dim = config['max_neighbors']
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config['ppo_lr'],
            gamma=config['ppo_gamma'],
            eps_clip=config['ppo_eps_clip'],
            k_epochs=config['ppo_k_epochs']
        )
        
        self.agent.set_training_mode(True)
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
    
    def train(self, num_episodes, save_path=None):
        """Train the GBICR agent"""
        logger.info(f"Starting GBICR training for {num_episodes} episodes")
        
        best_avg_reward = float('-inf')
        
        for episode in range(num_episodes):
            episode_reward, episode_length, success_rate = self._run_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rates.append(success_rate)
            
            # Update agent
            self.agent.update()
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_success = np.mean(self.success_rates)
                
                logger.info(
                    f"Episode {episode}: Avg Reward: {avg_reward:.3f}, "
                    f"Avg Length: {avg_length:.1f}, Success Rate: {avg_success:.3f}"
                )
                
                # Save best model
                if avg_reward > best_avg_reward and save_path:
                    best_avg_reward = avg_reward
                    self.agent.save_model(save_path)
                    logger.info(f"New best model saved with avg reward: {avg_reward:.3f}")
            
            # Early stopping
            if len(self.success_rates) >= 50 and np.mean(self.success_rates) > 0.95:
                logger.info(f"Early stopping at episode {episode} with success rate > 95%")
                break
        
        logger.info("Training completed")
        
        if save_path:
            self.agent.save_model(save_path)
            logger.info(f"Final model saved to {save_path}")
    
    def _run_episode(self):
        """Run a single training episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        successful_transmissions = 0
        total_transmissions = 0
        
        while True:
            # Select random source and destination
            current_drone = random.randint(0, self.env.num_drones - 1)
            destination = random.randint(0, self.env.num_drones - 1)
            
            if current_drone == destination:
                continue
            
            # Get available neighbors (simplified)
            available_actions = list(range(self.env.num_drones))
            available_actions.remove(current_drone)
            
            if len(available_actions) == 0:
                break
            
            # Select action
            action, action_prob = self.agent.select_action(state, available_actions)
            
            # Execute action
            next_state, reward, done, _ = self.env.step(action, current_drone, destination)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done, action_prob)
            
            episode_reward += reward
            episode_length += 1
            total_transmissions += 1
            
            if reward > 0:
                successful_transmissions += 1
            
            state = next_state
            
            if done:
                break
        
        success_rate = successful_transmissions / total_transmissions if total_transmissions > 0 else 0
        
        return episode_reward, episode_length, success_rate
    
    def evaluate(self, num_episodes=100):
        """Evaluate the trained agent"""
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        self.agent.set_training_mode(False)
        
        eval_rewards = []
        eval_success_rates = []
        
        for episode in range(num_episodes):
            episode_reward, _, success_rate = self._run_episode()
            eval_rewards.append(episode_reward)
            eval_success_rates.append(success_rate)
        
        avg_reward = np.mean(eval_rewards)
        avg_success_rate = np.mean(eval_success_rates)
        
        logger.info(f"Evaluation Results: Avg Reward: {avg_reward:.3f}, Success Rate: {avg_success_rate:.3f}")
        
        self.agent.set_training_mode(True)
        
        return avg_reward, avg_success_rate


def main():
    parser = argparse.ArgumentParser(description='Train GBICR PPO Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save_path', type=str, default='./models/gbicr_trained.npy', help='Path to save trained model')
    parser.add_argument('--config', type=str, default='default', help='Configuration preset to use')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_gbicr_config()
    
    # Create trainer
    trainer = GbicrTrainer(config)
    
    # Train agent
    start_time = time.time()
    trainer.train(args.episodes, args.save_path)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate agent
    trainer.evaluate(args.eval_episodes)


if __name__ == '__main__':
    main()