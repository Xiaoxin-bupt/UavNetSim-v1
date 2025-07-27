#!/usr/bin/env python3
"""
GBICR Protocol Usage Example

This script demonstrates how to integrate and use the GBICR protocol
in UavNetSim-v1 simulations.

Example scenarios:
1. Basic GBICR setup
2. Training mode simulation
3. Inference mode simulation
4. Performance comparison with other protocols
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import simulation components (these would be actual imports in UavNetSim-v1)
try:
    from src.simulation import Simulation
    from src.drone import Drone
    from src.environment import Environment
except ImportError:
    # Mock imports for demonstration
    class Simulation:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            pass
    
    class Drone:
        def __init__(self, *args, **kwargs):
            self.id = kwargs.get('id', 0)
            self.position = [0, 0, 0]
            self.velocity = [0, 0, 0]
    
    class Environment:
        def __init__(self, *args, **kwargs):
            pass

# Import GBICR components
from gbicr import Gbicr
from gbicr_config import get_gbicr_config, update_gbicr_config
from train_gbicr import GbicrTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GbicrSimulationExample:
    """Example class showing GBICR usage in simulations"""
    
    def __init__(self):
        self.config = get_gbicr_config()
        self.simulation = None
        self.drones = []
        self.gbicr_instances = []
    
    def setup_basic_simulation(self, num_drones=10, map_size=(1000, 1000, 100)):
        """Setup a basic simulation with GBICR protocol"""
        logger.info(f"Setting up simulation with {num_drones} drones")
        
        # Create simulation environment
        self.simulation = Simulation(
            map_size=map_size,
            sim_time=300,  # 5 minutes
            time_step=0.1
        )
        
        # Create drones
        for i in range(num_drones):
            # Random initial positions
            x = np.random.uniform(0, map_size[0])
            y = np.random.uniform(0, map_size[1])
            z = np.random.uniform(10, map_size[2])
            
            # Random initial velocities
            vx = np.random.uniform(-20, 20)
            vy = np.random.uniform(-20, 20)
            vz = np.random.uniform(-5, 5)
            
            drone = Drone(
                id=i,
                position=[x, y, z],
                velocity=[vx, vy, vz]
            )
            
            self.drones.append(drone)
        
        # Initialize GBICR for each drone
        for drone in self.drones:
            gbicr = Gbicr(
                drone=drone,
                simulator=self.simulation
            )
            self.gbicr_instances.append(gbicr)
        
        logger.info("Basic simulation setup completed")
    
    def run_training_simulation(self, episodes=100):
        """Run simulation in training mode to collect experience"""
        logger.info(f"Running training simulation for {episodes} episodes")
        
        # Set all GBICR instances to training mode
        for gbicr in self.gbicr_instances:
            gbicr.agent.set_training_mode(True)
        
        training_rewards = []
        
        for episode in range(episodes):
            logger.info(f"Training episode {episode + 1}/{episodes}")
            
            # Reset simulation
            self._reset_simulation()
            
            # Run episode
            episode_reward = self._run_episode()
            training_rewards.append(episode_reward)
            
            # Update agents
            for gbicr in self.gbicr_instances:
                gbicr.agent.update()
            
            if episode % 10 == 0:
                avg_reward = np.mean(training_rewards[-10:])
                logger.info(f"Average reward (last 10 episodes): {avg_reward:.3f}")
        
        # Save trained models
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        
        for i, gbicr in enumerate(self.gbicr_instances):
            model_path = model_dir / f"gbicr_drone_{i}.npy"
            gbicr.agent.save_model(str(model_path))
        
        logger.info("Training simulation completed")
        return training_rewards
    
    def run_inference_simulation(self, model_path=None):
        """Run simulation in inference mode with pre-trained models"""
        logger.info("Running inference simulation")
        
        # Load pre-trained models if available
        if model_path:
            for i, gbicr in enumerate(self.gbicr_instances):
                drone_model_path = f"{model_path}_drone_{i}.npy"
                if os.path.exists(drone_model_path):
                    gbicr.agent.load_model(drone_model_path)
                    logger.info(f"Loaded model for drone {i}")
        
        # Set all GBICR instances to inference mode
        for gbicr in self.gbicr_instances:
            gbicr.agent.set_training_mode(False)
        
        # Run simulation
        results = self._run_episode(collect_metrics=True)
        
        logger.info("Inference simulation completed")
        return results
    
    def compare_with_baseline(self, baseline_protocol="qgeo"):
        """Compare GBICR performance with baseline protocol"""
        logger.info(f"Comparing GBICR with {baseline_protocol}")
        
        # Run GBICR simulation
        gbicr_results = self.run_inference_simulation()
        
        # TODO: Implement baseline protocol comparison
        # This would involve running the same scenario with QGeo or other protocols
        
        logger.info("Performance comparison completed")
        return {
            'gbicr': gbicr_results,
            'baseline': {}  # Placeholder
        }
    
    def _reset_simulation(self):
        """Reset simulation state for new episode"""
        # Reset drone positions and velocities
        for drone in self.drones:
            # Random positions
            drone.position = [
                np.random.uniform(0, 1000),
                np.random.uniform(0, 1000),
                np.random.uniform(10, 100)
            ]
            
            # Random velocities
            drone.velocity = [
                np.random.uniform(-20, 20),
                np.random.uniform(-20, 20),
                np.random.uniform(-5, 5)
            ]
        
        # Reset GBICR tables
        for gbicr in self.gbicr_instances:
            gbicr.table.neighbor_table.clear()
            gbicr.table.beacon_table.clear()
            gbicr.table.collaborative_q_values.clear()
    
    def _run_episode(self, collect_metrics=False):
        """Run a single simulation episode"""
        episode_reward = 0
        metrics = {
            'packet_delivery_ratio': 0,
            'average_delay': 0,
            'routing_overhead': 0,
            'energy_consumption': 0
        } if collect_metrics else None
        
        # Simulate for specified time
        sim_steps = int(300 / 0.1)  # 300 seconds with 0.1s time step
        
        for step in range(sim_steps):
            # Update drone positions
            self._update_drone_positions()
            
            # Process GBICR protocols
            for gbicr in self.gbicr_instances:
                # Periodic hello broadcasts
                if step % int(self.config['hello_interval'] / 0.1) == 0:
                    gbicr._broadcast_hello()
                
                # Periodic beacon broadcasts
                if step % int(self.config['beacon_interval'] / 0.1) == 0:
                    gbicr._broadcast_beacon()
                
                # Process packet queue
                gbicr._process_packet_queue()
            
            # Generate random data packets
            if step % 50 == 0:  # Every 5 seconds
                self._generate_random_data_packet()
            
            # Collect metrics if requested
            if collect_metrics and step % 100 == 0:
                self._update_metrics(metrics)
        
        return metrics if collect_metrics else episode_reward
    
    def _update_drone_positions(self):
        """Update drone positions based on mobility model"""
        dt = 0.1  # time step
        
        for drone in self.drones:
            # Update position
            for i in range(3):
                drone.position[i] += drone.velocity[i] * dt
                
                # Boundary conditions
                if drone.position[i] < 0:
                    drone.position[i] = 0
                    drone.velocity[i] *= -1
                elif (i < 2 and drone.position[i] > 1000) or (i == 2 and drone.position[i] > 100):
                    drone.position[i] = 1000 if i < 2 else 100
                    drone.velocity[i] *= -1
            
            # Add random mobility variations
            for i in range(3):
                drone.velocity[i] += np.random.uniform(-1, 1)
                drone.velocity[i] = np.clip(drone.velocity[i], -30, 30)
    
    def _generate_random_data_packet(self):
        """Generate random data packets for testing"""
        if len(self.drones) < 2:
            return
        
        # Select random source and destination
        source_idx = np.random.randint(0, len(self.drones))
        dest_idx = np.random.randint(0, len(self.drones))
        
        while dest_idx == source_idx:
            dest_idx = np.random.randint(0, len(self.drones))
        
        source_drone = self.drones[source_idx]
        dest_drone = self.drones[dest_idx]
        
        # Create and send data packet through GBICR
        gbicr = self.gbicr_instances[source_idx]
        # gbicr.send_data_packet(dest_drone.id, "test_payload")
    
    def _update_metrics(self, metrics):
        """Update performance metrics"""
        # TODO: Implement actual metric collection
        # This would involve tracking packet delivery, delays, etc.
        pass


def example_basic_usage():
    """Example 1: Basic GBICR setup and usage"""
    print("\n=== Example 1: Basic GBICR Usage ===")
    
    # Create simulation
    sim_example = GbicrSimulationExample()
    sim_example.setup_basic_simulation(num_drones=5)
    
    # Run a short inference simulation
    results = sim_example.run_inference_simulation()
    print(f"Simulation completed with results: {results}")


def example_training():
    """Example 2: Training GBICR agents"""
    print("\n=== Example 2: Training GBICR Agents ===")
    
    # Create simulation
    sim_example = GbicrSimulationExample()
    sim_example.setup_basic_simulation(num_drones=3)
    
    # Run training
    rewards = sim_example.run_training_simulation(episodes=10)
    print(f"Training completed. Final average reward: {np.mean(rewards[-5:]):.3f}")


def example_configuration():
    """Example 3: Custom configuration"""
    print("\n=== Example 3: Custom Configuration ===")
    
    # Update configuration
    custom_config = {
        'hello_interval': 1.0,  # More frequent hello packets
        'beacon_interval': 3.0,  # More frequent beacons
        'ppo_lr': 0.001,  # Higher learning rate
        'max_neighbors': 15  # More neighbors
    }
    
    update_gbicr_config(custom_config)
    
    # Create simulation with custom config
    sim_example = GbicrSimulationExample()
    sim_example.setup_basic_simulation(num_drones=8)
    
    print(f"Simulation created with custom configuration: {custom_config}")


def example_standalone_training():
    """Example 4: Standalone training without full simulation"""
    print("\n=== Example 4: Standalone Training ===")
    
    # Create trainer
    config = get_gbicr_config()
    trainer = GbicrTrainer(config)
    
    # Train for a few episodes
    trainer.train(num_episodes=20, save_path="./models/example_model.npy")
    
    # Evaluate
    avg_reward, success_rate = trainer.evaluate(num_episodes=10)
    print(f"Training completed. Avg reward: {avg_reward:.3f}, Success rate: {success_rate:.3f}")


def main():
    """Run all examples"""
    print("GBICR Protocol Usage Examples")
    print("=============================")
    
    try:
        example_basic_usage()
        example_training()
        example_configuration()
        example_standalone_training()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()