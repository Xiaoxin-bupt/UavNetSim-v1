"""GBICR Configuration Module

This module contains configuration parameters and utility functions for the GBICR routing protocol.
"""

import os

# GBICR Protocol Parameters
GBICR_CONFIG = {
    # Timing parameters (in microseconds)
    'hello_interval': 0.5 * 1e6,  # 500ms
    'beacon_interval': 2.0 * 1e6,  # 2s
    'check_interval': 0.6 * 1e6,   # 600ms
    
    # Learning parameters
    'learning_rate': 0.3,
    'reward_max': 10.0,
    'reward_min': -10.0,
    'exploration_rate': 0.1,
    
    # PPO Agent parameters
    'ppo_lr': 3e-4,
    'ppo_gamma': 0.99,
    'ppo_eps_clip': 0.2,
    'ppo_k_epochs': 4,
    'ppo_batch_size': 32,
    
    # State space parameters
    'max_neighbors': 10,
    'state_dimension': None,  # Will be calculated automatically
    
    # Network parameters
    'entry_lifetime': 3 * 1e6,    # 3s for neighbor entries
    'beacon_lifetime': 5 * 1e6,   # 5s for beacon information
    'stability_window': 10,       # samples for stability calculation
    
    # Reward calculation weights
    'geographic_weight': 0.4,
    'collaborative_weight': 0.3,
    'link_quality_weight': 0.2,
    'stability_weight': 0.1,
    
    # Model paths
    'pretrained_model_path': None,
    'model_save_path': './models/gbicr_model.npy',
    'training_log_path': './logs/gbicr_training.log',
}

# Training Configuration
TRAINING_CONFIG = {
    'training_episodes': 1000,
    'evaluation_episodes': 100,
    'save_interval': 50,  # episodes
    'log_interval': 10,   # episodes
    'early_stopping_patience': 100,
    'target_success_rate': 0.95,
}

# Environment Configuration for Training
TRAINING_ENV_CONFIG = {
    'map_size': (600, 600, 100),  # (length, width, height) in meters
    'num_drones': [5, 10, 15, 20],  # different scenarios
    'mobility_models': ['random_walk', 'gauss_markov', 'random_waypoint'],
    'traffic_patterns': ['uniform', 'hotspot', 'random'],
    'simulation_time': 30 * 1e6,  # 30 seconds
}


def get_gbicr_config():
    """Get GBICR configuration dictionary"""
    return GBICR_CONFIG.copy()


def get_training_config():
    """Get training configuration dictionary"""
    return TRAINING_CONFIG.copy()


def get_training_env_config():
    """Get training environment configuration dictionary"""
    return TRAINING_ENV_CONFIG.copy()


def update_config(config_dict, **kwargs):
    """Update configuration with new parameters"""
    config_dict.update(kwargs)
    return config_dict


def create_model_directory(model_path):
    """Create directory for model saving"""
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)


def create_log_directory(log_path):
    """Create directory for logging"""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)


def validate_config(config):
    """Validate configuration parameters"""
    errors = []
    
    # Check timing parameters
    if config.get('hello_interval', 0) <= 0:
        errors.append("hello_interval must be positive")
    
    if config.get('beacon_interval', 0) <= 0:
        errors.append("beacon_interval must be positive")
    
    # Check learning parameters
    lr = config.get('learning_rate', 0)
    if not (0 < lr <= 1):
        errors.append("learning_rate must be between 0 and 1")
    
    # Check reward parameters
    r_max = config.get('reward_max', 0)
    r_min = config.get('reward_min', 0)
    if r_max <= r_min:
        errors.append("reward_max must be greater than reward_min")
    
    # Check PPO parameters
    ppo_lr = config.get('ppo_lr', 0)
    if ppo_lr <= 0:
        errors.append("ppo_lr must be positive")
    
    gamma = config.get('ppo_gamma', 0)
    if not (0 < gamma <= 1):
        errors.append("ppo_gamma must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))
    
    return True


# Example usage configurations
EXAMPLE_CONFIGS = {
    'default': get_gbicr_config(),
    
    'high_mobility': update_config(
        get_gbicr_config(),
        hello_interval=0.3 * 1e6,  # faster hello for high mobility
        exploration_rate=0.15,     # more exploration
        stability_weight=0.15      # higher stability weight
    ),
    
    'dense_network': update_config(
        get_gbicr_config(),
        max_neighbors=15,          # more neighbors
        beacon_interval=1.5 * 1e6, # faster beacons
        collaborative_weight=0.4   # higher collaboration weight
    ),
    
    'sparse_network': update_config(
        get_gbicr_config(),
        hello_interval=0.8 * 1e6,  # slower hello for sparse network
        geographic_weight=0.5,     # higher geographic weight
        exploration_rate=0.2       # more exploration
    ),
    
    'training_mode': update_config(
        get_gbicr_config(),
        exploration_rate=0.3,      # high exploration for training
        ppo_lr=5e-4,              # higher learning rate
        ppo_k_epochs=6            # more update epochs
    )
}


def get_example_config(config_name):
    """Get a predefined example configuration"""
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(EXAMPLE_CONFIGS.keys())}")
    
    return EXAMPLE_CONFIGS[config_name].copy()