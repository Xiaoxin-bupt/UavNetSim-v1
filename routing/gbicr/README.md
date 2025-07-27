# GBICR (Geographic Beacon-based Intelligent Collaborative Routing) Protocol

## Overview

GBICR is an intelligent routing protocol for UAV networks that combines geographic routing with reinforcement learning (PPO) and collaborative decision-making. It is designed to work with the UavNetSim-v1 simulation platform.

## Key Features

- **PPO-based Intelligent Routing**: Uses Proximal Policy Optimization for next-hop selection
- **Geographic Beacon System**: Maintains global network awareness through beacon packets
- **Collaborative Q-value Sharing**: Neighbors share Q-values for better routing decisions
- **Link Quality Assessment**: Multi-dimensional link evaluation including stability and quality
- **Energy-Free State Space**: Focuses on position, velocity, and network topology features

## Architecture

### Core Components

1. **gbicr.py**: Main protocol implementation
2. **gbicr_packet.py**: Packet definitions (Hello, Beacon, ACK, Data)
3. **gbicr_table.py**: Routing table and neighbor management
4. **gbicr_agent.py**: PPO agent and intelligent decision-making
5. **gbicr_state.py**: State feature extraction for PPO
6. **gbicr_config.py**: Configuration parameters
7. **train_gbicr.py**: Training script for PPO agent

### Packet Types

- **GbicrHelloPacket**: Neighbor discovery and maintenance
- **GbicrBeaconPacket**: Geographic information sharing
- **GbicrAckPacket**: Acknowledgments with reward feedback
- **GbicrDataPacket**: Routing metadata for data packets

## Installation

1. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. The GBICR module is located in `routing/gbicr/` directory

## Configuration

### Basic Configuration

Edit `gbicr_config.py` to adjust protocol parameters:

```python
# Timing parameters
HELLO_INTERVAL = 2.0  # seconds
BEACON_INTERVAL = 5.0  # seconds

# PPO parameters
PPO_LR = 0.0003
PPO_GAMMA = 0.99
PPO_EPS_CLIP = 0.2

# State space configuration
MAX_NEIGHBORS = 10
STATE_NORMALIZE = True
```

### Training Configuration

For training the PPO agent:

```python
# Training parameters
TRAINING_EPISODES = 1000
BATCH_SIZE = 64
UPDATE_FREQUENCY = 10
```

## Usage

### 1. Training the PPO Agent

Before using GBICR in simulations, train the PPO agent:

```bash
cd routing/gbicr
python train_gbicr.py --episodes 1000 --save_path ./models/gbicr_trained.npy
```

Training options:
- `--episodes`: Number of training episodes (default: 1000)
- `--save_path`: Path to save the trained model
- `--config`: Configuration preset to use
- `--eval_episodes`: Number of evaluation episodes

### 2. Using GBICR in Simulations

To use GBICR in UavNetSim-v1:

1. Import the GBICR protocol:
```python
from routing.gbicr.gbicr import Gbicr
```

2. Initialize the protocol for each drone:
```python
# In your drone initialization code
routing_protocol = Gbicr(
    drone=drone_instance,
    simulator=sim_instance
)
```

3. The protocol will automatically:
   - Load the pre-trained PPO model
   - Start periodic hello and beacon broadcasts
   - Handle packet routing using intelligent next-hop selection

### 3. Model Management

The protocol supports both training and inference modes:

```python
# Training mode (for collecting experience)
routing_protocol.agent.set_training_mode(True)

# Inference mode (for deployment)
routing_protocol.agent.set_training_mode(False)
```

## State Space Design

The GBICR state space includes:

### Current Node Features
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Number of neighbors

### Destination Features
- Position (x, y, z)
- Distance to destination

### Neighbor Features (for each neighbor)
- Relative position
- Relative velocity
- Link quality
- Link stability
- Last update time

### Network Features
- Network density
- Connectivity metrics
- Topology stability

**Note**: Energy factors are explicitly excluded from the state space as per design requirements.

## Reward System

The PPO agent uses a multi-component reward function:

1. **Geographic Progress**: Reward for moving closer to destination
2. **Link Quality**: Reward for selecting high-quality links
3. **Stability**: Reward for choosing stable neighbors
4. **Collaboration**: Reward based on neighbor Q-value feedback
5. **Penalties**: For void areas, failed transmissions, or invalid actions

## Performance Tuning

### PPO Hyperparameters

- **Learning Rate**: Start with 0.0003, adjust based on convergence
- **Gamma**: 0.99 for long-term reward consideration
- **Epsilon Clip**: 0.2 for stable policy updates
- **K Epochs**: 4-10 for sufficient policy improvement

### Network Parameters

- **Hello Interval**: 2-5 seconds depending on mobility
- **Beacon Interval**: 5-10 seconds for global awareness
- **Max Neighbors**: 10-20 based on network density

### Training Tips

1. **Diverse Scenarios**: Train with various network topologies and mobility patterns
2. **Curriculum Learning**: Start with simple scenarios, gradually increase complexity
3. **Early Stopping**: Monitor success rate and stop when > 95%
4. **Model Validation**: Regular evaluation during training

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the trained model exists at the specified path
2. **Slow Convergence**: Adjust learning rate or increase training episodes
3. **Poor Performance**: Check state normalization and reward function
4. **Memory Issues**: Reduce batch size or state dimension

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

The training script provides:
- Episode rewards and success rates
- Average episode length
- Model checkpointing
- Training logs

## Integration with UavNetSim-v1

GBICR is designed to integrate seamlessly with the existing UavNetSim-v1 platform:

1. **Packet Handling**: Compatible with existing packet processing pipeline
2. **Event System**: Uses SimPy events for timing and scheduling
3. **Configuration**: Follows the same configuration pattern as other protocols
4. **Metrics**: Compatible with existing performance measurement tools

## Future Enhancements

- **Multi-objective Optimization**: Balance multiple routing objectives
- **Federated Learning**: Distributed training across multiple UAVs
- **Dynamic Adaptation**: Real-time hyperparameter adjustment
- **Security Features**: Secure routing and authentication mechanisms

## References

- Proximal Policy Optimization (PPO) Algorithm
- Geographic Routing in Mobile Ad-hoc Networks
- UAV Network Simulation and Modeling
- Collaborative Routing Protocols

## License

This implementation is part of the UavNetSim-v1 project and follows the same licensing terms.