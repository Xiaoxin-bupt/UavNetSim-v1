#!/usr/bin/env python3
"""
GBICR (Geographic Beacon-based Intelligent Collaborative Routing) Package

This package provides the GBICR routing protocol implementation with
intelligent decision making capabilities.

Components:
- gbicr: PPO-based routing protocol
- gbicr_agent: PPO agent implementation
- gbicr_state: State extraction
- gbicr_table: Advanced neighbor table
- gbicr_packet: Complete packet definitions
- gbicr_config: Configuration system

Author: AI Assistant
Created at: 2025/1/20
"""

# Initialize exports list
__all__ = []

# GBICR implementation imports
try:
    from .gbicr import GBICR
    from .gbicr_agent import GbicrIntelligentAgent
    from .gbicr_state import GbicrStateExtractor
    from .gbicr_table import GbicrTable
    from .gbicr_packet import (
        GbicrHelloPacket,
        GbicrBeaconPacket,
        GbicrAckPacket,
        GbicrDataPacket
    )
    from . import gbicr_config
    
    # Set default alias
    Gbicr = GBICR
    
    # Add to exports
    __all__.extend([
        'GBICR',
        'Gbicr',  # Default alias
        'GbicrIntelligentAgent',
        'GbicrStateExtractor', 
        'GbicrTable',
        'GbicrHelloPacket',
        'GbicrBeaconPacket',
        'GbicrAckPacket',
        'GbicrDataPacket',
        'gbicr_config',
    ])
    
except ImportError as e:
    print(f"Warning: Could not import GBICR components: {e}")
    pass

# Version information
__version__ = '2.0.0-simple'
__author__ = 'AI Assistant'
__description__ = 'Geographic Beacon-based Intelligent Collaborative Routing Protocol'

# Usage examples in docstring
__doc__ += """

Usage Examples:

1. Basic usage:
    from routing.gbicr import GBICR
    
    routing = GBICR(simulator, my_drone)
    has_route, packet, enquire = routing.next_hop_selection(packet)

2. Using default alias:
    from routing.gbicr import Gbicr
    
    routing = Gbicr(simulator, my_drone)
    has_route, packet, enquire = routing.next_hop_selection(packet)

3. With pretrained model:
    from routing.gbicr import GBICR
    
    routing = GBICR(simulator, my_drone, pretrained_model_path="model.pth")
"""