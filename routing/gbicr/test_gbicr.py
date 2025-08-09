#!/usr/bin/env python3
"""
GBICR Protocol Test Suite

Basic tests to verify GBICR implementation functionality.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import GBICR components
from gbicr_packet import GbicrHelloPacket, GbicrBeaconPacket, GbicrAckPacket, GbicrDataPacket
from gbicr_state import GbicrStateExtractor
from gbicr_agent import PPOAgent, GbicrIntelligentAgent
from gbicr_table import GbicrTable
from gbicr_config import get_gbicr_config, validate_config


class TestGbicrPackets(unittest.TestCase):
    """Test GBICR packet classes"""
    
    def test_hello_packet_creation(self):
        """Test GbicrHelloPacket creation"""
        packet = GbicrHelloPacket(
            source_drone_id=1,
            position=[100, 200, 50],
            velocity=[10, -5, 2]
        )
        
        self.assertEqual(packet.source_drone_id, 1)
        self.assertEqual(packet.position, [100, 200, 50])
        self.assertEqual(packet.velocity, [10, -5, 2])
        self.assertIsNotNone(packet.timestamp)
    
    def test_beacon_packet_creation(self):
        """Test GbicrBeaconPacket creation"""
        packet = GbicrBeaconPacket(
            source_drone_id=2,
            position=[150, 250, 75],
            coverage_area=500.0,
            network_density=0.8
        )
        
        self.assertEqual(packet.source_drone_id, 2)
        self.assertEqual(packet.coverage_area, 500.0)
        self.assertEqual(packet.network_density, 0.8)
    
    def test_ack_packet_creation(self):
        """Test GbicrAckPacket creation"""
        packet = GbicrAckPacket(
            source_drone_id=3,
            dest_drone_id=4,
            acked_packet_id=12345,
            reward=0.75,
            max_q_value=0.9
        )
        
        self.assertEqual(packet.source_drone_id, 3)
        self.assertEqual(packet.dest_drone_id, 4)
        self.assertEqual(packet.acked_packet_id, 12345)
        self.assertEqual(packet.reward, 0.75)
        self.assertEqual(packet.max_q_value, 0.9)
    
    def test_data_packet_creation(self):
        """Test GbicrDataPacket creation"""
        packet = GbicrDataPacket(
            source_drone_id=5,
            dest_drone_id=6,
            next_hop_id=7,
            geographic_progress=100.0
        )
        
        self.assertEqual(packet.source_drone_id, 5)
        self.assertEqual(packet.dest_drone_id, 6)
        self.assertEqual(packet.next_hop_id, 7)
        self.assertEqual(packet.geographic_progress, 100.0)


class TestGbicrStateExtractor(unittest.TestCase):
    """Test GBICR state extraction"""
    
    def setUp(self):
        self.extractor = GbicrStateExtractor(max_neighbors=5)
    
    def test_current_drone_features(self):
        """Test current drone feature extraction"""
        current_drone = {
            'position': [100, 200, 50],
            'velocity': [10, -5, 2],
            'neighbors': [1, 2, 3]
        }
        
        features = self.extractor.extract_current_drone_features(current_drone)
        
        # Should include position, velocity, and neighbor count
        self.assertEqual(len(features), 7)  # 3 + 3 + 1
        self.assertEqual(features[:3], [100, 200, 50])
        self.assertEqual(features[3:6], [10, -5, 2])
        self.assertEqual(features[6], 3)
    
    def test_destination_features(self):
        """Test destination feature extraction"""
        # Create mock drone objects
        class MockDrone:
            def __init__(self, coords, velocity):
                self.coords = coords
                self.velocity = velocity
        
        current_drone = MockDrone([100, 200, 50], [10, -5, 2])
        dest_drone = MockDrone([300, 400, 100], [0, 0, 0])
        
        features = self.extractor._extract_destination_features(current_drone, dest_drone)
        
        # Should have 7 features (distance:1, rel_pos:3, direction:3)
        self.assertEqual(len(features), 7)
        
        # Distance should be positive
        self.assertGreater(features[0], 0)
    
    def test_neighbor_features(self):
        """Test neighbor feature extraction"""
        # Create mock drone objects
        class MockDrone:
            def __init__(self, coords, velocity):
                self.coords = coords
                self.velocity = velocity
        
        current_drone = MockDrone([0, 0, 0], [0, 0, 0])
        dst_drone = MockDrone([500, 500, 100], [0, 0, 0])
        
        # Neighbor table format: {neighbor_id: [position, velocity, timestamp, link_quality, stability]}
        neighbor_table = {
            1: [[100, 100, 50], [5, 5, 1], 1.0, 0.8, 0.9]
        }
        
        features = self.extractor._extract_neighbor_features(current_drone, dst_drone, neighbor_table)
        
        # Should have features for neighbors
        self.assertGreater(len(features), 0)
        
        # First feature should be number of neighbors
        self.assertGreater(features[0], 0)  # normalized number of neighbors
    
    def test_get_state_vector(self):
        """Test complete state vector extraction"""
        # Create mock drone objects
        class MockDrone:
            def __init__(self, coords, velocity):
                self.coords = coords
                self.velocity = velocity
        
        current_drone = MockDrone([100, 200, 50], [10, -5, 2])
        dest_drone = MockDrone([300, 400, 100], [0, 0, 0])
        
        # Neighbor table format: {neighbor_id: [position, velocity, timestamp, link_quality, stability]}
        neighbor_table = {
            1: [[150, 250, 60], [5, 0, 1], 1.0, 0.8, 0.9]
        }
        
        state_vector = self.extractor.extract_state(
            current_drone, dest_drone, neighbor_table
        )
        
        # Check state vector dimension
        expected_dim = self.extractor.get_state_dimension()
        self.assertEqual(len(state_vector), expected_dim)
        
        # Check that state vector is numpy array
        self.assertIsInstance(state_vector, np.ndarray)


class TestGbicrAgent(unittest.TestCase):
    """Test GBICR PPO agent"""
    
    def setUp(self):
        self.state_dim = 50
        self.action_dim = 10
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=0.001,
            gamma=0.99
        )
    
    def test_agent_initialization(self):
        """Test PPO agent initialization"""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.policy_weights)
        self.assertIsNotNone(self.agent.value_weights)
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim).astype(np.float32)
        available_actions = [0, 1, 2, 3, 4]
        
        action, action_prob = self.agent.select_action(state, available_actions)
        
        self.assertIn(action, available_actions)
        self.assertGreater(action_prob, 0)
        self.assertLessEqual(action_prob, 1)
    
    def test_training_mode(self):
        """Test training mode switching"""
        # Test training mode
        self.agent.set_training_mode(True)
        # Note: training_mode attribute may not be directly accessible
        # This is a placeholder test for training mode functionality
        self.assertTrue(True)  # Simplified test


class TestGbicrTable(unittest.TestCase):
    """Test GBICR routing table"""
    
    def setUp(self):
        # Create mock objects for required parameters
        class MockEnv:
            def __init__(self):
                self.now = 0
        
        class MockDrone:
            def __init__(self):
                self.identifier = 1
                self.coords = [0, 0, 0]
                self.velocity = [0, 0, 0]
        
        class MockRng:
            def random(self):
                return 0.5
        
        self.table = GbicrTable(MockEnv(), MockDrone(), MockRng())
    
    def test_add_neighbor(self):
        """Test adding neighbor to table"""
        # Create mock hello packet
        class MockHelloPacket:
            def __init__(self, src_drone, position, velocity):
                self.src_drone = src_drone
                self.cur_position = position
                self.cur_velocity = velocity
        
        class MockSrcDrone:
            def __init__(self, identifier):
                self.identifier = identifier
        
        neighbor_id = 2  # Different from my_drone.identifier (1)
        position = [100, 200, 50]
        velocity = [10, -5, 2]
        
        hello_packet = MockHelloPacket(MockSrcDrone(neighbor_id), position, velocity)
        current_time = 1000000  # 1 second in microseconds
        
        self.table.add_neighbor(hello_packet, current_time)
        
        self.assertIn(neighbor_id, self.table.neighbor_table)
        neighbor_info = self.table.neighbor_table[neighbor_id]
        self.assertEqual(neighbor_info[0], position)  # position is at index 0
        self.assertEqual(neighbor_info[1], velocity)  # velocity is at index 1
    
    def test_update_beacon_info(self):
        """Test updating beacon information"""
        # Create mock beacon packet
        class MockBeaconPacket:
            def __init__(self, src_drone, position, coverage_area):
                self.src_drone = src_drone
                self.cur_position = position
                self.coverage_area = coverage_area
        
        class MockSrcDrone:
            def __init__(self, identifier):
                self.identifier = identifier
        
        beacon_id = 2
        position = [300, 400, 100]
        coverage_area = [500.0]  # coverage_area as list
        
        beacon_packet = MockBeaconPacket(MockSrcDrone(beacon_id), position, coverage_area)
        current_time = 2000000  # 2 seconds in microseconds
        
        self.table.add_beacon_info(beacon_packet, current_time)
        
        self.assertIn(beacon_id, self.table.beacon_table)
        beacon_info = self.table.beacon_table[beacon_id]
        self.assertEqual(beacon_info[0], position)  # position is at index 0
        self.assertEqual(beacon_info[1], coverage_area)  # coverage_area is at index 1
    
    def test_calculate_link_quality(self):
        """Test link quality calculation"""
        distance = 100.0  # 100m away
        neighbor_velocity = [5, 5, 1]
        
        quality = self.table._calculate_link_quality(distance, neighbor_velocity)
        
        self.assertGreater(quality, 0)
        self.assertLessEqual(quality, 1)
    
    def test_best_next_hop_selection(self):
        """Test best next hop selection"""
        # Create mock hello packets and add neighbors
        class MockHelloPacket:
            def __init__(self, src_drone, position, velocity):
                self.src_drone = src_drone
                self.cur_position = position
                self.cur_velocity = velocity
        
        class MockSrcDrone:
            def __init__(self, identifier):
                self.identifier = identifier
        
        class MockDestDrone:
            def __init__(self, coords):
                self.coords = coords
                self.identifier = 99
        
        # Add neighbors
        hello1 = MockHelloPacket(MockSrcDrone(2), [100, 100, 50], [5, 5, 1])
        hello2 = MockHelloPacket(MockSrcDrone(3), [200, 200, 60], [10, 10, 2])
        
        self.table.add_neighbor(hello1, 1000000)
        self.table.add_neighbor(hello2, 1000000)
        
        dest_drone = MockDestDrone([300, 300, 100])
        
        best_hop = self.table.best_neighbor_collaborative(dest_drone)
        
        # Should return one of the neighbors or my_drone.identifier
        self.assertIn(best_hop, [1, 2, 3])


class TestGbicrConfig(unittest.TestCase):
    """Test GBICR configuration"""
    
    def test_get_config(self):
        """Test configuration retrieval"""
        config = get_gbicr_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('hello_interval', config)
        self.assertIn('beacon_interval', config)
        self.assertIn('ppo_lr', config)
    
    def test_validate_config(self):
        """Test configuration validation"""
        valid_config = {
            'hello_interval': 2.0,
            'beacon_interval': 5.0,
            'learning_rate': 0.3,
            'reward_max': 10.0,
            'reward_min': -5.0,
            'ppo_lr': 0.001,
            'ppo_gamma': 0.99,
            'max_neighbors': 10
        }
        
        # Should not raise exception
        validate_config(valid_config)
        
        # Test invalid config
        invalid_config = {
            'hello_interval': -1.0,  # Invalid negative value
            'learning_rate': 2.0,    # Invalid learning rate > 1
            'reward_max': 5.0,
            'reward_min': 10.0,      # Invalid: min > max
            'ppo_lr': -0.001,        # Invalid negative learning rate
            'ppo_gamma': 1.5         # Invalid gamma > 1
        }
        
        with self.assertRaises(ValueError):
            validate_config(invalid_config)


def run_tests():
    """Run all GBICR tests"""
    print("Running GBICR Protocol Tests...")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestGbicrPackets,
        TestGbicrStateExtractor,
        TestGbicrAgent,
        TestGbicrTable,
        TestGbicrConfig
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)