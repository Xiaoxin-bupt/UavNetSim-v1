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
from gbicr_config import get_gbicr_config, validate_gbicr_config


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
        current_pos = [100, 200, 50]
        dest_drone = {
            'position': [300, 400, 100]
        }
        
        features = self.extractor.extract_destination_features(current_pos, dest_drone)
        
        # Should include destination position and distance
        self.assertEqual(len(features), 4)  # 3 + 1
        self.assertEqual(features[:3], [300, 400, 100])
        
        # Check distance calculation
        expected_distance = np.sqrt((300-100)**2 + (400-200)**2 + (100-50)**2)
        self.assertAlmostEqual(features[3], expected_distance, places=2)
    
    def test_neighbor_features(self):
        """Test neighbor feature extraction"""
        current_drone = {
            'position': [100, 200, 50],
            'velocity': [10, -5, 2]
        }
        
        neighbors = {
            1: {
                'position': [150, 250, 60],
                'velocity': [5, 0, 1],
                'link_quality': 0.8,
                'stability': 0.9,
                'last_update': 1.0
            }
        }
        
        features = self.extractor.extract_neighbor_features(current_drone, neighbors)
        
        # Should have features for max_neighbors (5)
        expected_length = 5 * 8  # 8 features per neighbor
        self.assertEqual(len(features), expected_length)
        
        # First neighbor should have actual values
        self.assertNotEqual(features[0], 0)  # relative_x
        self.assertNotEqual(features[1], 0)  # relative_y
    
    def test_get_state_vector(self):
        """Test complete state vector extraction"""
        current_drone = {
            'position': [100, 200, 50],
            'velocity': [10, -5, 2],
            'neighbors': [1]
        }
        
        dest_drone = {
            'position': [300, 400, 100]
        }
        
        neighbors = {
            1: {
                'position': [150, 250, 60],
                'velocity': [5, 0, 1],
                'link_quality': 0.8,
                'stability': 0.9,
                'last_update': 1.0
            }
        }
        
        network_info = {
            'density': 0.7,
            'connectivity': 0.85
        }
        
        state_vector = self.extractor.get_state_vector(
            current_drone, dest_drone, neighbors, network_info
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
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.value_net)
    
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
        self.assertTrue(self.agent.training_mode)
        
        # Test inference mode
        self.agent.set_training_mode(False)
        self.assertFalse(self.agent.training_mode)


class TestGbicrTable(unittest.TestCase):
    """Test GBICR routing table"""
    
    def setUp(self):
        self.table = GbicrTable()
    
    def test_add_neighbor(self):
        """Test adding neighbor to table"""
        neighbor_id = 1
        position = [100, 200, 50]
        velocity = [10, -5, 2]
        
        self.table.add_neighbor(neighbor_id, position, velocity)
        
        self.assertIn(neighbor_id, self.table.neighbor_table)
        neighbor_info = self.table.neighbor_table[neighbor_id]
        self.assertEqual(neighbor_info['position'], position)
        self.assertEqual(neighbor_info['velocity'], velocity)
    
    def test_update_beacon_info(self):
        """Test updating beacon information"""
        beacon_id = 2
        position = [300, 400, 100]
        coverage_area = 500.0
        
        self.table.update_beacon_info(beacon_id, position, coverage_area)
        
        self.assertIn(beacon_id, self.table.beacon_table)
        beacon_info = self.table.beacon_table[beacon_id]
        self.assertEqual(beacon_info['position'], position)
        self.assertEqual(beacon_info['coverage_area'], coverage_area)
    
    def test_calculate_link_quality(self):
        """Test link quality calculation"""
        current_pos = [0, 0, 0]
        neighbor_pos = [100, 0, 0]  # 100m away
        
        quality = self.table.calculate_link_quality(current_pos, neighbor_pos)
        
        self.assertGreater(quality, 0)
        self.assertLessEqual(quality, 1)
    
    def test_best_next_hop_selection(self):
        """Test best next hop selection"""
        # Add some neighbors
        self.table.add_neighbor(1, [100, 100, 50], [5, 5, 1])
        self.table.add_neighbor(2, [200, 200, 60], [10, 10, 2])
        
        current_pos = [0, 0, 0]
        dest_pos = [300, 300, 100]
        
        best_hop = self.table.select_best_next_hop(current_pos, dest_pos)
        
        # Should return one of the neighbors or None
        self.assertIn(best_hop, [1, 2, None])


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
            'ppo_lr': 0.001,
            'max_neighbors': 10
        }
        
        # Should not raise exception
        validate_gbicr_config(valid_config)
        
        # Test invalid config
        invalid_config = {
            'hello_interval': -1.0,  # Invalid negative value
            'ppo_lr': 2.0  # Invalid learning rate > 1
        }
        
        with self.assertRaises(ValueError):
            validate_gbicr_config(invalid_config)


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