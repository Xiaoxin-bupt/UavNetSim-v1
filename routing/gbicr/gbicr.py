import copy
import math
import random
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from simulator.log import logger
from entities.packet import DataPacket
from topology.virtual_force.vf_packet import VfPacket
from routing.gbicr.gbicr_packet import (
    GbicrHelloPacket, GbicrBeaconPacket, GbicrAckPacket, GbicrDataPacket
)
from routing.gbicr.gbicr_table import GbicrTable
from routing.gbicr.gbicr_state import GbicrStateExtractor
from routing.gbicr.gbicr_agent import GbicrIntelligentAgent
from utils import config
from utils import util_function
from utils.util_function import euclidean_distance_3d
from phy.large_scale_fading import maximum_communication_range


class GBICR:
    """
    Geographic Beacon-based Intelligent Collaborative Routing (GBICR) Protocol
    
    A novel routing protocol that combines:
    - Geographic routing with beacon-based global awareness
    - PPO-based intelligent decision making
    - Collaborative Q-value sharing among neighbors
    - Multi-dimensional link quality assessment
    
    Attributes:
        simulator: simulation platform
        my_drone: drone that installed GBICR
        rng_routing: random number generator for routing decisions
        hello_interval: interval for hello packet broadcasting
        beacon_interval: interval for beacon packet broadcasting
        check_interval: interval for checking waiting list
        table: neighbor table and collaborative Q-value management
        state_extractor: state feature extraction for PPO
        intelligent_agent: PPO-based decision making agent
        
    References:
        Based on QGeo and enhanced with PPO and collaborative mechanisms
        
    Author: liuwenxin
    Created: 2025/07/27
    """
    
    def __init__(self, simulator, my_drone, pretrained_model_path=None):
        self.simulator = simulator
        self.my_drone = my_drone
        self.rng_routing = random.Random(self.my_drone.identifier + self.my_drone.simulator.seed + 20)
        
        # Protocol timing parameters
        self.hello_interval = 0.5 * 1e6  # 500ms for hello packets
        self.beacon_interval = 2.0 * 1e6  # 2s for beacon packets
        self.check_interval = 0.6 * 1e6  # 600ms for waiting list check
        
        # Reward parameters for learning
        self.r_max = 10.0
        self.r_min = -10.0
        self.learning_rate = 0.3
        
        # Initialize routing table
        self.table = GbicrTable(self.simulator.env, my_drone, self.rng_routing)
        
        # Initialize intelligent components
        self.state_extractor = GbicrStateExtractor(max_neighbors=10)
        self.intelligent_agent = GbicrIntelligentAgent(
            self.state_extractor, 
            max_neighbors=10,
            pretrained_model_path=pretrained_model_path
        )
        
        # Performance tracking
        self.packet_success_count = 0
        self.packet_failure_count = 0
        self.total_delay = 0.0
        self.hop_count_sum = 0
        
        # Start periodic processes
        self.simulator.env.process(self.broadcast_hello_packet_periodically())
        self.simulator.env.process(self.broadcast_beacon_packet_periodically())
        self.simulator.env.process(self.check_waiting_list())
    
    def broadcast_hello_packet(self):
        """Broadcast hello packet for neighbor discovery"""
        config.GL_ID_HELLO_PACKET += 1
        
        # Channel assignment
        channel_id = self.my_drone.channel_assigner.channel_assign()
        
        hello_packet = GbicrHelloPacket(
            src_drone=self.my_drone,
            creation_time=self.simulator.env.now,
            id_hello_packet=config.GL_ID_HELLO_PACKET,
            hello_packet_length=config.HELLO_PACKET_LENGTH,
            simulator=self.simulator,
            channel_id=channel_id
        )
        
        # Add neighbor count and link stability info
        hello_packet.neighbor_count = self.table.get_neighbor_count()
        hello_packet.link_stability = self.table.get_average_link_quality()
        hello_packet.transmission_mode = 1  # broadcast mode
        
        logger.info('At time: %s (us) ---- UAV: %s broadcasts GBICR hello packet',
                    self.simulator.env.now, self.my_drone.identifier)
        
        self.simulator.metrics.control_packet_num += 1
        self.my_drone.transmitting_queue.put(hello_packet)
    
    def broadcast_beacon_packet(self):
        """Broadcast beacon packet for geographic information sharing"""
        config.GL_ID_HELLO_PACKET += 1  # Reuse hello packet ID counter
        
        # Channel assignment
        channel_id = self.my_drone.channel_assigner.channel_assign()
        
        # Calculate coverage area based on neighbors
        coverage_area = self._calculate_coverage_area()
        
        beacon_packet = GbicrBeaconPacket(
            src_drone=self.my_drone,
            creation_time=self.simulator.env.now,
            id_beacon_packet=config.GL_ID_HELLO_PACKET,
            beacon_packet_length=config.HELLO_PACKET_LENGTH + 64,  # slightly larger
            simulator=self.simulator,
            channel_id=channel_id,
            coverage_area=coverage_area
        )
        
        beacon_packet.network_density = self.table.network_density
        beacon_packet.connectivity_metric = self.table.connectivity_metric
        beacon_packet.transmission_mode = 1  # broadcast mode
        
        logger.info('At time: %s (us) ---- UAV: %s broadcasts GBICR beacon packet',
                    self.simulator.env.now, self.my_drone.identifier)
        
        self.simulator.metrics.control_packet_num += 1
        self.my_drone.transmitting_queue.put(beacon_packet)
    
    def broadcast_hello_packet_periodically(self):
        """Periodically broadcast hello packets"""
        while True:
            self.broadcast_hello_packet()
            jitter = self.rng_routing.randint(1000, 2000)  # add jitter
            yield self.simulator.env.timeout(self.hello_interval + jitter)
    
    def broadcast_beacon_packet_periodically(self):
        """Periodically broadcast beacon packets"""
        while True:
            # Wait for initial hello packets to establish neighbors
            yield self.simulator.env.timeout(self.beacon_interval)
            self.broadcast_beacon_packet()
    
    def next_hop_selection(self, packet):
        """
        Select next hop using GBICR intelligent routing
        
        Args:
            packet: data packet to be routed
            
        Returns:
            tuple: (has_route, packet, enquire)
        """
        enquire = False
        has_route = True
        
        # Update neighbor table
        self.table.purge()
        
        dst_drone = packet.dst_drone
        
        # Add current drone to intermediate path
        packet.intermediate_drones.append(self.my_drone.identifier)
        
        # Use intelligent agent for next hop selection
        best_next_hop_id = self.intelligent_agent.select_next_hop(
            self.my_drone, dst_drone, self.table.neighbor_table
        )
        
        if best_next_hop_id == self.my_drone.identifier:
            has_route = False  # no available next hop
        else:
            packet.next_hop_id = best_next_hop_id
            
            # Update packet metadata
            if hasattr(packet, 'hop_count'):
                packet.hop_count += 1
            if hasattr(packet, 'route_history'):
                packet.route_history.append(self.my_drone.identifier)
        
        return has_route, packet, enquire
    
    def packet_reception(self, packet, src_drone_id):
        """
        Handle packet reception at network layer
        
        Args:
            packet: received packet
            src_drone_id: ID of the sending drone
        """
        current_time = self.simulator.env.now
        
        if isinstance(packet, GbicrHelloPacket):
            self._handle_hello_packet(packet, current_time)
            
        elif isinstance(packet, GbicrBeaconPacket):
            self._handle_beacon_packet(packet, current_time)
            
        elif isinstance(packet, DataPacket):
            yield from self._handle_data_packet(packet, src_drone_id)
            
        elif isinstance(packet, GbicrAckPacket):
            self._handle_ack_packet(packet, src_drone_id)
            
        elif isinstance(packet, VfPacket):
            yield from self._handle_vf_packet(packet, src_drone_id, current_time)
    
    def _handle_hello_packet(self, packet, current_time):
        """Handle hello packet reception"""
        self.table.add_neighbor(packet, current_time)
        logger.debug('At time: %s (us) ---- UAV: %s received hello from UAV: %s',
                    current_time, self.my_drone.identifier, packet.src_drone.identifier)
    
    def _handle_beacon_packet(self, packet, current_time):
        """Handle beacon packet reception"""
        self.table.add_beacon_info(packet, current_time)
        logger.debug('At time: %s (us) ---- UAV: %s received beacon from UAV: %s',
                    current_time, self.my_drone.identifier, packet.src_drone.identifier)
    
    def _handle_data_packet(self, packet, src_drone_id):
        """Handle data packet reception"""
        packet_copy = copy.copy(packet)
        packet_copy.previous_drone = self.simulator.drones[src_drone_id]
        
        if packet_copy.dst_drone.identifier == self.my_drone.identifier:
            # Packet reached destination
            yield from self._handle_destination_packet(packet_copy, src_drone_id)
        else:
            # Packet needs forwarding
            yield from self._handle_forwarding_packet(packet_copy, src_drone_id)
    
    def _handle_destination_packet(self, packet, src_drone_id):
        """Handle packet that reached its destination"""
        if packet.packet_id not in self.simulator.metrics.datapacket_arrived:
            self.simulator.metrics.calculate_metrics(packet)
            
            logger.info('At time: %s (us) ---- Data packet: %s reached destination UAV: %s',
                       self.simulator.env.now, packet.packet_id, self.my_drone.identifier)
            
            # Update success statistics
            self.packet_success_count += 1
            
            # Provide positive reward to intelligent agent
            self.intelligent_agent.update_reward(self.r_max, done=True)
        
        # Send acknowledgment
        yield from self._send_ack_packet(packet, src_drone_id, reward=self.r_max, success=True)
    
    def _handle_forwarding_packet(self, packet, src_drone_id):
        """Handle packet that needs forwarding"""
        if self.my_drone.transmitting_queue.qsize() < self.my_drone.max_queue_size:
            logger.info('At time: %s (us) ---- Data packet: %s received by intermediate UAV: %s',
                       self.simulator.env.now, packet.packet_id, self.my_drone.identifier)
            
            self.my_drone.transmitting_queue.put(packet)
            
            # Calculate reward based on geographic progress
            reward = self._calculate_forwarding_reward(packet, src_drone_id)
            
            # Update intelligent agent
            self.intelligent_agent.update_reward(reward)
            
            # Send acknowledgment
            yield from self._send_ack_packet(packet, src_drone_id, reward=reward, success=True)
        else:
            # Queue full - drop packet
            logger.warning('At time: %s (us) ---- UAV: %s queue full, dropping packet: %s',
                          self.simulator.env.now, self.my_drone.identifier, packet.packet_id)
            
            # Negative reward for queue overflow
            self.intelligent_agent.update_reward(self.r_min)
    
    def _send_ack_packet(self, original_packet, dst_drone_id, reward, success):
        """Send acknowledgment packet"""
        config.GL_ID_ACK_PACKET += 1
        dst_drone = self.simulator.drones[dst_drone_id]
        
        # Get collaborative Q-values to share
        neighbor_q_values = self.table.get_weighted_q_values(original_packet.dst_drone.identifier)
        
        # Calculate link quality
        distance = euclidean_distance_3d(self.my_drone.coords, dst_drone.coords)
        link_quality = max(0.0, 1.0 - distance / maximum_communication_range())
        
        ack_packet = GbicrAckPacket(
            src_drone=self.my_drone,
            dst_drone=dst_drone,
            ack_packet_id=config.GL_ID_ACK_PACKET,
            ack_packet_length=config.ACK_PACKET_LENGTH,
            acked_packet=original_packet,
            reward=reward,
            neighbor_q_values=neighbor_q_values,
            link_quality=link_quality,
            simulator=self.simulator,
            channel_id=original_packet.channel_id
        )
        
        ack_packet.transmission_success = success
        
        yield self.simulator.env.timeout(config.SIFS_DURATION)
        
        if not self.my_drone.sleep:
            ack_packet.increase_ttl()
            self.my_drone.mac_protocol.phy.unicast(ack_packet, dst_drone_id)
            yield self.simulator.env.timeout(ack_packet.packet_length / config.BIT_RATE * 1e6)
            self.simulator.drones[dst_drone_id].receive()
    
    def _handle_ack_packet(self, packet, src_drone_id):
        """Handle acknowledgment packet reception"""
        # Update collaborative Q-values
        for neighbor_id, q_value in packet.neighbor_q_values.items():
            self.table.update_collaborative_q_values(
                src_drone_id, packet.acked_packet.dst_drone.identifier, q_value
            )
        
        # Remove acknowledged packet from queue
        self.my_drone.remove_from_queue(packet.acked_packet)
        
        # Handle MAC layer acknowledgment process
        key = 'wait_ack' + str(self.my_drone.identifier) + '_' + str(packet.acked_packet.packet_id)
        
        if self.my_drone.mac_protocol.wait_ack_process_finish[key] == 0:
            if not self.my_drone.mac_protocol.wait_ack_process_dict[key].triggered:
                logger.info('At time: %s (us) ---- wait_ack process (id: %s) of UAV: %s interrupted by UAV: %s',
                           self.simulator.env.now, key, self.my_drone.identifier, src_drone_id)
                
                self.my_drone.mac_protocol.wait_ack_process_finish[key] = 1
                self.my_drone.mac_protocol.wait_ack_process_dict[key].interrupt()
    
    def _handle_vf_packet(self, packet, src_drone_id, current_time):
        """Handle virtual force packet (for topology control)"""
        logger.info('At time: %s (us) ---- UAV: %s receives VF packet from UAV: %s',
                   self.simulator.env.now, self.my_drone.identifier, src_drone_id)
        
        # Update motion controller's neighbor table
        self.my_drone.motion_controller.neighbor_table.add_neighbor(packet, current_time)
        
        if packet.msg_type == 'hello':
            config.GL_ID_VF_PACKET += 1
            
            ack_packet = VfPacket(
                src_drone=self.my_drone,
                creation_time=self.simulator.env.now,
                id_hello_packet=config.GL_ID_VF_PACKET,
                hello_packet_length=config.HELLO_PACKET_LENGTH,
                simulator=self.simulator,
                channel_id=packet.channel_id
            )
            ack_packet.msg_type = 'ack'
            
            self.my_drone.transmitting_queue.put(ack_packet)
    
    def _calculate_forwarding_reward(self, packet, src_drone_id):
        """Calculate reward for packet forwarding"""
        # Geographic progress reward
        src_distance = euclidean_distance_3d(
            self.simulator.drones[src_drone_id].coords, packet.dst_drone.coords
        )
        my_distance = euclidean_distance_3d(
            self.my_drone.coords, packet.dst_drone.coords
        )
        
        geographic_progress = (src_distance - my_distance) / maximum_communication_range()
        
        # Link quality bonus
        link_quality_bonus = self.table.get_average_link_quality() * 0.5
        
        # Connectivity bonus
        connectivity_bonus = self.table.get_network_connectivity() * 0.3
        
        # Combined reward
        reward = geographic_progress + link_quality_bonus + connectivity_bonus
        
        # Ensure reward is within bounds
        return max(self.r_min, min(self.r_max, reward))
    
    def _calculate_coverage_area(self):
        """Calculate coverage area for beacon packets"""
        coverage_area = []
        
        # Simple coverage area based on neighbor positions
        for neighbor_id, neighbor_info in self.table.neighbor_table.items():
            neighbor_pos = neighbor_info[0]
            coverage_area.append(neighbor_pos)
        
        return coverage_area
    
    def check_waiting_list(self):
        """Periodically check and process waiting list"""
        while True:
            if not self.my_drone.sleep:
                yield self.simulator.env.timeout(self.check_interval)
                
                for waiting_packet in list(self.my_drone.waiting_list):
                    if self.simulator.env.now > waiting_packet.creation_time + waiting_packet.deadline:
                        # Packet expired
                        self.my_drone.waiting_list.remove(waiting_packet)
                        self.packet_failure_count += 1
                        
                        # Negative reward for expired packet
                        self.intelligent_agent.update_reward(self.r_min, done=True)
                    else:
                        # Try to find route
                        has_route, packet, enquire = self.next_hop_selection(waiting_packet)
                        if has_route:
                            self.my_drone.transmitting_queue.put(waiting_packet)
                            self.my_drone.waiting_list.remove(waiting_packet)
            else:
                break
    
    def penalize(self, packet):
        """Penalize for failed packet transmission"""
        self.packet_failure_count += 1
        
        # Provide negative reward to intelligent agent
        self.intelligent_agent.update_reward(self.r_min, done=True)
        
        logger.warning('At time: %s (us) ---- UAV: %s penalized for failed transmission of packet: %s',
                      self.simulator.env.now, self.my_drone.identifier, packet.packet_id)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        total_packets = self.packet_success_count + self.packet_failure_count
        success_rate = self.packet_success_count / total_packets if total_packets > 0 else 0.0
        
        stats = {
            'success_rate': success_rate,
            'total_packets': total_packets,
            'neighbor_count': self.table.get_neighbor_count(),
            'avg_link_quality': self.table.get_average_link_quality(),
            'connectivity': self.table.get_network_connectivity()
        }
        
        # Add intelligent agent statistics
        agent_stats = self.intelligent_agent.get_training_stats()
        stats.update(agent_stats)
        
        return stats
    
    def set_training_mode(self, training=True):
        """Set training mode for the intelligent agent"""
        self.intelligent_agent.set_training_mode(training)
    
    def save_model(self, model_path):
        """Save the trained model"""
        self.intelligent_agent.save_model(model_path)
    
    def load_model(self, model_path):
        """Load a pretrained model"""
        return self.intelligent_agent.load_pretrained_model(model_path)