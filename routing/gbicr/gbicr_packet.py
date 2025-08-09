import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from entities.packet import Packet


class GbicrHelloPacket(Packet):
    """GBICR Hello packet for neighbor discovery"""
    
    def __init__(self,
                 src_drone,
                 creation_time,
                 id_hello_packet,
                 hello_packet_length,
                 simulator,
                 channel_id):
        super().__init__(id_hello_packet, hello_packet_length, creation_time, simulator, channel_id)
        
        self.src_drone = src_drone
        self.cur_position = src_drone.coords
        self.cur_velocity = src_drone.velocity
        self.neighbor_count = 0  # number of neighbors
        self.link_stability = 1.0  # link stability metric


class GbicrBeaconPacket(Packet):
    """GBICR Beacon packet for geographic information sharing"""
    
    def __init__(self,
                 src_drone,
                 creation_time,
                 id_beacon_packet,
                 beacon_packet_length,
                 simulator,
                 channel_id,
                 coverage_area=None):
        super().__init__(id_beacon_packet, beacon_packet_length, creation_time, simulator, channel_id)
        
        self.src_drone = src_drone
        self.cur_position = src_drone.coords
        self.cur_velocity = src_drone.velocity
        self.coverage_area = coverage_area or []
        self.network_density = 0.0
        self.connectivity_metric = 0.0


class GbicrAckPacket(Packet):
    """GBICR Acknowledgment packet with reward and Q-value information"""
    
    def __init__(self,
                 src_drone,
                 dst_drone,
                 ack_packet_id,
                 ack_packet_length,
                 acked_packet,
                 reward,
                 neighbor_q_values,
                 link_quality,
                 simulator,
                 channel_id,
                 creation_time=None):
        super().__init__(ack_packet_id, ack_packet_length, creation_time, simulator, channel_id)
        
        self.src_drone = src_drone
        self.src_coords = src_drone.coords
        self.src_velocity = src_drone.velocity
        self.dst_drone = dst_drone
        self.acked_packet = acked_packet
        self.reward = reward
        self.neighbor_q_values = neighbor_q_values  # Q-values from neighbors
        self.link_quality = link_quality
        self.transmission_success = True


class GbicrDataPacket(Packet):
    """GBICR Data packet with routing metadata"""
    
    def __init__(self,
                 src_drone,
                 dst_drone,
                 packet_id,
                 packet_length,
                 creation_time,
                 simulator,
                 channel_id,
                 payload=None):
        super().__init__(packet_id, packet_length, creation_time, simulator, channel_id)
        
        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.payload = payload
        self.hop_count = 0
        self.route_history = []  # record of intermediate nodes
        self.geographic_progress = 0.0  # progress towards destination
        self.expected_delay = 0.0