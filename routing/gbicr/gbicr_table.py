import numpy as np
import math
from collections import defaultdict
from utils.util_function import euclidean_distance_3d
from phy.large_scale_fading import maximum_communication_range


class GbicrTable:
    """GBICR routing table management including neighbor table and collaborative Q-values"""
    
    def __init__(self, env, my_drone, rng_routing, max_neighbors=20):
        self.env = env
        self.my_drone = my_drone
        self.rng_routing = rng_routing
        self.max_neighbors = max_neighbors
        
        # Neighbor table: {neighbor_id: [position, velocity, timestamp, link_quality, stability]}
        self.neighbor_table = defaultdict(list)
        
        # Collaborative Q-values: {neighbor_id: {destination_id: q_value}}
        self.collaborative_q_values = defaultdict(lambda: defaultdict(float))
        
        # Geographic beacon information: {beacon_id: [position, coverage_area, timestamp]}
        self.beacon_table = defaultdict(list)
        
        # Link quality metrics: {neighbor_id: quality_score}
        self.link_quality = defaultdict(float)
        
        # Network topology information
        self.network_density = 0.0
        self.connectivity_metric = 0.0
        
        # Configuration parameters
        self.entry_life_time = 3 * 1e6  # 3 seconds in microseconds
        self.beacon_life_time = 5 * 1e6  # 5 seconds for beacon information
        self.max_comm_range = maximum_communication_range()
        
        # Link stability tracking
        self.link_history = defaultdict(list)  # {neighbor_id: [quality_samples]}
        self.stability_window = 10  # number of samples for stability calculation
    
    def add_neighbor(self, hello_packet, current_time):
        """
        Add or update neighbor information from hello packet
        
        Args:
            hello_packet: received hello packet
            current_time: current simulation time
        """
        if hello_packet.src_drone.identifier == self.my_drone.identifier:
            return
        
        neighbor_id = hello_packet.src_drone.identifier
        position = hello_packet.cur_position
        velocity = hello_packet.cur_velocity
        
        # Calculate link quality
        distance = euclidean_distance_3d(self.my_drone.coords, position)
        link_quality = self._calculate_link_quality(distance, velocity)
        
        # Calculate link stability
        stability = self._calculate_link_stability(neighbor_id, link_quality)
        
        # Update neighbor table
        self.neighbor_table[neighbor_id] = [
            position, velocity, current_time, link_quality, stability
        ]
        
        # Update link quality tracking
        self.link_quality[neighbor_id] = link_quality
        
        # Update link history for stability calculation
        if neighbor_id not in self.link_history:
            self.link_history[neighbor_id] = []
        
        self.link_history[neighbor_id].append(link_quality)
        if len(self.link_history[neighbor_id]) > self.stability_window:
            self.link_history[neighbor_id].pop(0)
    
    def add_beacon_info(self, beacon_packet, current_time):
        """
        Add geographic beacon information
        
        Args:
            beacon_packet: received beacon packet
            current_time: current simulation time
        """
        beacon_id = beacon_packet.src_drone.identifier
        position = beacon_packet.cur_position
        coverage_area = beacon_packet.coverage_area
        
        self.beacon_table[beacon_id] = [position, coverage_area, current_time]
    
    def update_collaborative_q_values(self, neighbor_id, destination_id, q_value):
        """
        Update collaborative Q-values from neighbors
        
        Args:
            neighbor_id: ID of the neighbor providing Q-value
            destination_id: destination for which Q-value is provided
            q_value: Q-value from the neighbor
        """
        self.collaborative_q_values[neighbor_id][destination_id] = q_value
    
    def get_weighted_q_values(self, destination_id):
        """
        Get weighted Q-values from neighbors for collaborative decision making
        
        Args:
            destination_id: target destination
            
        Returns:
            dict: {neighbor_id: weighted_q_value}
        """
        weighted_q_values = {}
        
        for neighbor_id in self.neighbor_table.keys():
            if neighbor_id in self.collaborative_q_values:
                q_value = self.collaborative_q_values[neighbor_id].get(destination_id, 0.0)
                
                # Weight by link quality and stability
                link_quality = self.link_quality.get(neighbor_id, 0.0)
                stability = self.neighbor_table[neighbor_id][4] if neighbor_id in self.neighbor_table else 0.0
                
                weight = 0.6 * link_quality + 0.4 * stability
                weighted_q_values[neighbor_id] = q_value * weight
        
        return weighted_q_values
    
    def best_neighbor_collaborative(self, dst_drone, exploration_rate=0.1):
        """
        Select best neighbor using collaborative Q-values and geographic information
        
        Args:
            dst_drone: destination drone
            exploration_rate: exploration probability
            
        Returns:
            int: best neighbor ID
        """
        self.purge()
        
        if self.is_empty():
            return self.my_drone.identifier
        
        dst_id = dst_drone.identifier
        
        # Exploration vs exploitation
        if self.rng_routing.random() < exploration_rate:
            return self.rng_routing.choice(list(self.neighbor_table.keys()))
        
        # Get collaborative Q-values
        weighted_q_values = self.get_weighted_q_values(dst_id)
        
        # Combine with geographic progress
        neighbor_scores = {}
        my_distance = euclidean_distance_3d(self.my_drone.coords, dst_drone.coords)
        
        for neighbor_id in self.neighbor_table.keys():
            neighbor_info = self.neighbor_table[neighbor_id]
            neighbor_pos = neighbor_info[0]
            link_quality = neighbor_info[3]
            stability = neighbor_info[4]
            
            # Geographic progress
            neighbor_distance = euclidean_distance_3d(neighbor_pos, dst_drone.coords)
            geographic_progress = (my_distance - neighbor_distance) / self.max_comm_range
            
            # Collaborative Q-value
            collaborative_q = weighted_q_values.get(neighbor_id, 0.0)
            
            # Combined score
            score = (
                0.4 * geographic_progress +
                0.3 * collaborative_q +
                0.2 * link_quality +
                0.1 * stability
            )
            
            neighbor_scores[neighbor_id] = score
        
        # Select best neighbor
        if neighbor_scores:
            best_neighbor = max(neighbor_scores.keys(), key=lambda x: neighbor_scores[x])
            return best_neighbor
        
        return self.my_drone.identifier
    
    def _calculate_link_quality(self, distance, neighbor_velocity):
        """
        Calculate link quality based on distance and mobility
        
        Args:
            distance: distance to neighbor
            neighbor_velocity: neighbor's velocity vector
            
        Returns:
            float: link quality score [0, 1]
        """
        # Distance-based quality
        distance_quality = max(0.0, 1.0 - distance / self.max_comm_range)
        
        # Mobility-based quality
        relative_velocity = np.array(neighbor_velocity) - np.array(self.my_drone.velocity)
        relative_speed = np.linalg.norm(relative_velocity)
        mobility_quality = max(0.0, 1.0 - relative_speed / 100.0)  # assuming max 100 m/s
        
        # Combined quality
        quality = 0.7 * distance_quality + 0.3 * mobility_quality
        return min(1.0, max(0.0, quality))
    
    def _calculate_link_stability(self, neighbor_id, current_quality):
        """
        Calculate link stability based on quality history
        
        Args:
            neighbor_id: neighbor identifier
            current_quality: current link quality
            
        Returns:
            float: stability score [0, 1]
        """
        if neighbor_id not in self.link_history or len(self.link_history[neighbor_id]) < 2:
            return current_quality  # Use current quality as initial stability
        
        quality_history = self.link_history[neighbor_id]
        
        # Calculate variance of quality
        mean_quality = np.mean(quality_history)
        variance = np.var(quality_history)
        
        # Stability is inversely related to variance
        stability = max(0.0, 1.0 - variance)
        
        # Weight with mean quality
        stability = 0.7 * stability + 0.3 * mean_quality
        
        return min(1.0, max(0.0, stability))
    
    def update_network_metrics(self):
        """
        Update network topology metrics
        """
        num_neighbors = len(self.neighbor_table)
        
        # Network density (local)
        area = math.pi * (self.max_comm_range ** 2)
        self.network_density = num_neighbors / area if area > 0 else 0.0
        
        # Connectivity metric
        self.connectivity_metric = min(num_neighbors / 8.0, 1.0)  # assuming 8 neighbors for good connectivity
    
    def is_empty(self):
        """Check if neighbor table is empty"""
        return len(self.neighbor_table) == 0
    
    def get_updated_time(self, neighbor_id):
        """Get last update time for a neighbor"""
        if neighbor_id not in self.neighbor_table:
            raise RuntimeError('Neighbor not in table')
        return self.neighbor_table[neighbor_id][2]
    
    def is_neighbor(self, neighbor_id):
        """Check if a drone is a valid neighbor"""
        if neighbor_id in self.neighbor_table:
            return self.get_updated_time(neighbor_id) + self.entry_life_time > self.env.now
        return False
    
    def remove_neighbor(self, neighbor_id):
        """Remove a neighbor from the table"""
        if neighbor_id in self.neighbor_table:
            del self.neighbor_table[neighbor_id]
        if neighbor_id in self.link_quality:
            del self.link_quality[neighbor_id]
        if neighbor_id in self.collaborative_q_values:
            del self.collaborative_q_values[neighbor_id]
        if neighbor_id in self.link_history:
            del self.link_history[neighbor_id]
    
    def purge(self):
        """Remove expired entries"""
        current_time = self.env.now
        
        # Purge expired neighbors
        expired_neighbors = []
        for neighbor_id in list(self.neighbor_table.keys()):
            if self.get_updated_time(neighbor_id) + self.entry_life_time <= current_time:
                expired_neighbors.append(neighbor_id)
        
        for neighbor_id in expired_neighbors:
            self.remove_neighbor(neighbor_id)
        
        # Purge expired beacon information
        expired_beacons = []
        for beacon_id in list(self.beacon_table.keys()):
            beacon_time = self.beacon_table[beacon_id][2]
            if beacon_time + self.beacon_life_time <= current_time:
                expired_beacons.append(beacon_id)
        
        for beacon_id in expired_beacons:
            del self.beacon_table[beacon_id]
        
        # Update network metrics
        self.update_network_metrics()
    
    def clear(self):
        """Clear all tables"""
        self.neighbor_table.clear()
        self.collaborative_q_values.clear()
        self.beacon_table.clear()
        self.link_quality.clear()
        self.link_history.clear()
    
    def get_neighbor_count(self):
        """Get current number of neighbors"""
        return len(self.neighbor_table)
    
    def get_average_link_quality(self):
        """Get average link quality of all neighbors"""
        if not self.link_quality:
            return 0.0
        return np.mean(list(self.link_quality.values()))
    
    def get_network_connectivity(self):
        """Get network connectivity metric"""
        return self.connectivity_metric
    
    def void_area_judgment(self, dst_drone):
        """
        Determine if current node is in void area (geographic routing)
        
        Args:
            dst_drone: destination drone
            
        Returns:
            int: 1 if in void area, 0 otherwise
        """
        my_distance = euclidean_distance_3d(self.my_drone.coords, dst_drone.coords)
        
        for neighbor_id in self.neighbor_table.keys():
            neighbor_pos = self.neighbor_table[neighbor_id][0]
            neighbor_distance = euclidean_distance_3d(neighbor_pos, dst_drone.coords)
            
            if neighbor_distance < my_distance:
                return 0  # Not in void area
        
        return 1  # In void area