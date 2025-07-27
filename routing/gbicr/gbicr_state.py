import numpy as np
from utils.util_function import euclidean_distance_3d
from phy.large_scale_fading import maximum_communication_range


class GbicrStateExtractor:
    """State feature extraction for GBICR PPO agent"""
    
    def __init__(self, max_neighbors=10):
        self.max_neighbors = max_neighbors
        self.max_comm_range = maximum_communication_range()
    
    def extract_state(self, my_drone, dst_drone, neighbor_table, q_table=None):
        """
        Extract state features for PPO decision making
        
        Args:
            my_drone: current drone
            dst_drone: destination drone
            neighbor_table: current neighbor information
            q_table: Q-values table (optional)
            
        Returns:
            state: normalized state vector
        """
        state_features = []
        
        # 1. Current node information
        current_features = self._extract_current_node_features(my_drone, dst_drone)
        state_features.extend(current_features)
        
        # 2. Destination information
        dest_features = self._extract_destination_features(my_drone, dst_drone)
        state_features.extend(dest_features)
        
        # 3. Neighbor information
        neighbor_features = self._extract_neighbor_features(my_drone, dst_drone, neighbor_table)
        state_features.extend(neighbor_features)
        
        # 4. Network topology features
        topology_features = self._extract_topology_features(neighbor_table)
        state_features.extend(topology_features)
        
        return np.array(state_features, dtype=np.float32)
    
    def _extract_current_node_features(self, my_drone, dst_drone):
        """Extract current node features"""
        features = []
        
        # Position (normalized)
        pos_x = my_drone.coords[0] / 1000.0  # normalize to km
        pos_y = my_drone.coords[1] / 1000.0
        pos_z = my_drone.coords[2] / 1000.0
        features.extend([pos_x, pos_y, pos_z])
        
        # Velocity (normalized)
        vel_x = my_drone.velocity[0] / 50.0  # normalize assuming max 50 m/s
        vel_y = my_drone.velocity[1] / 50.0
        vel_z = my_drone.velocity[2] / 50.0
        features.extend([vel_x, vel_y, vel_z])
        
        # Speed magnitude
        speed = np.linalg.norm(my_drone.velocity) / 50.0
        features.append(speed)
        
        return features
    
    def _extract_destination_features(self, my_drone, dst_drone):
        """Extract destination-related features"""
        features = []
        
        # Distance to destination (normalized)
        distance = euclidean_distance_3d(my_drone.coords, dst_drone.coords)
        normalized_distance = distance / self.max_comm_range
        features.append(normalized_distance)
        
        # Relative position to destination (normalized)
        rel_x = (dst_drone.coords[0] - my_drone.coords[0]) / self.max_comm_range
        rel_y = (dst_drone.coords[1] - my_drone.coords[1]) / self.max_comm_range
        rel_z = (dst_drone.coords[2] - my_drone.coords[2]) / self.max_comm_range
        features.extend([rel_x, rel_y, rel_z])
        
        # Direction to destination (unit vector)
        if distance > 0:
            dir_x = (dst_drone.coords[0] - my_drone.coords[0]) / distance
            dir_y = (dst_drone.coords[1] - my_drone.coords[1]) / distance
            dir_z = (dst_drone.coords[2] - my_drone.coords[2]) / distance
        else:
            dir_x = dir_y = dir_z = 0.0
        features.extend([dir_x, dir_y, dir_z])
        
        return features
    
    def _extract_neighbor_features(self, my_drone, dst_drone, neighbor_table):
        """Extract neighbor-related features"""
        features = []
        
        # Number of neighbors (normalized)
        num_neighbors = len(neighbor_table)
        normalized_num_neighbors = min(num_neighbors / self.max_neighbors, 1.0)
        features.append(normalized_num_neighbors)
        
        # Neighbor features (padded to max_neighbors)
        neighbor_features = []
        neighbors = list(neighbor_table.keys())[:self.max_neighbors]
        
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                neighbor_id = neighbors[i]
                neighbor_info = neighbor_table[neighbor_id]
                neighbor_pos = neighbor_info[0]
                neighbor_vel = neighbor_info[1]
                
                # Distance to neighbor (normalized)
                neighbor_distance = euclidean_distance_3d(my_drone.coords, neighbor_pos)
                norm_neighbor_dist = neighbor_distance / self.max_comm_range
                
                # Relative position to neighbor (normalized)
                rel_neighbor_x = (neighbor_pos[0] - my_drone.coords[0]) / self.max_comm_range
                rel_neighbor_y = (neighbor_pos[1] - my_drone.coords[1]) / self.max_comm_range
                rel_neighbor_z = (neighbor_pos[2] - my_drone.coords[2]) / self.max_comm_range
                
                # Neighbor's distance to destination
                neighbor_to_dest_dist = euclidean_distance_3d(neighbor_pos, dst_drone.coords)
                norm_neighbor_to_dest = neighbor_to_dest_dist / self.max_comm_range
                
                # Geographic progress if choosing this neighbor
                my_to_dest_dist = euclidean_distance_3d(my_drone.coords, dst_drone.coords)
                geographic_progress = (my_to_dest_dist - neighbor_to_dest_dist) / self.max_comm_range
                
                # Relative velocity
                rel_vel_x = (neighbor_vel[0] - my_drone.velocity[0]) / 50.0
                rel_vel_y = (neighbor_vel[1] - my_drone.velocity[1]) / 50.0
                rel_vel_z = (neighbor_vel[2] - my_drone.velocity[2]) / 50.0
                
                # Link stability (based on relative velocity)
                rel_vel_magnitude = np.linalg.norm([rel_vel_x * 50.0, rel_vel_y * 50.0, rel_vel_z * 50.0])
                link_stability = max(0.0, 1.0 - rel_vel_magnitude / 100.0)  # normalize to [0,1]
                
                neighbor_features.extend([
                    norm_neighbor_dist,
                    rel_neighbor_x, rel_neighbor_y, rel_neighbor_z,
                    norm_neighbor_to_dest,
                    geographic_progress,
                    rel_vel_x, rel_vel_y, rel_vel_z,
                    link_stability
                ])
            else:
                # Padding for missing neighbors
                neighbor_features.extend([0.0] * 10)
        
        features.extend(neighbor_features)
        return features
    
    def _extract_topology_features(self, neighbor_table):
        """Extract network topology features"""
        features = []
        
        # Network density (local)
        num_neighbors = len(neighbor_table)
        # Assuming a circular communication area
        area = np.pi * (self.max_comm_range ** 2)
        density = num_neighbors / area * 1e6  # normalize to per km^2
        normalized_density = min(density / 100.0, 1.0)  # assuming max 100 drones per km^2
        features.append(normalized_density)
        
        # Average neighbor distance
        if num_neighbors > 0:
            total_distance = 0
            for neighbor_info in neighbor_table.values():
                neighbor_pos = neighbor_info[0]
                distance = euclidean_distance_3d([0, 0, 0], neighbor_pos)  # relative to origin
                total_distance += distance
            avg_distance = total_distance / num_neighbors
            normalized_avg_distance = avg_distance / self.max_comm_range
        else:
            normalized_avg_distance = 0.0
        features.append(normalized_avg_distance)
        
        # Connectivity metric (based on neighbor distribution)
        connectivity = min(num_neighbors / 8.0, 1.0)  # assuming 8 neighbors for good connectivity
        features.append(connectivity)
        
        return features
    
    def get_state_dimension(self):
        """Get the dimension of the state vector"""
        # Current node: 7 features (pos:3, vel:3, speed:1)
        # Destination: 7 features (distance:1, rel_pos:3, direction:3)
        # Neighbors: 1 + max_neighbors * 10 features
        # Topology: 3 features
        return 7 + 7 + 1 + self.max_neighbors * 10 + 3