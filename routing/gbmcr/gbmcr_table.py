import time
import math
from collections import defaultdict
from utils.util_function import euclidean_distance_3d as euclidean_distance


class GbmcrNeighborTable:
    """
    GBMCR协议的邻居表和信标管理模块
    维护邻居节点信息、管理地理信标信息、计算路径质量和选择最佳下一跳
    """
    
    def __init__(self, drone_id, communication_range=200, neighbor_timeout=10.0):
        self.drone_id = drone_id
        self.communication_range = communication_range
        self.neighbor_timeout = neighbor_timeout
        
        # 邻居表：{neighbor_id: neighbor_info}
        self.neighbors = {}
        
        # Q值表：{(current_node, destination): {neighbor: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 地理信标表：{beacon_id: beacon_info}
        self.beacons = {}
        
        # 全局位置表：{node_id: position_info}
        self.global_positions = {}
        
        # 协作评分表：{neighbor_id: cooperation_score}
        self.cooperation_scores = defaultdict(float)
        
        # 链路质量历史：{neighbor_id: [quality_history]}
        self.link_quality_history = defaultdict(list)
        
        # 自适应参数
        self.adaptive_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'hello_interval': 2.0
        }
    
    def add_or_update_neighbor(self, neighbor_id, coords, velocity, neighbor_count, 
                              energy_level=1.0, load_factor=0.0, cooperation_score=0.0):
        """添加或更新邻居信息"""
        current_time = time.time()
        
        # 计算距离
        distance = euclidean_distance(coords, self.get_own_coords()) if hasattr(self, 'own_coords') else 0
        
        neighbor_info = {
            'coords': coords,
            'velocity': velocity,
            'neighbor_count': neighbor_count,
            'energy_level': energy_level,
            'load_factor': load_factor,
            'cooperation_score': cooperation_score,
            'distance': distance,
            'last_seen': current_time,
            'link_remaining_time': self.calculate_link_remaining_time(coords, velocity),
            'link_stability': self.calculate_link_stability(velocity),
            'hello_count': self.neighbors.get(neighbor_id, {}).get('hello_count', 0) + 1,
            'ack_count': self.neighbors.get(neighbor_id, {}).get('ack_count', 0)
        }
        
        self.neighbors[neighbor_id] = neighbor_info
        
        # 更新协作评分
        self.update_cooperation_score(neighbor_id, cooperation_score)
    
    def update_beacon_info(self, beacon_id, beacon_coords, coverage_range, beacon_type, quality_metric):
        """更新信标信息"""
        self.beacons[beacon_id] = {
            'coords': beacon_coords,
            'coverage_range': coverage_range,
            'type': beacon_type,
            'quality_metric': quality_metric,
            'timestamp': time.time()
        }
    
    def update_global_position(self, node_id, coords, velocity, timestamp):
        """更新全局位置信息"""
        self.global_positions[node_id] = {
            'coords': coords,
            'velocity': velocity,
            'timestamp': timestamp
        }
    
    def calculate_distance(self, coords1, coords2):
        """计算两点间距离"""
        return euclidean_distance(coords1, coords2)
    
    def calculate_link_remaining_time(self, neighbor_coords, neighbor_velocity):
        """计算链路剩余时间"""
        if not hasattr(self, 'own_coords') or not hasattr(self, 'own_velocity'):
            return float('inf')
        
        # 相对位置和速度
        rel_pos = [neighbor_coords[i] - self.own_coords[i] for i in range(len(neighbor_coords))]
        rel_vel = [neighbor_velocity[i] - self.own_velocity[i] for i in range(len(neighbor_velocity))]
        
        # 计算相对速度的模长
        rel_speed = math.sqrt(sum(v**2 for v in rel_vel))
        
        if rel_speed == 0:
            return float('inf')
        
        # 当前距离
        current_distance = math.sqrt(sum(p**2 for p in rel_pos))
        
        # 计算到达通信范围边界的时间
        # 使用简化模型：t = (R - d) / v_rel
        remaining_time = max(0, (self.communication_range - current_distance) / rel_speed)
        
        return remaining_time
    
    def calculate_link_stability(self, neighbor_velocity, max_speed=60.0):
        """计算链路稳定性"""
        if not hasattr(self, 'own_velocity'):
            return 1.0
        
        # 计算相对速度
        rel_velocity = [neighbor_velocity[i] - self.own_velocity[i] for i in range(len(neighbor_velocity))]
        rel_speed = math.sqrt(sum(v**2 for v in rel_velocity))
        
        # 稳定性与相对速度成反比
        stability = max(0, 1.0 - rel_speed / max_speed)
        
        return stability
    
    def calculate_mobility_factor(self, neighbor_id):
        """计算移动性因子"""
        if neighbor_id not in self.neighbors:
            return 0.0
        
        neighbor = self.neighbors[neighbor_id]
        link_time = neighbor.get('link_remaining_time', 0)
        stability = neighbor.get('link_stability', 0)
        
        # 移动性因子结合链路剩余时间和稳定性
        mobility_factor = 0.6 * min(1.0, link_time / 10.0) + 0.4 * stability
        
        return mobility_factor
    
    def select_best_next_hop(self, destination_coords, current_coords, destination_id=None):
        """选择最佳下一跳"""
        if not self.neighbors:
            return None
        
        best_neighbor = None
        best_score = -float('inf')
        
        for neighbor_id, neighbor_info in self.neighbors.items():
            # 检查邻居是否仍然有效
            if time.time() - neighbor_info['last_seen'] > self.neighbor_timeout:
                continue
            
            # 计算到目标的进展
            current_distance = self.calculate_distance(current_coords, destination_coords)
            neighbor_distance = self.calculate_distance(neighbor_info['coords'], destination_coords)
            progress = max(0, current_distance - neighbor_distance)
            
            # 只考虑有正向进展的邻居
            if progress <= 0:
                continue
            
            # 计算综合评分
            score = self.calculate_neighbor_score(neighbor_id, destination_id, progress)
            
            if score > best_score:
                best_score = score
                best_neighbor = neighbor_id
        
        return best_neighbor
    
    def calculate_neighbor_score(self, neighbor_id, destination_id, progress):
        """计算邻居综合评分"""
        if neighbor_id not in self.neighbors:
            return -float('inf')
        
        neighbor = self.neighbors[neighbor_id]
        
        # 获取Q值
        q_value = 0.0
        if destination_id:
            q_value = self.q_table[(self.drone_id, destination_id)][neighbor_id]
        
        # 计算链路效用度量
        link_utility = self.calculate_link_utility(neighbor_id, progress)
        
        # 自适应权重系数
        alpha = self.calculate_adaptive_weight()
        
        # 融合评分
        score = (1 - alpha) * q_value + alpha * link_utility
        
        return score
    
    def calculate_link_utility(self, neighbor_id, progress):
        """计算链路效用度量"""
        if neighbor_id not in self.neighbors:
            return 0.0
        
        neighbor = self.neighbors[neighbor_id]
        
        # 归一化各个因子
        progress_norm = min(1.0, progress / self.communication_range)
        delay_norm = 1.0 - min(1.0, neighbor.get('estimated_delay', 0.1) / 1.0)  # 假设最大延迟1秒
        mobility_norm = self.calculate_mobility_factor(neighbor_id)
        neighbor_count_norm = min(1.0, neighbor.get('neighbor_count', 1) / 10.0)  # 假设最大邻居数10
        
        # 权重设置
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
        
        # 计算链路效用
        utility = (w1 * progress_norm + 
                  w2 * delay_norm + 
                  w3 * mobility_norm + 
                  w4 * neighbor_count_norm)
        
        return utility
    
    def calculate_adaptive_weight(self):
        """计算自适应权重系数"""
        # 基于邻居变化频率调整权重
        current_time = time.time()
        recent_neighbors = set()
        old_neighbors = set()
        
        for neighbor_id, neighbor_info in self.neighbors.items():
            if current_time - neighbor_info['last_seen'] < 5.0:  # 最近5秒的邻居
                recent_neighbors.add(neighbor_id)
            elif current_time - neighbor_info['last_seen'] < 10.0:  # 5-10秒前的邻居
                old_neighbors.add(neighbor_id)
        
        # 计算邻居变化率
        total_neighbors = len(recent_neighbors) + len(old_neighbors)
        if total_neighbors == 0:
            return 0.5
        
        change_rate = len(recent_neighbors.symmetric_difference(old_neighbors)) / total_neighbors
        
        # 权重范围[0.2, 0.8]
        alpha_min, alpha_max = 0.2, 0.8
        alpha = alpha_min + (alpha_max - alpha_min) * change_rate
        
        return alpha
    

    
    def update_cooperation_score(self, neighbor_id, new_score):
        """更新协作评分"""
        # 使用指数移动平均更新协作评分
        alpha = 0.3
        old_score = self.cooperation_scores[neighbor_id]
        self.cooperation_scores[neighbor_id] = alpha * new_score + (1 - alpha) * old_score
    
    def update_q_value(self, current_state, action, reward, next_state, next_best_action):
        """更新Q值"""
        learning_rate = self.adaptive_params['learning_rate']
        discount_factor = self.adaptive_params['discount_factor']
        
        current_q = self.q_table[current_state][action]
        next_q = self.q_table[next_state][next_best_action] if next_best_action else 0
        
        # Q学习更新公式
        new_q = current_q + learning_rate * (reward + discount_factor * next_q - current_q)
        self.q_table[current_state][action] = new_q
    
    def update_adaptive_parameters(self):
        """更新自适应参数"""
        # 基于链路质量更新学习率
        avg_link_quality = self.calculate_average_link_quality()
        self.adaptive_params['learning_rate'] = 0.05 + 0.15 * (1 - avg_link_quality)
        
        # 基于邻居变化频率更新折扣因子
        neighbor_change_rate = self.calculate_neighbor_change_rate()
        self.adaptive_params['discount_factor'] = 0.8 + 0.15 * (1 - neighbor_change_rate)
        
        # 基于网络稳定性更新Hello间隔
        stability = 1 - neighbor_change_rate
        self.adaptive_params['hello_interval'] = 1.0 + 3.0 * stability
    
    def calculate_average_link_quality(self):
        """计算平均链路质量"""
        if not self.neighbors:
            return 0.5
        
        total_quality = 0
        count = 0
        
        for neighbor_id, neighbor_info in self.neighbors.items():
            hello_count = neighbor_info.get('hello_count', 0)
            ack_count = neighbor_info.get('ack_count', 0)
            
            if hello_count > 0:
                quality = min(1.0, ack_count / hello_count)
                total_quality += quality
                count += 1
        
        return total_quality / count if count > 0 else 0.5
    
    def calculate_neighbor_change_rate(self):
        """计算邻居变化率"""
        current_time = time.time()
        recent_count = 0
        total_count = 0
        
        for neighbor_info in self.neighbors.values():
            total_count += 1
            if current_time - neighbor_info['last_seen'] < 5.0:
                recent_count += 1
        
        if total_count == 0:
            return 0.5
        
        return 1.0 - (recent_count / total_count)
    
    def cleanup_expired_entries(self):
        """清除过期条目"""
        current_time = time.time()
        
        # 清除过期邻居
        expired_neighbors = []
        for neighbor_id, neighbor_info in self.neighbors.items():
            if current_time - neighbor_info['last_seen'] > self.neighbor_timeout:
                expired_neighbors.append(neighbor_id)
        
        for neighbor_id in expired_neighbors:
            del self.neighbors[neighbor_id]
            if neighbor_id in self.cooperation_scores:
                del self.cooperation_scores[neighbor_id]
        
        # 清除过期信标（假设信标有效期为60秒）
        expired_beacons = []
        for beacon_id, beacon_info in self.beacons.items():
            if current_time - beacon_info['timestamp'] > 60.0:
                expired_beacons.append(beacon_id)
        
        for beacon_id in expired_beacons:
            del self.beacons[beacon_id]
        
        # 清除过期全局位置信息（有效期30秒）
        expired_positions = []
        for node_id, position_info in self.global_positions.items():
            if current_time - position_info['timestamp'] > 30.0:
                expired_positions.append(node_id)
        
        for node_id in expired_positions:
            del self.global_positions[node_id]
    
    def set_own_position(self, coords, velocity):
        """设置自身位置和速度"""
        self.own_coords = coords
        self.own_velocity = velocity
    
    def get_own_coords(self):
        """获取自身坐标"""
        return getattr(self, 'own_coords', [0, 0, 0])
    
    def get_neighbor_count(self):
        """获取邻居数量"""
        current_time = time.time()
        active_neighbors = 0
        
        for neighbor_info in self.neighbors.values():
            if current_time - neighbor_info['last_seen'] <= self.neighbor_timeout:
                active_neighbors += 1
        
        return active_neighbors
    
    def get_max_q_value(self):
        """获取最大Q值"""
        max_q = 0.0
        for state_actions in self.q_table.values():
            for q_value in state_actions.values():
                max_q = max(max_q, q_value)
        return max_q