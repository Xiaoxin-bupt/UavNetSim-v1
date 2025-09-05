import time
import random
import math
from collections import defaultdict

from .gbmcr_packet import GbmcrHelloPacket, GbmcrBeaconPacket, GbmcrHolePacket
from .gbmcr_table import GbmcrNeighborTable
from entities.packet import Packet
from utils.util_function import euclidean_distance_3d as euclidean_distance


class Gbmcr:
    """
    基于地理信标的多目标智能协作路由策略 (Geographic Beacon-based Routing Strategy for Multi-Destination Intelligent Collaboration)
    
    主要特性：
    1. 地理信标机制：通过稀疏触发的全局位置梯度同步策略减少控制信令开销
    2. 多维链路效用度量模型：综合评估转发进展、链路稳定性、单跳延迟及邻居密度
    3. 智能协作路由决策：通过链路效用加权与局部信息交互对Q值计算进行优化
    4. 路由空洞避免：特殊的路由空洞避免算法与路由恢复准则
    """
    
    def __init__(self, simulator, drone):
        self.simulator = simulator
        self.drone = drone
        self.random_gen = random.Random(self.drone.identifier + self.simulator.seed + 10)
        
        # 协议参数
        self.hello_interval = 1.0  # Hello包发送间隔
        self.beacon_interval = 3.0  # 信标更新间隔
        self.neighbor_timeout = 3.0  # 邻居超时时间
        self.ack_timeout = 1.0  # ACK超时时间
        
        # 邻居表和信标管理
        self.neighbor_table = GbmcrNeighborTable(
            drone_id=drone.identifier,
            communication_range=drone.communication_range,
            neighbor_timeout=self.neighbor_timeout
        )
        
        # 等待列表：存储等待路由的数据包
        self.waiting_list = []
        
        # ACK等待列表：{packet_id: (timestamp, next_hop, packet)}
        self.ack_waiting = {}
        
        # 路由缓存：{destination: route_info}
        self.route_cache = {}
        
        # 信标相关
        self.is_beacon = False
        self.beacon_quality = 1.0
        self.beacon_coverage = drone.communication_range
        
        # Q学习参数
        self.epsilon = 0.1  # ε-greedy策略参数
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
        # 统计信息
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_dropped': 0,
            'hello_sent': 0,
            'beacon_sent': 0,
            'route_discoveries': 0
        }
        
        # 启动周期性进程
        self.simulator.env.process(self.broadcast_hello_packet_periodically())
        self.simulator.env.process(self.broadcast_beacon_periodically())
        self.simulator.env.process(self.check_waiting_list_periodically())
        self.simulator.env.process(self.check_ack_timeouts_periodically())
        self.simulator.env.process(self.update_beacon_status_periodically())
    
    def broadcast_hello_packet_periodically(self):
        """周期性广播Hello包"""
        while True:
            try:
                # 更新自身位置信息到邻居表
                self.neighbor_table.set_own_position(
                    self.drone.coords, 
                    getattr(self.drone, 'velocity', [0, 0, 0])
                )
                
                # 创建Hello包
                hello_packet = self.create_hello_packet()
                
                # 广播Hello包
                self.broadcast_hello_packet(hello_packet)
                
                # 清理过期条目
                self.neighbor_table.cleanup_expired_entries()
                
                # 更新自适应参数
                self.neighbor_table.update_adaptive_parameters()
                
                # 使用自适应Hello间隔
                interval = self.neighbor_table.adaptive_params['hello_interval']
                yield self.simulator.env.timeout(interval)
                
            except Exception as e:
                print(f"Error in broadcast_hello_packet_periodically: {e}")
                yield self.simulator.env.timeout(self.hello_interval)
    
    def create_hello_packet(self):
        """创建Hello包"""
        return GbmcrHelloPacket(
            src_drone_id=self.drone.identifier,
            src_coords=self.drone.coords,
            src_velocity=getattr(self.drone, 'velocity', [0, 0, 0]),
            neighbor_count=self.neighbor_table.get_neighbor_count(),
            timestamp=time.time(),
            simulator=self.simulator,
            channel_id=0
        )
    
    def broadcast_hello_packet(self, hello_packet):
        """广播Hello包"""
        try:
            self.drone.transmitting_queue.put(hello_packet)
            self.stats['hello_sent'] += 1
        except Exception as e:
            print(f"Error broadcasting hello packet: {e}")
    
    def broadcast_beacon_periodically(self):
        """周期性广播信标包"""
        while True:
            try:
                if self.is_beacon:
                    beacon_packet = self.create_beacon_packet()
                    self.broadcast_beacon_packet(beacon_packet)
                
                # 信标更新周期是Hello间隔的α倍（这里设为5倍）
                yield self.simulator.env.timeout(self.beacon_interval)
                
            except Exception as e:
                print(f"Error in broadcast_beacon_periodically: {e}")
                yield self.simulator.env.timeout(self.beacon_interval)
    
    def create_beacon_packet(self):
        """创建信标包"""
        return GbmcrBeaconPacket(
            src_drone_id=self.drone.identifier,
            beacon_id=f"beacon_{self.drone.identifier}",
            beacon_coords=self.drone.coords,
            coverage_range=self.beacon_coverage,
            beacon_type="normal",
            timestamp=time.time(),
            simulator=self.simulator,
            channel_id=0
        )
    
    def broadcast_beacon_packet(self, beacon_packet):
        """广播信标包"""
        try:
            self.drone.transmitting_queue.put(beacon_packet)
            self.stats['beacon_sent'] += 1
        except Exception as e:
            print(f"Error broadcasting beacon packet: {e}")
    

    # 下一跳选择算法，路由被外界调用的基本方法
    def next_hop_selection(self, packet):
        """下一跳选择算法"""
        if not hasattr(packet, 'dst_drone_id') or packet.dst_drone_id is None:
            return None
        
        destination_id = packet.dst_drone_id
        
        # 如果目标就是自己，不需要转发
        if destination_id == self.drone.identifier:
            return None
        
        # 首先尝试从邻居表中找到目标节点
        if destination_id in self.neighbor_table.neighbors:
            return destination_id
        
        # 尝试从全局位置表获取目标位置
        destination_coords = None
        if destination_id in self.neighbor_table.global_positions:
            destination_coords = self.neighbor_table.global_positions[destination_id]['coords']
        
        if destination_coords is None:
            # 无法确定目标位置，将数据包加入等待列表
            self.add_to_waiting_list(packet)
            return None
        
        # 使用ε-greedy策略选择下一跳
        if self.random_gen.random() < self.epsilon:
            # 探索：随机选择一个有进展的邻居
            return self.select_random_progressive_neighbor(destination_coords)
        else:
            # 利用：选择最佳下一跳
            return self.select_best_next_hop(destination_coords, destination_id)
    
    def select_best_next_hop(self, destination_coords, destination_id):
        """选择最佳下一跳"""
        # 获取有进展的邻居
        progressive_neighbors = self.get_progressive_neighbors(destination_coords)
        
        if not progressive_neighbors:
            # 遇到路由空洞，触发空洞避免机制
            return self.handle_routing_hole(destination_id, destination_coords)
        
        # 使用邻居表的选择算法
        best_neighbor = self.neighbor_table.select_best_next_hop(
            destination_coords, self.drone.coords, destination_id
        )
        
        return best_neighbor
    
    def get_progressive_neighbors(self, destination_coords):
        """获取有进展的邻居"""
        progressive_neighbors = []
        current_distance = euclidean_distance(self.drone.coords, destination_coords)
        
        for neighbor_id, neighbor_info in self.neighbor_table.neighbors.items():
            # 检查邻居是否仍然有效
            if time.time() - neighbor_info['last_seen'] > self.neighbor_timeout:
                continue
            
            neighbor_distance = euclidean_distance(neighbor_info['coords'], destination_coords)
            
            # 只考虑有正向进展的邻居
            if neighbor_distance < current_distance:
                progressive_neighbors.append(neighbor_id)
        
        return progressive_neighbors
    
    def select_random_progressive_neighbor(self, destination_coords):
        """随机选择一个有进展的邻居（用于探索）"""
        progressive_neighbors = self.get_progressive_neighbors(destination_coords)
        
        if not progressive_neighbors:
            return None
        
        return self.random_gen.choice(progressive_neighbors)
    
    def handle_routing_hole(self, destination_id, destination_coords):
        """处理路由空洞"""
        # 广播HOLE消息
        hole_packet = GbmcrHolePacket(self.drone.identifier, destination_id)
        self.broadcast_hello_packet(hole_packet)
        
        # 使用路由恢复准则选择下一跳
        return self.select_hole_recovery_neighbor(destination_coords)
    
    def select_hole_recovery_neighbor(self, destination_coords):
        """路由空洞恢复：选择节点度大且距离目标较近的邻居"""
        if not self.neighbor_table.neighbors:
            return None
        
        best_neighbor = None
        best_score = -float('inf')
        
        for neighbor_id, neighbor_info in self.neighbor_table.neighbors.items():
            # 检查邻居是否仍然有效
            if time.time() - neighbor_info['last_seen'] > self.neighbor_timeout:
                continue
            
            # 计算距离目标的距离
            distance_to_dest = euclidean_distance(neighbor_info['coords'], destination_coords)
            distance_score = 1.0 / (1.0 + distance_to_dest / self.drone.communication_range)
            
            # 节点度评分
            neighbor_count = neighbor_info.get('neighbor_count', 1)
            degree_score = min(1.0, neighbor_count / 10.0)  # 假设最大邻居数为10
            
            # 综合评分
            score = 0.6 * distance_score + 0.4 * degree_score
            
            if score > best_score:
                best_score = score
                best_neighbor = neighbor_id
        
        return best_neighbor
    

    
    def packet_reception(self, packet):
        """数据包接收处理"""
        try:
            self.stats['packets_received'] += 1
            
            if isinstance(packet, GbmcrHelloPacket):
                self.handle_hello_packet(packet)
            elif isinstance(packet, GbmcrBeaconPacket):
                self.handle_beacon_packet(packet)
            elif isinstance(packet, GbmcrHolePacket):
                self.handle_hole_packet(packet)
            elif packet.packet_type == "DATA":
                self.handle_data_packet(packet)
            elif packet.packet_type == "ACK":
                self.handle_ack_packet(packet)
            else:
                print(f"Unknown packet type: {packet.packet_type}")
                
        except Exception as e:
            print(f"Error in packet_reception: {e}")
    
    def handle_hello_packet(self, packet):
        """处理Hello包"""
        # 更新邻居表
        self.neighbor_table.add_or_update_neighbor(
            neighbor_id=packet.src_drone_id,
            coords=packet.src_coords,
            velocity=packet.src_velocity,
            neighbor_count=packet.neighbor_count,
            energy_level=getattr(packet, 'energy_level', 1.0),
            load_factor=getattr(packet, 'load_factor', 0.0),
            cooperation_score=getattr(packet, 'cooperation_score', 0.0)
        )
        
        # 发送HACK回复（Hello ACK）
        self.send_hack_packet(packet.src_drone_id)
    
    def handle_beacon_packet(self, packet):
        """处理信标包"""
        # 更新信标信息
        self.neighbor_table.update_beacon_info(
            beacon_id=packet.beacon_id,
            beacon_coords=packet.beacon_coords,
            coverage_range=packet.coverage_range,
            beacon_type=packet.beacon_type,
            quality_metric=getattr(packet, 'quality_metric', 1.0)
        )
        
        # 更新全局位置表
        for node_info in packet.path_nodes:
            self.neighbor_table.update_global_position(
                node_id=node_info['node_id'],
                coords=node_info['coords'],
                velocity=node_info['velocity'],
                timestamp=node_info['timestamp']
            )
        
        # 如果信标包还没有访问过当前节点，继续转发
        if self.drone.identifier not in packet.visited_nodes:
            packet.add_path_node(
                self.drone.identifier,
                self.drone.coords,
                getattr(self.drone, 'velocity', [0, 0, 0])
            )
            
            # 向未访问过的邻居转发
            self.forward_beacon_packet(packet)
    
    def handle_hole_packet(self, packet):
        """处理路由空洞通知包"""
        # 将发送节点到目标的Q值设为最小值
        destination_id = packet.dst_drone_id
        sender_id = packet.src_drone_id
        
        # 惩罚该路径
        state = (self.drone.identifier, destination_id)
        self.neighbor_table.q_table[state][sender_id] = -1.0
    
    def handle_data_packet(self, packet):
        """处理数据包"""
        # 如果是目标节点，接收数据包
        if packet.dst_drone_id == self.drone.identifier:
            self.send_ack_packet(packet)
            return
        
        # 否则转发数据包
        next_hop = self.next_hop_selection(packet)
        
        if next_hop:
            # 记录发送时间用于计算奖励
            packet.forwarding_start_time = time.time()
            packet.current_hop = self.drone.identifier
            packet.next_hop = next_hop
            
            # 添加到ACK等待列表
            packet_id = f"{packet.src_drone_id}_{packet.dst_drone_id}_{packet.creation_time}"
            self.ack_waiting[packet_id] = (time.time(), next_hop, packet)
            
            # 转发数据包
            self.drone.transmitting_queue.put(packet)
            self.stats['packets_sent'] += 1
        else:
            # 无法找到下一跳，丢弃数据包
            self.stats['packets_dropped'] += 1
    
    def handle_ack_packet(self, packet):
        """处理ACK包"""
        # 从ACK等待列表中移除
        packet_id = getattr(packet, 'original_packet_id', None)
        if packet_id and packet_id in self.ack_waiting:
            send_time, next_hop, original_packet = self.ack_waiting[packet_id]
            del self.ack_waiting[packet_id]
            
            # 计算奖励并更新Q值
            self.calculate_and_update_reward(original_packet, packet, send_time)
    
    def send_ack_packet(self, original_packet):
        """发送ACK包"""
        try:
            ack_packet = Packet(
                packet_id=f"ACK_{self.drone.identifier}_{original_packet.src_drone_id}_{time.time()}",
                packet_length=32,
                creation_time=self.simulator.env.now,
                simulator=self.simulator,
                channel_id=0
            )
            
            # 设置ACK包的源和目标信息
            ack_packet.src_drone_id = self.drone.identifier
            ack_packet.dst_drone_id = original_packet.src_drone_id
            ack_packet.packet_type = "ACK"
            
            # 添加原始数据包信息
            ack_packet.original_packet_id = f"{original_packet.src_drone_id}_{original_packet.dst_drone_id}_{original_packet.creation_time}"
            ack_packet.transmission_delay = getattr(original_packet, 'transmission_delay', 0.0)
            ack_packet.mac_delay = getattr(original_packet, 'mac_delay', 0.0)
            
            self.drone.transmitting_queue.put(ack_packet)
            
        except Exception as e:
            print(f"Error sending ACK packet: {e}")
    
    def send_hack_packet(self, target_drone_id):
        """发送HACK包（Hello ACK）"""
        try:
            hack_packet = Packet(
                packet_id=f"HACK_{self.drone.identifier}_{target_drone_id}_{time.time()}",
                packet_length=32,
                creation_time=self.simulator.env.now,
                simulator=self.simulator,
                channel_id=0
            )
            
            # 设置HACK包的源和目标信息
            hack_packet.src_drone_id = self.drone.identifier
            hack_packet.dst_drone_id = target_drone_id
            hack_packet.packet_type = "HACK"
            
            self.drone.transmitting_queue.put(hack_packet)
            
        except Exception as e:
            print(f"Error sending HACK packet: {e}")
    
    def forward_beacon_packet(self, beacon_packet):
        """转发信标包"""
        # 向未在信标中出现过的邻居节点转发
        for neighbor_id in self.neighbor_table.neighbors:
            if neighbor_id not in beacon_packet.visited_nodes:
                # 创建新的信标包副本
                new_beacon = GbmcrBeaconPacket(
                    src_drone_id=self.drone.identifier,
                    beacon_id=beacon_packet.beacon_id,
                    beacon_coords=beacon_packet.beacon_coords,
                    coverage_range=beacon_packet.coverage_range,
                    beacon_type=beacon_packet.beacon_type
                )
                
                # 复制路径信息
                new_beacon.path_nodes = beacon_packet.path_nodes.copy()
                new_beacon.visited_nodes = beacon_packet.visited_nodes.copy()
                
                # 设置目标
                new_beacon.dst_drone_id = neighbor_id
                
                self.drone.transmitting_queue.put(new_beacon)
    
    def calculate_and_update_reward(self, original_packet, ack_packet, send_time):
        """计算奖励并更新Q值"""
        try:
            # 计算各项指标
            current_time = time.time()
            transmission_delay = current_time - send_time
            
            # 从ACK包中获取MAC延迟
            mac_delay = getattr(ack_packet, 'mac_delay', 0.0)
            total_delay = transmission_delay + mac_delay
            
            # 计算距离进展
            next_hop = original_packet.next_hop
            if next_hop in self.neighbor_table.neighbors:
                neighbor_coords = self.neighbor_table.neighbors[next_hop]['coords']
                
                # 获取目标位置
                destination_coords = None
                if original_packet.dst_drone_id in self.neighbor_table.global_positions:
                    destination_coords = self.neighbor_table.global_positions[original_packet.dst_drone_id]['coords']
                elif original_packet.dst_drone_id in self.neighbor_table.neighbors:
                    destination_coords = self.neighbor_table.neighbors[original_packet.dst_drone_id]['coords']
                
                if destination_coords:
                    current_distance = euclidean_distance(self.drone.coords, destination_coords)
                    neighbor_distance = euclidean_distance(neighbor_coords, destination_coords)
                    progress = max(0, current_distance - neighbor_distance)
                    
                    # 计算奖励
                    reward = self.calculate_reward(progress, total_delay, next_hop)
                    
                    # 更新Q值
                    current_state = (self.drone.identifier, original_packet.dst_drone_id)
                    next_state = (next_hop, original_packet.dst_drone_id)
                    
                    # 获取下一状态的最佳动作
                    next_best_action = self.get_best_action_for_state(next_state)
                    
                    # 更新Q值
                    self.neighbor_table.update_q_value(
                        current_state, next_hop, reward, next_state, next_best_action
                    )
                    
        except Exception as e:
            print(f"Error calculating reward: {e}")
    
    def calculate_reward(self, progress, delay, next_hop):
        """计算奖励函数"""
        # 归一化各项因子
        progress_norm = min(1.0, progress / self.drone.communication_range)
        delay_norm = max(0, 1.0 - delay / 1.0)  # 假设最大可接受延迟为1秒
        
        # 获取邻居信息
        mobility_factor = 0.5  # 默认值
        neighbor_count_norm = 0.5  # 默认值
        
        if next_hop in self.neighbor_table.neighbors:
            mobility_factor = self.neighbor_table.calculate_mobility_factor(next_hop)
            neighbor_count = self.neighbor_table.neighbors[next_hop].get('neighbor_count', 1)
            neighbor_count_norm = min(1.0, neighbor_count / 10.0)
        
        # 权重设置
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
        
        # 计算奖励
        reward = (w1 * progress_norm + 
                 w2 * delay_norm + 
                 w3 * mobility_factor + 
                 w4 * neighbor_count_norm)
        
        return reward
    
    def get_best_action_for_state(self, state):
        """获取状态的最佳动作"""
        if state not in self.neighbor_table.q_table:
            return None
        
        state_actions = self.neighbor_table.q_table[state]
        if not state_actions:
            return None
        
        return max(state_actions, key=state_actions.get)
    
    def add_to_waiting_list(self, packet):
        """将数据包添加到等待列表"""
        self.waiting_list.append({
            'packet': packet,
            'timestamp': time.time()
        })
    
    def check_waiting_list_periodically(self):
        """周期性检查等待列表"""
        while True:
            try:
                self.check_waiting_list()
                yield self.simulator.env.timeout(1.0)  # 每秒检查一次
            except Exception as e:
                print(f"Error in check_waiting_list_periodically: {e}")
                yield self.simulator.env.timeout(1.0)
    
    def check_waiting_list(self):
        """检查等待列表中的数据包"""
        current_time = time.time()
        processed_packets = []
        
        for item in self.waiting_list:
            packet = item['packet']
            timestamp = item['timestamp']
            
            # 检查是否超时（10秒）
            if current_time - timestamp > 10.0:
                processed_packets.append(item)
                self.stats['packets_dropped'] += 1
                continue
            
            # 尝试找到下一跳
            next_hop = self.next_hop_selection(packet)
            
            if next_hop:
                # 找到下一跳，转发数据包
                packet.forwarding_start_time = time.time()
                packet.current_hop = self.drone.identifier
                packet.next_hop = next_hop
                
                # 添加到ACK等待列表
                packet_id = f"{packet.src_drone_id}_{packet.dst_drone_id}_{packet.creation_time}"
                self.ack_waiting[packet_id] = (time.time(), next_hop, packet)
                
                # 转发数据包
                self.drone.transmitting_queue.put(packet)
                self.stats['packets_sent'] += 1
                
                processed_packets.append(item)
        
        # 从等待列表中移除已处理的数据包
        for item in processed_packets:
            self.waiting_list.remove(item)
    
    def check_ack_timeouts_periodically(self):
        """周期性检查ACK超时"""
        while True:
            try:
                self.check_ack_timeouts()
                yield self.simulator.env.timeout(0.5)  # 每0.5秒检查一次
            except Exception as e:
                print(f"Error in check_ack_timeouts_periodically: {e}")
                yield self.simulator.env.timeout(0.5)
    
    def check_ack_timeouts(self):
        """检查ACK超时"""
        current_time = time.time()
        timeout_packets = []
        
        for packet_id, (send_time, next_hop, packet) in self.ack_waiting.items():
            if current_time - send_time > self.ack_timeout:
                timeout_packets.append(packet_id)
                
                # 给予负奖励
                current_state = (self.drone.identifier, packet.dst_drone_id)
                next_state = (next_hop, packet.dst_drone_id)
                next_best_action = self.get_best_action_for_state(next_state)
                
                # 负奖励
                negative_reward = -0.5
                self.neighbor_table.update_q_value(
                    current_state, next_hop, negative_reward, next_state, next_best_action
                )
        
        # 移除超时的数据包
        for packet_id in timeout_packets:
            del self.ack_waiting[packet_id]
    
    def update_beacon_status_periodically(self):
        """周期性更新信标状态"""
        while True:
            try:
                self.evaluate_beacon_candidacy()
                yield self.simulator.env.timeout(30.0)  # 每30秒评估一次
            except Exception as e:
                print(f"Error in update_beacon_status_periodically: {e}")
                yield self.simulator.env.timeout(30.0)
    
    def evaluate_beacon_candidacy(self):
        """评估信标候选资格"""
        # 基于邻居数量、位置稳定性等因素决定是否成为信标
        neighbor_count = self.neighbor_table.get_neighbor_count()
        
        # 如果邻居数量较多且位置相对稳定，可以成为信标
        if neighbor_count >= 3 and not self.is_beacon:
            self.become_beacon()
        elif neighbor_count < 2 and self.is_beacon:
            self.stop_being_beacon()
    
    def become_beacon(self):
        """成为信标"""
        self.is_beacon = True
        self.beacon_quality = self.calculate_beacon_quality()
        print(f"Drone {self.drone.identifier} became a beacon")
    
    def stop_being_beacon(self):
        """停止成为信标"""
        self.is_beacon = False
        print(f"Drone {self.drone.identifier} stopped being a beacon")
    
    def calculate_beacon_quality(self):
        """计算信标质量"""
        neighbor_count = self.neighbor_table.get_neighbor_count()
        avg_link_quality = self.neighbor_table.calculate_average_link_quality()
        
        # 信标质量基于邻居数量和链路质量
        quality = 0.6 * min(1.0, neighbor_count / 5.0) + 0.4 * avg_link_quality
        
        return quality
    
    def penalize(self, neighbor_id, destination_id):
        """惩罚机制"""
        # 对特定邻居和目标的路径给予负奖励
        state = (self.drone.identifier, destination_id)
        current_q = self.neighbor_table.q_table[state][neighbor_id]
        
        # 减少Q值
        penalty = 0.1
        self.neighbor_table.q_table[state][neighbor_id] = current_q - penalty
    
    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0