#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from entities.packet import Packet


class GbmcrHelloPacket(Packet):
    """
    GBMCR协议的Hello包，用于邻居发现和信标信息传播
    包含节点标识、时间戳、节点位置、速度、节点数量等信息
    """
    def __init__(self, src_drone_id, src_coords, src_velocity, neighbor_count, timestamp=None, simulator=None, channel_id=0):
        packet_id = f"HELLO_{src_drone_id}_{timestamp if timestamp else time.time()}"
        super().__init__(packet_id, 64, timestamp if timestamp else time.time(), simulator, channel_id)  # Hello包大小64字节
        self.src_drone_id = src_drone_id
        self.src_coords = src_coords
        self.src_velocity = src_velocity
        self.neighbor_count = neighbor_count
        self.timestamp = timestamp if timestamp else time.time()
        self.energy_level = 1.0  # 能量水平
        self.load_factor = 0.0   # 负载因子
        self.cooperation_score = 0.0  # 协作评分


class GbmcrBeaconPacket(Packet):
    """
    GBMCR协议的地理信标包，用于建立和维护地理信标
    包含信标ID、位置、覆盖范围、类型和质量指标等
    """
    def __init__(self, src_drone_id, beacon_id, beacon_coords, coverage_range, beacon_type="normal", timestamp=None, simulator=None, channel_id=0):
        packet_id = f"BEACON_{beacon_id}_{timestamp if timestamp else time.time()}"
        super().__init__(packet_id, 128, timestamp if timestamp else time.time(), simulator, channel_id)  # 信标包大小128字节
        self.beacon_id = beacon_id
        self.beacon_coords = beacon_coords
        self.coverage_range = coverage_range
        self.beacon_type = beacon_type  # normal, emergency, strategic
        self.quality_metric = 1.0
        self.timestamp = time.time()
        self.path_nodes = []  # 记录途径的节点信息
        self.visited_nodes = set()  # 已访问的节点集合
        
    def add_path_node(self, node_id, coords, velocity):
        """添加途径节点信息"""
        if node_id not in self.visited_nodes:
            self.path_nodes.append({
                'node_id': node_id,
                'coords': coords,
                'velocity': velocity,
                'timestamp': time.time()
            })
            self.visited_nodes.add(node_id)


class GbmcrRouteRequest(Packet):
    """
    GBMCR协议的路由请求包
    包含源/目的ID、位置、跳数、路径质量和已访问信标等
    """
    def __init__(self, src_drone_id, dst_drone_id, src_coords, dst_coords, request_id, timestamp=None, simulator=None, channel_id=0):
        packet_id = f"RREQ_{request_id}_{timestamp if timestamp else time.time()}"
        super().__init__(packet_id, 256, timestamp if timestamp else time.time(), simulator, channel_id)  # 路由请求包大小256字节
        self.src_coords = src_coords
        self.dst_coords = dst_coords
        self.request_id = request_id
        self.hop_count = 0
        self.path_quality = 1.0
        self.visited_beacons = []  # 已访问的信标列表
        self.path_nodes = []  # 路径节点列表
        self.timestamp = time.time()
        
    def add_hop(self, node_id, coords, link_quality):
        """添加一跳"""
        self.hop_count += 1
        self.path_nodes.append({
            'node_id': node_id,
            'coords': coords,
            'timestamp': time.time()
        })
        self.path_quality *= link_quality


class GbmcrRouteReply(Packet):
    """
    GBMCR协议的路由回复包
    包含请求ID、路由路径、路径质量、预估延迟和路径可靠性等
    """
    def __init__(self, src_drone_id, dst_drone_id, request_id, route_path, timestamp=None, simulator=None, channel_id=0):
        packet_id = f"RREP_{request_id}_{timestamp if timestamp else time.time()}"
        super().__init__(packet_id, 256, timestamp if timestamp else time.time(), simulator, channel_id)  # 路由回复包大小256字节
        self.request_id = request_id
        self.route_path = route_path  # 完整路由路径
        self.path_quality = 1.0
        self.estimated_delay = 0.0
        self.path_reliability = 1.0
        self.timestamp = time.time()
        
    def calculate_metrics(self, link_qualities, delays):
        """计算路径指标"""
        if link_qualities:
            self.path_quality = 1.0
            for quality in link_qualities:
                self.path_quality *= quality
        
        if delays:
            self.estimated_delay = sum(delays)
            
        # 路径可靠性基于质量和延迟计算
        self.path_reliability = self.path_quality * (1.0 / (1.0 + self.estimated_delay))


class GbmcrHolePacket(Packet):
    """
    GBMCR协议的路由空洞通知包
    用于通知邻居节点当前节点遇到路由空洞
    """
    def __init__(self, src_drone_id, dst_drone_id, timestamp=None, simulator=None, channel_id=0):
        packet_id = f"HOLE_{src_drone_id}_{dst_drone_id}_{timestamp if timestamp else time.time()}"
        super().__init__(packet_id, 32, timestamp if timestamp else time.time(), simulator, channel_id)  # 空洞通知包大小32字节
        self.dst_drone_id = dst_drone_id  # 目标节点ID
        self.timestamp = time.time()