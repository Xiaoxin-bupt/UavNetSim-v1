#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBMCR路由协议简化测试脚本

测试内容：
1. 基本路由功能测试
2. 地理信标机制测试
3. Q学习算法测试
4. 数据包处理测试
"""

import sys
import os
import time
import random
import simpy
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from entities.packet import Packet
from routing.gbmcr.gbmcr import Gbmcr
from routing.gbmcr.gbmcr_packet import GbmcrHelloPacket, GbmcrBeaconPacket
from routing.gbmcr.gbmcr_table import GbmcrNeighborTable
from utils.util_function import euclidean_distance_3d as euclidean_distance
from utils.config import *


class MockDrone:
    """模拟无人机类，用于测试"""
    
    def __init__(self, identifier, coords, env, communication_range=200):
        self.identifier = identifier
        self.coords = coords
        self.velocity = [random.uniform(-5, 5), random.uniform(-5, 5), 0]
        self.communication_range = communication_range
        self.transmitting_queue = simpy.Store(env, capacity=100)
        self.routing_protocol = None
        self.env = env
    
    def set_routing_protocol(self, protocol):
        self.routing_protocol = protocol


class MockSimulator:
    """模拟仿真器类，用于测试"""
    
    def __init__(self):
        self.env = simpy.Environment()
        self.drones = []
        self.seed = 42  # 添加seed属性用于随机数生成器初始化
    
    def run_for_duration(self, duration):
        """运行指定时间"""
        self.env.run(until=self.env.now + duration)


class GbmcrTester:
    """GBMCR协议测试类"""
    
    def __init__(self):
        self.simulator = None
        self.drones = []
    
    def setup_test_environment(self, num_drones=10, area_size=500):
        """设置测试环境"""
        print(f"设置测试环境：{num_drones}个无人机，区域大小{area_size}x{area_size}")
        
        # 创建仿真器
        self.simulator = MockSimulator()
        
        # 创建无人机
        self.drones = []
        for i in range(num_drones):
            # 随机位置 - 缩小区域以确保无人机在通信范围内
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            z = random.uniform(100, 300)  # 飞行高度
            
            # 创建无人机
            drone = MockDrone(
                identifier=i,
                coords=[x, y, z],
                env=self.simulator.env,
                communication_range=200
            )
            
            self.drones.append(drone)
            self.simulator.drones.append(drone)
        
        # 为每个无人机设置GBMCR路由协议
        for drone in self.drones:
            routing_protocol = Gbmcr(
                simulator=self.simulator,
                drone=drone
            )
            drone.set_routing_protocol(routing_protocol)
        
        print(f"成功创建{len(self.drones)}个无人机")
    
    def test_neighbor_table(self):
        """测试邻居表功能"""
        print("\n=== 邻居表功能测试 ===")
        
        # 选择测试无人机
        test_drone = self.drones[0]
        neighbor_table = test_drone.routing_protocol.neighbor_table
        
        # 设置邻居表的自身位置信息
        neighbor_table.set_own_position(test_drone.coords, test_drone.velocity)
        
        # 添加邻居
        for i, drone in enumerate(self.drones[1:6]):  # 添加5个邻居
            distance = euclidean_distance(test_drone.coords, drone.coords)
            if distance <= test_drone.communication_range:
                neighbor_table.add_or_update_neighbor(
                neighbor_id=drone.identifier,
                coords=drone.coords,
                velocity=drone.velocity,
                neighbor_count=random.randint(1, 5),
                energy_level=random.uniform(0.5, 1.0),
                load_factor=random.uniform(0.0, 0.5),
                    cooperation_score=random.uniform(0.0, 1.0)
                )
                print(f"添加邻居 {drone.identifier}，距离: {distance:.2f}m")
        
        neighbor_count = neighbor_table.get_neighbor_count()
        print(f"邻居表中共有 {neighbor_count} 个邻居")
        
        return neighbor_count > 0
    
    def test_hello_packet_creation(self):
        """测试Hello包创建"""
        print("\n=== Hello包创建测试 ===")
        
        test_drone = self.drones[0]
        routing_protocol = test_drone.routing_protocol
        
        # 创建Hello包
        hello_packet = routing_protocol.create_hello_packet()
        
        print(f"Hello包信息:")
        print(f"  源节点ID: {hello_packet.src_drone_id}")
        print(f"  源坐标: {hello_packet.src_coords}")
        print(f"  源速度: {hello_packet.src_velocity}")
        print(f"  邻居数量: {hello_packet.neighbor_count}")
        
        return isinstance(hello_packet, GbmcrHelloPacket)
    
    def test_beacon_mechanism(self):
        """测试信标机制"""
        print("\n=== 信标机制测试 ===")
        
        # 让一些无人机成为信标
        beacon_count = 0
        for i, drone in enumerate(self.drones[:3]):  # 前3个无人机成为信标
            drone.routing_protocol.become_beacon()
            beacon_count += 1
            print(f"无人机 {drone.identifier} 成为信标，质量: {drone.routing_protocol.beacon_quality:.3f}")
        
        # 测试信标包创建
        beacon_drone = self.drones[0]
        if beacon_drone.routing_protocol.is_beacon:
            beacon_packet = beacon_drone.routing_protocol.create_beacon_packet()
            print(f"\n信标包信息:")
            print(f"  信标ID: {beacon_packet.beacon_id}")
            print(f"  信标坐标: {beacon_packet.beacon_coords}")
            print(f"  覆盖范围: {beacon_packet.coverage_range}")
            print(f"  信标类型: {beacon_packet.beacon_type}")
        
        return beacon_count > 0
    
    def test_q_learning_basic(self):
        """测试Q学习基本功能"""
        print("\n=== Q学习基本功能测试 ===")
        
        test_drone = self.drones[0]
        neighbor_table = test_drone.routing_protocol.neighbor_table
        
        # 添加一些邻居用于Q学习
        for drone in self.drones[1:4]:
            neighbor_table.add_or_update_neighbor(
                neighbor_id=drone.identifier,
                coords=drone.coords,
                velocity=drone.velocity,
                neighbor_count=random.randint(1, 5)
            )
        
        # 模拟Q值更新
        destination_id = self.drones[-1].identifier  # 选择最后一个无人机作为目标
        state = (test_drone.identifier, destination_id)
        
        # 为每个邻居更新Q值
        for neighbor_id in neighbor_table.neighbors.keys():
            reward = random.uniform(0.3, 0.9)
            next_state = (neighbor_id, destination_id)
            next_action = neighbor_id
            
            neighbor_table.update_q_value(state, neighbor_id, reward, next_state, next_action)
            
            q_value = neighbor_table.q_table[state].get(neighbor_id, 0)
            print(f"邻居 {neighbor_id} 的Q值: {q_value:.3f}")
        
        # 测试最佳下一跳选择
        destination_coords = self.drones[-1].coords
        best_neighbor = neighbor_table.select_best_next_hop(
            destination_coords, test_drone.coords, destination_id
        )
        
        if best_neighbor:
            print(f"\n选择的最佳下一跳: {best_neighbor}")
            return True
        else:
            print("\n未找到最佳下一跳")
            return False
    
    def test_next_hop_selection(self):
        """测试下一跳选择"""
        print("\n=== 下一跳选择测试 ===")
        
        # 设置邻居关系
        self.setup_neighbor_relationships()
        
        # 测试路由选择
        src_drone = self.drones[0]
        dst_drone = self.drones[-1]
        
        # 创建测试数据包
        packet = Packet(
            packet_id=f"test_packet_{src_drone.identifier}_{dst_drone.identifier}",
            packet_length=1024,
            creation_time=self.simulator.env.now,
            simulator=self.simulator,
            channel_id=0
        )
        # 手动设置源和目标信息用于测试
        packet.src_drone_id = src_drone.identifier
        packet.dst_drone_id = dst_drone.identifier
        packet.packet_type = "DATA"
        
        # 选择下一跳
        next_hop = src_drone.routing_protocol.next_hop_selection(packet)
        
        if next_hop is not None:
            print(f"路由选择成功: {src_drone.identifier} -> {dst_drone.identifier}")
            print(f"下一跳: {next_hop}")
            return True
        else:
            print(f"路由选择失败: {src_drone.identifier} -> {dst_drone.identifier}")
            return False
    
    def setup_neighbor_relationships(self):
        """设置邻居关系"""
        for i, drone in enumerate(self.drones):
            neighbor_table = drone.routing_protocol.neighbor_table
            
            # 为每个无人机添加通信范围内的邻居
            for j, other_drone in enumerate(self.drones):
                if i != j:
                    distance = euclidean_distance(drone.coords, other_drone.coords)
                    if distance <= drone.communication_range:
                        neighbor_table.add_or_update_neighbor(
                            neighbor_id=other_drone.identifier,
                            coords=other_drone.coords,
                            velocity=other_drone.velocity,
                            neighbor_count=random.randint(1, 5)
                        )
    
    def test_packet_processing(self):
        """测试数据包处理"""
        print("\n=== 数据包处理测试 ===")
        
        test_drone = self.drones[0]
        routing_protocol = test_drone.routing_protocol
        
        # 测试Hello包处理
        hello_packet = GbmcrHelloPacket(
            src_drone_id=self.drones[1].identifier,
            src_coords=self.drones[1].coords,
            src_velocity=self.drones[1].velocity,
            neighbor_count=3,
            timestamp=time.time(),
            simulator=self.simulator,
            channel_id=0
        )
        
        print("处理Hello包...")
        routing_protocol.handle_hello_packet(hello_packet)
        
        # 检查邻居是否被添加
        neighbor_count = routing_protocol.neighbor_table.get_neighbor_count()
        print(f"处理Hello包后，邻居数量: {neighbor_count}")
        
        # 测试数据包处理
        data_packet = Packet(
            packet_id=f"test_data_{self.drones[1].identifier}_{test_drone.identifier}",
            packet_length=1024,
            creation_time=self.simulator.env.now,
            simulator=self.simulator,
            channel_id=0
        )
        # 手动设置源和目标信息用于测试
        data_packet.src_drone_id = self.drones[1].identifier
        data_packet.dst_drone_id = test_drone.identifier
        data_packet.packet_type = "DATA"
        
        print("处理数据包...")
        routing_protocol.handle_data_packet(data_packet)
        
        return neighbor_count > 0
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("开始GBMCR路由协议综合测试")
        print("=" * 50)
        
        # 设置测试环境
        self.setup_test_environment(num_drones=10, area_size=500)
        
        # 运行各项测试
        test_results = {}
        
        try:
            # 1. 邻居表功能测试
            test_results['neighbor_table'] = self.test_neighbor_table()
            
            # 2. Hello包创建测试
            test_results['hello_packet'] = self.test_hello_packet_creation()
            
            # 3. 信标机制测试
            test_results['beacon_mechanism'] = self.test_beacon_mechanism()
            
            # 4. Q学习基本功能测试
            test_results['q_learning'] = self.test_q_learning_basic()
            
            # 5. 下一跳选择测试
            test_results['next_hop_selection'] = self.test_next_hop_selection()
            
            # 6. 数据包处理测试
            test_results['packet_processing'] = self.test_packet_processing()
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        # 输出测试总结
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, results):
        """打印测试总结"""
        print("\n" + "=" * 50)
        print("GBMCR路由协议测试总结")
        print("=" * 50)
        
        passed_tests = 0
        total_tests = len(results)
        
        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
            if result:
                passed_tests += 1
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        print(f"\n测试通过率: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("测试结果: 优秀 ✓")
        elif success_rate >= 0.6:
            print("测试结果: 良好 ✓")
        elif success_rate >= 0.4:
            print("测试结果: 一般 ⚠")
        else:
            print("测试结果: 需要改进 ✗")


def main():
    """主函数"""
    print("GBMCR路由协议测试程序")
    print("基于地理信标的智能协作路由策略测试")
    print()
    
    # 创建测试器
    tester = GbmcrTester()
    
    # 运行综合测试
    results = tester.run_comprehensive_test()
    
    print("\n测试完成！")
    
    return results


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 运行测试
    test_results = main()