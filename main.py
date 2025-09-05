import simpy
import argparse
import json
import os
from pathlib import Path
from utils import config
from simulator.simulator import Simulator
from visualization.visualizer import SimulationVisualizer

# """
#   _   _                   _   _          _     ____    _             
#  | | | |   __ _  __   __ | \ | |   ___  | |_  / ___|  (_)  _ __ ___  
#  | | | |  / _` | \ \ / / |  \| |  / _ \ | __| \___ \  | | | '_ ` _ \ 
#  | |_| | | (_| |  \ V /  | |\  | |  __/ | |_   ___) | | | | | | | | |
#   \___/   \__,_|   \_/   |_| \_|  \___|  \__| |____/  |_| |_| |_| |_|
                                                                                                                                                                                                                                                                                           
# """

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UavNetSim 仿真程序')
    parser.add_argument('--protocol', type=str, default=config.ROUTING_PROTOCOL,
                       choices=['GBMCR', 'QGeo', 'Greedy', 'Dsdv', 'Grad', 'QRouting', 'QMR', 'Opar'],
                       help='路由协议')
    parser.add_argument('--nodes', type=int, default=config.NUMBER_OF_DRONES,
                       help='无人机节点数量')
    parser.add_argument('--speed', type=int, default=config.DEFAULT_DRONE_SPEED,
                       help='无人机速度 (m/s)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='随机种子')
    parser.add_argument('--output', type=str, default="results/" + "test.json",
                       help='结果输出文件路径')
    parser.add_argument('--sim-time', type=int, default=config.SIM_TIME,
                       help='仿真时间 (微秒)')
    return parser.parse_args()

def run_simulation(protocol, num_nodes, drone_speed, seed, sim_time, output_file=None):
    """运行仿真并收集结果"""
    # 备份原始配置
    original_protocol = config.ROUTING_PROTOCOL
    original_nodes = config.NUMBER_OF_DRONES
    original_speed = config.DEFAULT_DRONE_SPEED
    original_sim_time = config.SIM_TIME
    
    try:
        # 临时修改配置
        config.ROUTING_PROTOCOL = protocol
        config.NUMBER_OF_DRONES = num_nodes
        config.DEFAULT_DRONE_SPEED = drone_speed
        config.SIM_TIME = sim_time
        
        # 仿真设置
        env = simpy.Environment()
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(num_nodes)}
        sim = Simulator(seed=seed, env=env, channel_states=channel_states, n_drones=num_nodes)
        
        # 运行仿真
        env.run(until=sim_time)
        
        # 收集结果
        metrics = sim.metrics
        import numpy as np
        
        # 计算各项指标
        pdr = len(metrics.datapacket_arrived) / metrics.datapacket_generated_num * 100 if metrics.datapacket_generated_num > 0 else 0
        avg_delay = np.mean(list(metrics.deliver_time_dict.values())) / 1e3 if metrics.deliver_time_dict else 0  # 转换为ms
        avg_throughput = np.mean(list(metrics.throughput_dict.values())) / 1e3 if metrics.throughput_dict else 0
        avg_hop_count = np.mean(list(metrics.hop_cnt_dict.values())) if metrics.hop_cnt_dict else 0
        avg_mac_delay = np.mean(metrics.mac_delay) / 1e3 if metrics.mac_delay else 0  # 转换为ms
        
        # 计算路由负载
        routing_load = metrics.control_packet_num / len(metrics.datapacket_arrived) if len(metrics.datapacket_arrived) > 0 else 0
        
        results = {
            'routing_protocol': protocol,
            'num_nodes': num_nodes,
            'drone_speed': drone_speed,
            'seed': seed,
            'sim_time': sim_time,
            'datapacket_generated_num': metrics.datapacket_generated_num,
            'packet_delivery_ratio': pdr,
            'average_delay': avg_delay,
            'routing_load': routing_load,
            'throughput': avg_throughput,
            'hop_count': avg_hop_count,
            'collision_num': metrics.collision_num,
            'mac_delay': avg_mac_delay
        }
        
        # 保存结果
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"结果已保存到: {output_file}")
        
        return results
        
    finally:
        # 恢复原始配置
        config.ROUTING_PROTOCOL = original_protocol
        config.NUMBER_OF_DRONES = original_nodes
        config.DEFAULT_DRONE_SPEED = original_speed
        config.SIM_TIME = original_sim_time

if __name__ == "__main__":
    args = parse_arguments()
    
    # 运行仿真
    results = run_simulation(
        protocol=args.protocol,
        num_nodes=args.nodes,
        drone_speed=args.speed,
        seed=args.seed,
        sim_time=args.sim_time,
        output_file=args.output
    )
    
    # 打印结果摘要
    print(f"\n=== 仿真结果摘要 ===")
    print(f"协议: {results['routing_protocol']}")
    print(f"节点数: {results['num_nodes']}")
    print(f"速度: {results['drone_speed']} m/s")
    print(f"PDR: {results['packet_delivery_ratio']:.3f}")
    print(f"平均延迟: {results['average_delay']:.2f} 微秒")