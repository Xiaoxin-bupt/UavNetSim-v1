#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量仿真控制器
用于自动运行多种参数组合的仿真实验

Author: Generated for UavNetSim-v1
Created: 2025
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils import config

class BatchSimulation:
    def __init__(self, protocols=None, node_counts=None, speeds=None, seeds=None, output_dir=None, sim_time=None, max_workers=None):
        """
        初始化批量仿真控制器
        
        Args:
            protocols: 路由协议列表
            node_counts: 节点数量列表
            speeds: 无人机速度列表
            seeds: 随机种子列表
            output_dir: 输出目录名称
        """
        # 默认实验参数
        self.protocols = protocols or ['QGeo', 'Greedy', 'QMR']  # available options: "Greedy", "Dsdv", "QMR", "QGeo", "Opar"
        self.node_counts = node_counts or [15, 20, 25, 30]
        self.speeds = speeds or [10, 15, 20, 25, 30]
        self.seeds = seeds or [2025]
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"results/batch_simulation_{timestamp}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 仿真时间设置
        self.sim_time = sim_time or config.SIM_TIME
        
        # 并行执行设置
        self.max_workers = max_workers or min(4, os.cpu_count())  # 默认使用4个线程或CPU核心数
        self.lock = threading.Lock()  # 用于线程安全的计数器更新
        
        # 实验统计
        self.total_experiments = len(self.protocols) * len(self.node_counts) * len(self.speeds) * len(self.seeds)
        self.completed_experiments = 0
        self.failed_experiments = 0
        
        print(f"批量仿真初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"总实验数: {self.total_experiments}")
        print(f"并行线程数: {self.max_workers}")
    
    def save_experiment_config(self):
        """保存实验配置"""
        config = {
            'protocols': self.protocols,
            'node_counts': self.node_counts,
            'speeds': self.speeds,
            'seeds': self.seeds,
            'total_experiments': self.total_experiments,
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = self.output_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"实验配置已保存到: {config_file}")
    
    def run_single_simulation(self, protocol, num_nodes, speed, seed):
        """运行单次仿真"""
        output_file = f"{protocol}_{num_nodes}_{speed}_{seed}.json"
        output_path = self.output_dir / output_file
        
        # 构建命令
        cmd = [
            'python', 'main.py',
            '--protocol', protocol,
            '--nodes', str(num_nodes),
            '--speed', str(speed),
            '--seed', str(seed),
            '--output', str(output_path)
        ]
        
        print(f"运行仿真: {protocol}, 节点={num_nodes}, 速度={speed}, 种子={seed}")
        
        try:
            # 运行仿真
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✓ 仿真成功: {output_file}")
                with self.lock:
                    self.completed_experiments += 1
                return True
            else:
                print(f"  ✗ 仿真失败: {output_file}")
                print(f"    错误: {result.stderr}")
                with self.lock:
                    self.failed_experiments += 1
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 仿真超时: {output_file}")
            with self.lock:
                self.failed_experiments += 1
            return False
        except Exception as e:
            print(f"  ✗ 仿真异常: {output_file}, 错误: {e}")
            with self.lock:
                self.failed_experiments += 1
            return False
    
    def run_batch_simulation(self, parallel=True):
        """运行批量仿真
        
        Args:
            parallel: 是否使用并行执行，默认为True
        """
        if parallel:
            self._run_parallel_simulation()
        else:
            self._run_serial_simulation()
    
    def _run_serial_simulation(self):
        """串行运行批量仿真（原有方法）"""
        print("\n=== 开始批量仿真（串行模式）===")
        
        # 保存实验配置
        self.save_experiment_config()
        
        start_time = time.time()
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(
            self.protocols, self.node_counts, self.speeds, self.seeds
        ))
        
        # 运行所有实验
        for i, (protocol, nodes, speed, seed) in enumerate(param_combinations, 1):
            print(f"\n进度: {i}/{self.total_experiments}")
            self.run_single_simulation(protocol, nodes, speed, seed)
            
            # 显示进度
            progress = (i / self.total_experiments) * 100
            print(f"完成进度: {progress:.1f}%")
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 保存实验总结
        self.save_experiment_summary(total_time)
        
        print(f"\n=== 批量仿真完成 ===")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"成功: {self.completed_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"结果保存在: {self.output_dir}")
    
    def _run_parallel_simulation(self):
        """并行运行批量仿真"""
        print(f"\n=== 开始批量仿真（并行模式，{self.max_workers}线程）===")
        
        # 保存实验配置
        self.save_experiment_config()
        
        start_time = time.time()
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(
            self.protocols, self.node_counts, self.speeds, self.seeds
        ))
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_params = {
                executor.submit(self.run_single_simulation, protocol, nodes, speed, seed): 
                (protocol, nodes, speed, seed)
                for protocol, nodes, speed, seed in param_combinations
            }
            
            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_params):
                completed_count += 1
                progress = (completed_count / self.total_experiments) * 100
                print(f"\r完成进度: {progress:.1f}% ({completed_count}/{self.total_experiments})", end="", flush=True)
        
        print()  # 换行
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 保存实验总结
        self.save_experiment_summary(total_time)
        
        print(f"\n=== 批量仿真完成 ===")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"成功: {self.completed_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"结果保存在: {self.output_dir}")
    
    def save_experiment_summary(self, total_time):
        """保存实验总结"""
        summary = {
            'total_experiments': self.total_experiments,
            'completed_experiments': self.completed_experiments,
            'failed_experiments': self.failed_experiments,
            'success_rate': (self.completed_experiments / self.total_experiments) * 100,
            'total_time_seconds': total_time,
            'average_time_per_experiment': total_time / self.total_experiments,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"实验总结已保存到: {summary_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量仿真控制器')
    parser.add_argument('--protocols', nargs='+', default=['QGeo', 'Greedy'], 
                       help='路由协议列表')
    parser.add_argument('--nodes', nargs='+', type=int, default=[10, 15, 20], 
                       help='节点数量列表')
    parser.add_argument('--speeds', nargs='+', type=int, default=[15, 25], 
                       help='速度列表')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2025, 2026], 
                       help='随机种子列表')
    parser.add_argument('--output', default=None, help='输出目录')
    parser.add_argument('--parallel', action='store_true', default=True, 
                       help='使用并行执行（默认启用）')
    parser.add_argument('--serial', action='store_true', 
                       help='使用串行执行（覆盖--parallel）')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='最大并行线程数（默认为min(4, CPU核心数)）')
    parser.add_argument('--sim-time', type=int, default=None, 
                       help='仿真时间（秒）')
    
    args = parser.parse_args()
    
    # 确定是否使用并行执行
    use_parallel = args.parallel and not args.serial
    
    # 创建批量仿真实例
    batch_sim = BatchSimulation(
        protocols=args.protocols,
        node_counts=args.nodes,
        speeds=args.speeds,
        seeds=args.seeds,
        output_dir=args.output,
        sim_time=args.sim_time,
        max_workers=args.max_workers
    )
    
    # 运行批量仿真
    batch_sim.run_batch_simulation(parallel=use_parallel)

if __name__ == "__main__":
    main()