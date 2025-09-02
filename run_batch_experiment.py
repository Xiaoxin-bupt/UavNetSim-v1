#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量实验运行脚本
用于快速启动批量仿真实验

Author: Generated for UavNetSim-v1
Created: 2025
"""

import os
import sys
from pathlib import Path
from batch_simulation import BatchSimulation
from analyze_results import ResultAnalyzer

def run_quick_experiment(parallel=True, max_workers=None):
    """运行快速实验（较少参数组合）
    
    Args:
        parallel: 是否使用并行执行
        max_workers: 最大并行线程数
    """
    mode_str = "并行" if parallel else "串行"
    print(f"=== 运行快速批量实验（{mode_str}模式）===")
    
    # 定义快速实验参数
    protocols = ['QMR']  # available options: "Greedy", "Dsdv", "QMR", "QGeo", "Opar", "QRouting", "Grad"
    node_counts = [10,20,30,40]
    speeds = [20]
    seeds = [2025]
    
    batch_sim = BatchSimulation(
        protocols=protocols,
        node_counts=node_counts, 
        speeds=speeds,
        seeds=seeds,
        output_dir="quick_experiment_results",
        max_workers=max_workers
    )
    
    # 运行批量仿真
    batch_sim.run_batch_simulation(parallel=parallel)
    
    # 分析结果
    print("\n=== 分析实验结果 ===")
    analyzer = ResultAnalyzer(batch_sim.output_dir)
    analyzer.create_trend_plots()
    
    return batch_sim.output_dir

def main():
    """主函数"""
    print("UavNetSim 批量实验工具")
    print("=" * 50)
    
    print("\n请选择操作:")
    print("1. 运行快速实验 - 并行模式 (3个协议, 3个节点数, 3个速度)")
    print("2. 运行快速实验 - 串行模式 (3个协议, 3个节点数, 3个速度)")
    print("3. 退出")
    
    # choice = input("\n请输入选择 (1-3): ").strip()
    choice = '1'  # 默认选择并行模式
    
    try:
        if choice == '1':
            print("\n选择并行模式执行...")
            result_dir = run_quick_experiment(parallel=True, max_workers=20)
            print(f"\n快速实验完成! 结果保存在: {result_dir}")
        elif choice == '2':
            print("\n选择串行模式执行...")
            result_dir = run_quick_experiment(parallel=False)
            print(f"\n快速实验完成! 结果保存在: {result_dir}")
        elif choice == '3':
            print("再见!")
        else:
            print("无效选择")
    except Exception as e:
        print(f"\n错误: {e}")

if __name__ == "__main__":
    main()