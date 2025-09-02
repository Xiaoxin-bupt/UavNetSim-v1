#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量仿真结果分析脚本
用于分析和可视化批量仿真的结果数据

Author: Generated for UavNetSim-v1
Created: 2025
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置matplotlib后端为非交互式
import matplotlib
matplotlib.use('Agg')

# 设置高级配色方案和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ResultAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.data = None
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义高级配色方案
        self.colors = {
            'greedy': '#2E86AB',    # 深蓝色
            'qgeo': '#A23B72',      # 深紫红色
            'qmr': '#F18F01',       # 橙色
            'dsdv': '#C73E1D',      # 深红色
            'grad': '#6A994E',      # 绿色
            'opar': '#7209B7'       # 紫色
        }
        
    def load_results(self):
        """加载所有仿真结果"""
        results = []
        
        # 遍历结果目录中的所有JSON文件
        for json_file in self.results_dir.glob("*.json"):
            if json_file.name in ['experiment_config.json', 'experiment_summary.json']:
                continue
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"警告: 无法读取文件 {json_file}: {e}")
        
        if not results:
            raise ValueError(f"在目录 {self.results_dir} 中未找到有效的结果文件")
        
        self.data = pd.DataFrame(results)
        print(f"成功加载 {len(results)} 个仿真结果")
        
        return self.data
    
    def create_trend_plots(self):
        """创建性能指标趋势图表"""
        if self.data is None:
            self.load_results()
        
        # 定义要绘制的性能指标
        metrics = {
            'packet_delivery_ratio': 'Packet Delivery Ratio',
            'average_delay': 'Average Delay (ms)',
            'routing_load': 'Routing Load',
            'throughput': 'Throughput (Kbps)',
            'hop_count': 'Average Hop Count',
            'collision_num': 'Collision Number',
            'mac_delay': 'MAC Delay (ms)'
        }
        
        # 创建节点数量趋势图
        self._create_node_trend_plots(metrics)
        
        # 创建速度趋势图
        self._create_speed_trend_plots(metrics)
    
    def _create_node_trend_plots(self, metrics):
        """创建按节点数量变化的趋势图"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Performance Trends vs Number of Nodes', fontsize=18, fontweight='bold', y=0.95)
        
        axes_flat = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics.items()):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # 按节点数量和协议分组
            protocols = self.data['routing_protocol'].unique()
            
            for protocol in protocols:
                protocol_data = self.data[self.data['routing_protocol'] == protocol]
                grouped = protocol_data.groupby('num_nodes')[metric].mean().reset_index()
                
                color = self.colors.get(protocol.lower(), '#333333')
                
                ax.plot(grouped['num_nodes'], grouped[metric], 
                       marker='o', linewidth=2.5, markersize=7, 
                       label=protocol.upper(), color=color, 
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # 设置网格样式
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            
            ax.set_xlabel('Number of Nodes', fontsize=11, fontweight='medium')
            ax.set_ylabel(title, fontsize=11, fontweight='medium')
            ax.set_title(f'{title} vs Node Count', fontsize=13, fontweight='bold', pad=15)
            
            # 设置图例
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, 
                     framealpha=0.9, edgecolor='gray')
            
            # 设置x轴刻度
            nodes_unique = sorted(self.data['num_nodes'].unique())
            ax.set_xticks(nodes_unique)
            
            # 美化坐标轴
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
        
        # 隐藏多余的子图
        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = self.output_dir / "node_trends.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Node trends chart saved to: {plot_file}")
        
        plt.close(fig)
    
    def _create_speed_trend_plots(self, metrics):
        """创建按速度变化的趋势图"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Performance Trends vs UAV Speed', fontsize=18, fontweight='bold', y=0.95)
        
        axes_flat = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics.items()):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            
            # 按速度和协议分组
            protocols = self.data['routing_protocol'].unique()
            
            for protocol in protocols:
                protocol_data = self.data[self.data['routing_protocol'] == protocol]
                grouped = protocol_data.groupby('drone_speed')[metric].mean().reset_index()
                
                color = self.colors.get(protocol.lower(), '#333333')
                
                ax.plot(grouped['drone_speed'], grouped[metric], 
                       marker='s', linewidth=2.5, markersize=7, 
                       label=protocol.upper(), color=color,
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # 设置网格样式
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            
            ax.set_xlabel('UAV Speed (m/s)', fontsize=11, fontweight='medium')
            ax.set_ylabel(title, fontsize=11, fontweight='medium')
            ax.set_title(f'{title} vs Speed', fontsize=13, fontweight='bold', pad=15)
            
            # 设置图例
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, 
                     framealpha=0.9, edgecolor='gray')
            
            # 设置x轴刻度
            speeds_unique = sorted(self.data['drone_speed'].unique())
            ax.set_xticks(speeds_unique)
            
            # 美化坐标轴
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
        
        # 隐藏多余的子图
        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = self.output_dir / "speed_trends.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Speed trends chart saved to: {plot_file}")
        
        plt.close(fig)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析批量仿真结果并生成趋势图')
    parser.add_argument('results_dir', nargs='?', default='quick_experiment_results', 
                       help='仿真结果目录路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"错误: 目录 {args.results_dir} 不存在")
        return
    
    analyzer = ResultAnalyzer(args.results_dir)
    
    print("=== 开始生成性能趋势图表 ===")
    analyzer.create_trend_plots()
    print(f"\n=== 分析完成 ===")
    print(f"所有图表已保存到: {analyzer.output_dir}")

if __name__ == "__main__":
    main()