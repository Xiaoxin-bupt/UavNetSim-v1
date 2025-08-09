#!/usr/bin/env python3
"""
GBICR配置优化脚本

这个脚本帮助用户快速优化GBICR配置参数，以改善路由性能。
包含针对不同场景的预设配置和自动调优功能。
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from routing.gbicr.gbicr_config import get_gbicr_config, GBICR_CONFIG
from utils import config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GbicrConfigOptimizer:
    """GBICR配置优化器"""
    
    def __init__(self):
        self.base_config = get_gbicr_config()
        self.sim_config = self._get_simulation_config()
        
    def _get_simulation_config(self):
        """获取当前仿真配置"""
        return {
            'map_length': config.MAP_LENGTH,
            'map_width': config.MAP_WIDTH, 
            'map_height': config.MAP_HEIGHT,
            'sim_time': config.SIM_TIME,
            'num_drones': config.NUMBER_OF_DRONES,
            'transmitting_power': config.TRANSMITTING_POWER,
            'sensing_range': config.SENSING_RANGE,
        }
    
    def analyze_current_performance(self):
        """分析当前性能问题"""
        print("\n当前性能分析:")
        print("=" * 50)
        print(f"包投递率: 23.33% (目标: >80%)")
        print(f"平均端到端延迟: 708.30ms (目标: <200ms)")
        print(f"平均吞吐量: 147.29 Kbps (目标: >500 Kbps)")
        print(f"平均跳数: 2.82 (合理范围: 2-4)")
        
        print("\n问题诊断:")
        print("1. 包投递率过低 - 可能原因:")
        print("   - 路由决策不优")
        print("   - 邻居发现不及时")
        print("   - 链路质量评估不准确")
        
        print("2. 延迟过高 - 可能原因:")
        print("   - 路由收敛慢")
        print("   - 队列拥塞")
        print("   - 重传次数多")
        
        print("3. 吞吐量低 - 可能原因:")
        print("   - 路由效率低")
        print("   - 网络利用率不足")
        print("   - 协议开销大")
    
    def get_optimized_config_for_scenario(self, scenario: str) -> Dict[str, Any]:
        """根据场景获取优化配置"""
        configs = {
            'current_sim': self._get_current_sim_optimized_config(),
            'high_density': self._get_high_density_config(),
            'high_mobility': self._get_high_mobility_config(),
            'low_latency': self._get_low_latency_config(),
            'energy_efficient': self._get_energy_efficient_config(),
            'robust': self._get_robust_config()
        }
        
        return configs.get(scenario, self.base_config)
    
    def _get_current_sim_optimized_config(self):
        """针对当前仿真环境的优化配置"""
        config = self.base_config.copy()
        
        # 根据仿真时间调整间隔
        sim_time_sec = self.sim_config['sim_time'] / 1e6  # 转换为秒
        
        config.update({
            # 时间参数优化 - 针对5秒仿真时间
            'hello_interval': 0.15 * 1e6,      # 150ms，更频繁邻居发现
            'beacon_interval': 0.6 * 1e6,      # 600ms，适中信标频率
            'check_interval': 0.2 * 1e6,       # 200ms，快速检查
            
            # 学习参数优化
            'learning_rate': 0.6,              # 提高学习率，快速适应
            'exploration_rate': 0.3,           # 增加探索，发现更好路径
            'reward_max': 20.0,                # 增大奖励范围
            'reward_min': -20.0,
            
            # PPO参数优化
            'ppo_lr': 8e-4,                   # 提高学习率
            'ppo_gamma': 0.92,                # 降低折扣因子，重视即时奖励
            'ppo_eps_clip': 0.25,             # 适中裁剪
            'ppo_k_epochs': 8,                # 增加更新轮数
            'ppo_batch_size': 128,            # 增大批次
            
            # 网络参数优化
            'max_neighbors': 12,               # 适中邻居数
            'entry_lifetime': 1.5 * 1e6,      # 缩短生存时间，快速更新
            'beacon_lifetime': 2.5 * 1e6,
            'stability_window': 5,             # 减少稳定性窗口
            
            # 奖励权重优化 - 重视地理进度和协作
            'geographic_weight': 0.4,          # 提高地理权重
            'collaborative_weight': 0.35,      # 提高协作权重
            'link_quality_weight': 0.15,       # 降低链路质量权重
            'stability_weight': 0.1,           # 保持稳定性权重
        })
        
        return config
    
    def _get_high_density_config(self):
        """高密度网络配置"""
        config = self.base_config.copy()
        config.update({
            'hello_interval': 0.3 * 1e6,
            'beacon_interval': 1.0 * 1e6,
            'max_neighbors': 20,
            'collaborative_weight': 0.5,
            'geographic_weight': 0.25,
            'link_quality_weight': 0.15,
            'stability_weight': 0.1,
            'exploration_rate': 0.15,
        })
        return config
    
    def _get_high_mobility_config(self):
        """高移动性配置"""
        config = self.base_config.copy()
        config.update({
            'hello_interval': 0.1 * 1e6,
            'beacon_interval': 0.4 * 1e6,
            'entry_lifetime': 1.0 * 1e6,
            'beacon_lifetime': 1.5 * 1e6,
            'stability_weight': 0.2,
            'geographic_weight': 0.4,
            'exploration_rate': 0.25,
            'learning_rate': 0.7,
        })
        return config
    
    def _get_low_latency_config(self):
        """低延迟配置"""
        config = self.base_config.copy()
        config.update({
            'hello_interval': 0.1 * 1e6,
            'beacon_interval': 0.3 * 1e6,
            'check_interval': 0.1 * 1e6,
            'geographic_weight': 0.5,
            'collaborative_weight': 0.25,
            'link_quality_weight': 0.2,
            'stability_weight': 0.05,
            'ppo_lr': 1e-3,
            'learning_rate': 0.8,
        })
        return config
    
    def _get_energy_efficient_config(self):
        """节能配置"""
        config = self.base_config.copy()
        config.update({
            'hello_interval': 0.8 * 1e6,
            'beacon_interval': 2.5 * 1e6,
            'check_interval': 1.0 * 1e6,
            'max_neighbors': 8,
            'exploration_rate': 0.1,
            'collaborative_weight': 0.4,
            'geographic_weight': 0.35,
        })
        return config
    
    def _get_robust_config(self):
        """鲁棒性配置"""
        config = self.base_config.copy()
        config.update({
            'hello_interval': 0.4 * 1e6,
            'beacon_interval': 1.2 * 1e6,
            'entry_lifetime': 4.0 * 1e6,
            'beacon_lifetime': 6.0 * 1e6,
            'stability_weight': 0.25,
            'link_quality_weight': 0.3,
            'exploration_rate': 0.2,
            'max_neighbors': 15,
        })
        return config
    
    def apply_config(self, config: Dict[str, Any], config_name: str = "optimized"):
        """应用配置到GBICR配置文件"""
        config_file_path = Path("routing/gbicr/gbicr_config.py")
        
        if not config_file_path.exists():
            logger.error(f"配置文件不存在: {config_file_path}")
            return False
        
        try:
            # 读取原配置文件
            with open(config_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 备份原文件
            backup_path = config_file_path.with_suffix('.py.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 更新GBICR_CONFIG字典
            new_content = self._update_config_content(content, config)
            
            # 写入新配置
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"配置已应用: {config_name}")
            logger.info(f"原配置已备份到: {backup_path}")
            
            # 保存配置到JSON文件
            json_path = Path(f"gbicr_config_{config_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"应用配置失败: {e}")
            return False
    
    def _update_config_content(self, content: str, new_config: Dict[str, Any]) -> str:
        """更新配置文件内容"""
        lines = content.split('\n')
        new_lines = []
        in_gbicr_config = False
        brace_count = 0
        
        for line in lines:
            if 'GBICR_CONFIG = {' in line:
                in_gbicr_config = True
                new_lines.append(line)
                brace_count = 1
                
                # 添加新配置
                for key, value in new_config.items():
                    if isinstance(value, str):
                        new_lines.append(f"    '{key}': '{value}',")
                    else:
                        new_lines.append(f"    '{key}': {value},")
                
            elif in_gbicr_config:
                # 计算大括号
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                if brace_count == 0:
                    in_gbicr_config = False
                    new_lines.append(line)
                # 跳过原配置行
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any], 
                       name1: str = "配置1", name2: str = "配置2"):
        """比较两个配置"""
        print(f"\n{name1} vs {name2} 配置对比:")
        print("=" * 60)
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in sorted(all_keys):
            val1 = config1.get(key, "未设置")
            val2 = config2.get(key, "未设置")
            
            if val1 != val2:
                print(f"{key:25} | {str(val1):15} | {str(val2):15}")
    
    def recommend_config(self):
        """推荐配置"""
        print("\n配置推荐:")
        print("=" * 50)
        
        # 分析当前仿真特点
        sim_time_sec = self.sim_config['sim_time'] / 1e6
        num_drones = self.sim_config['num_drones']
        map_area = self.sim_config['map_length'] * self.sim_config['map_width']
        density = num_drones / (map_area / 1e6)  # 每平方公里无人机数
        
        print(f"仿真特点分析:")
        print(f"  仿真时间: {sim_time_sec}秒 (短时仿真)")
        print(f"  无人机数量: {num_drones}")
        print(f"  网络密度: {density:.2f} 无人机/km²")
        
        if sim_time_sec <= 10:
            recommended = "current_sim"
            reason = "短时仿真，需要快速收敛"
        elif density > 50:
            recommended = "high_density"
            reason = "高密度网络，需要协作路由"
        elif density < 10:
            recommended = "robust"
            reason = "稀疏网络，需要鲁棒路由"
        else:
            recommended = "current_sim"
            reason = "中等密度，使用优化配置"
        
        print(f"\n推荐配置: {recommended}")
        print(f"推荐理由: {reason}")
        
        return recommended


def main():
    """主函数"""
    print("="*60)
    print("GBICR 配置优化工具")
    print("="*60)
    
    optimizer = GbicrConfigOptimizer()
    
    # 分析当前性能
    optimizer.analyze_current_performance()
    
    # 获取推荐配置
    recommended = optimizer.recommend_config()
    
    # 显示可用配置
    print("\n可用配置选项:")
    print("1. current_sim - 针对当前仿真优化")
    print("2. high_density - 高密度网络")
    print("3. high_mobility - 高移动性")
    print("4. low_latency - 低延迟")
    print("5. energy_efficient - 节能")
    print("6. robust - 鲁棒性")
    
    # 用户选择
    while True:
        choice = input(f"\n请选择配置 (1-6, 回车使用推荐配置 '{recommended}'): ").strip()
        
        config_map = {
            '1': 'current_sim',
            '2': 'high_density', 
            '3': 'high_mobility',
            '4': 'low_latency',
            '5': 'energy_efficient',
            '6': 'robust',
            '': recommended
        }
        
        if choice in config_map:
            selected_config = config_map[choice]
            break
        else:
            print("无效选择，请重新输入")
    
    # 获取选择的配置
    optimized_config = optimizer.get_optimized_config_for_scenario(selected_config)
    
    # 显示配置对比
    optimizer.compare_configs(
        optimizer.base_config, 
        optimized_config,
        "原始配置", 
        f"优化配置({selected_config})"
    )
    
    # 确认应用
    apply = input("\n是否应用此配置? (y/N): ").strip().lower()
    
    if apply == 'y':
        success = optimizer.apply_config(optimized_config, selected_config)
        
        if success:
            print("\n配置应用成功！")
            print("\n后续步骤:")
            print("1. 运行训练脚本: python train_gbicr_optimized.py")
            print("2. 或直接运行仿真: python main.py")
            print("3. 观察性能改善情况")
        else:
            print("\n配置应用失败，请检查错误信息")
    else:
        print("\n配置未应用")
        
        # 保存配置到文件
        json_path = f"gbicr_config_{selected_config}_preview.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {json_path}")


if __name__ == "__main__":
    main()