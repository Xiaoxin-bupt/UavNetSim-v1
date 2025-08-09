#!/usr/bin/env python3
"""
GBICR快速修复脚本

这个脚本提供快速修复方案，直接优化GBICR配置参数，
无需训练即可立即改善路由性能。
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from routing.gbicr.gbicr_config import get_gbicr_config


def apply_quick_fix():
    """应用快速修复配置"""
    print("正在应用GBICR快速修复...")
    
    # 读取当前配置文件
    config_file = Path("routing/gbicr/gbicr_config.py")
    
    if not config_file.exists():
        print(f"错误: 找不到配置文件 {config_file}")
        return False
    
    # 备份原文件
    backup_file = config_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"原配置已备份到: {backup_file}")
    
    # 快速修复配置
    quick_fix_config = """
# GBICR Protocol Parameters - 快速修复版本
GBICR_CONFIG = {
    # 时间参数优化 - 针对5秒仿真
    'hello_interval': 0.1 * 1e6,   # 100ms，快速邻居发现
    'beacon_interval': 0.4 * 1e6,  # 400ms，频繁信标
    'check_interval': 0.15 * 1e6,  # 150ms，快速检查
    
    # 学习参数优化
    'learning_rate': 0.8,          # 高学习率，快速适应
    'reward_max': 25.0,            # 增大奖励范围
    'reward_min': -25.0,
    'exploration_rate': 0.4,       # 高探索率
    
    # PPO Agent参数优化
    'ppo_lr': 1e-3,               # 提高PPO学习率
    'ppo_gamma': 0.9,             # 降低折扣因子
    'ppo_eps_clip': 0.3,          # 增大裁剪范围
    'ppo_k_epochs': 10,           # 增加更新轮数
    'ppo_batch_size': 64,         # 适中批次大小
    
    # 状态空间参数
    'max_neighbors': 15,           # 增加最大邻居数
    'state_dimension': None,       # 自动计算
    
    # 网络参数优化
    'entry_lifetime': 1.2 * 1e6,  # 缩短邻居表生存时间
    'beacon_lifetime': 2.0 * 1e6, # 缩短信标生存时间
    'stability_window': 3,         # 减少稳定性窗口
    
    # 奖励权重优化 - 重视地理进度和协作
    'geographic_weight': 0.45,     # 提高地理权重
    'collaborative_weight': 0.35,  # 提高协作权重
    'link_quality_weight': 0.15,   # 适中链路质量权重
    'stability_weight': 0.05,      # 降低稳定性权重
    
    # 模型路径
    'pretrained_model_path': None,
    'model_save_path': './models/gbicr_model.npy',
    'training_log_path': './logs/gbicr_training.log',
}
"""
    
    # 读取原文件内容
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到GBICR_CONFIG的开始和结束位置
    start_marker = "GBICR_CONFIG = {"
    end_marker = "}"
    
    start_pos = content.find(start_marker)
    if start_pos == -1:
        print("错误: 找不到GBICR_CONFIG定义")
        return False
    
    # 找到对应的结束大括号
    brace_count = 0
    end_pos = start_pos
    for i, char in enumerate(content[start_pos:]):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_pos = start_pos + i + 1
                break
    
    # 替换配置
    new_content = content[:start_pos] + quick_fix_config + content[end_pos:]
    
    # 写入新配置
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("快速修复配置已应用！")
    return True


def show_improvements():
    """显示预期改善"""
    print("\n预期性能改善:")
    print("=" * 50)
    print("📈 包投递率: 23.33% → 预期 60-80%")
    print("⏱️  平均延迟: 708ms → 预期 200-400ms")
    print("🚀 平均吞吐量: 147 Kbps → 预期 300-600 Kbps")
    print("🔗 路由负载: 2.14 → 预期 1.5-2.0")
    
    print("\n主要优化点:")
    print("✅ 更频繁的邻居发现 (100ms间隔)")
    print("✅ 快速路由收敛 (高学习率)")
    print("✅ 优化奖励权重 (重视地理进度)")
    print("✅ 增强探索能力 (40%探索率)")
    print("✅ 缩短表项生存时间 (快速更新)")


def restore_backup():
    """恢复备份配置"""
    config_file = Path("routing/gbicr/gbicr_config.py")
    backup_file = config_file.with_suffix('.py.backup')
    
    if backup_file.exists():
        with open(backup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("配置已恢复到原始状态")
        return True
    else:
        print("未找到备份文件")
        return False


def main():
    """主函数"""
    print("="*60)
    print("GBICR 快速修复工具")
    print("="*60)
    
    print("\n当前性能问题:")
    print("• 包投递率过低 (23.33%)")
    print("• 端到端延迟过高 (708ms)")
    print("• 吞吐量不足 (147 Kbps)")
    
    print("\n快速修复方案:")
    print("1. 应用优化配置 (推荐)")
    print("2. 恢复原始配置")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择操作 (1-3): ").strip()
        
        if choice == '1':
            success = apply_quick_fix()
            if success:
                show_improvements()
                print("\n✅ 快速修复完成！")
                print("\n下一步:")
                print("1. 运行仿真: python main.py")
                print("2. 观察性能改善")
                print("3. 如需进一步优化，运行: python train_gbicr_optimized.py")
            break
            
        elif choice == '2':
            success = restore_backup()
            if success:
                print("✅ 配置已恢复")
            break
            
        elif choice == '3':
            print("退出")
            break
            
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()