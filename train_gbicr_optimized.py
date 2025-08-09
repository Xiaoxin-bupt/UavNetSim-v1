#!/usr/bin/env python3
"""
优化的GBICR训练脚本

这个脚本提供了针对当前仿真环境优化的GBICR训练方案，
旨在改善路由性能，提高包投递率，降低延迟。
"""

import os
import sys
import time
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from routing.gbicr.train_gbicr import GbicrTrainer, GbicrTrainingEnvironment
from routing.gbicr.gbicr_config import get_gbicr_config, get_training_config
from utils import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedGbicrTrainer:
    """优化的GBICR训练器"""
    
    def __init__(self):
        self.setup_optimized_config()
        self.training_history = {
            'rewards': [],
            'success_rates': [],
            'packet_delivery_ratios': [],
            'avg_delays': []
        }
    
    def setup_optimized_config(self):
        """设置针对当前环境优化的配置"""
        # 基础配置
        self.gbicr_config = get_gbicr_config()
        self.training_config = get_training_config()
        
        # 针对当前仿真环境的优化配置
        optimized_gbicr_config = {
            # 调整时间间隔以适应5秒仿真时间
            'hello_interval': 0.2 * 1e6,  # 200ms，更频繁的邻居发现
            'beacon_interval': 0.8 * 1e6,  # 800ms，适中的信标频率
            'check_interval': 0.3 * 1e6,   # 300ms，更频繁的检查
            
            # 优化学习参数
            'learning_rate': 0.5,          # 提高学习率
            'exploration_rate': 0.25,      # 增加探索率
            'reward_max': 15.0,            # 增大奖励范围
            'reward_min': -15.0,
            
            # 优化PPO参数
            'ppo_lr': 5e-4,               # 提高PPO学习率
            'ppo_gamma': 0.95,            # 调整折扣因子
            'ppo_eps_clip': 0.3,          # 增大裁剪范围
            'ppo_k_epochs': 6,            # 增加更新轮数
            'ppo_batch_size': 64,         # 增大批次大小
            
            # 调整网络参数
            'max_neighbors': 15,           # 增加最大邻居数
            'entry_lifetime': 2 * 1e6,    # 缩短邻居表生存时间
            'beacon_lifetime': 3 * 1e6,   # 缩短信标生存时间
            
            # 优化奖励权重
            'geographic_weight': 0.35,     # 地理进度权重
            'collaborative_weight': 0.35,  # 协作权重
            'link_quality_weight': 0.20,   # 链路质量权重
            'stability_weight': 0.10,      # 稳定性权重
        }
        
        # 优化训练配置
        optimized_training_config = {
            'training_episodes': 2000,      # 增加训练轮数
            'evaluation_episodes': 50,     # 评估轮数
            'save_interval': 100,          # 保存间隔
            'log_interval': 20,            # 日志间隔
            'early_stopping_patience': 200, # 早停耐心
            'target_success_rate': 0.80,   # 目标成功率
        }
        
        # 更新配置
        self.gbicr_config.update(optimized_gbicr_config)
        self.training_config.update(optimized_training_config)
        
        logger.info("已设置优化配置")
        logger.info(f"Hello间隔: {self.gbicr_config['hello_interval']/1e6:.1f}s")
        logger.info(f"Beacon间隔: {self.gbicr_config['beacon_interval']/1e6:.1f}s")
        logger.info(f"学习率: {self.gbicr_config['learning_rate']}")
        logger.info(f"探索率: {self.gbicr_config['exploration_rate']}")
    
    def create_training_environment(self):
        """创建适合当前仿真的训练环境"""
        env_config = {
            'map_size': (config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT),
            'num_drones': config.NUMBER_OF_DRONES,
            'simulation_time': config.SIM_TIME,
            'mobility_model': 'random_waypoint',  # 使用随机路点模型
            'traffic_pattern': 'uniform',         # 均匀流量模式
        }
        
        return GbicrTrainingEnvironment(env_config)
    
    def train_progressive(self):
        """渐进式训练策略"""
        logger.info("开始渐进式训练")
        
        # 阶段1：简单环境训练（少量无人机）
        logger.info("阶段1：简单环境训练")
        simple_config = self.gbicr_config.copy()
        simple_env_config = {
            'map_size': (400, 400, 80),
            'num_drones': [5],  # 从少量无人机开始
            'mobility_models': ['random_walk'],
            'traffic_patterns': ['uniform'],
            'simulation_time': config.SIM_TIME,
        }
        
        trainer1 = GbicrTrainer(simple_config)
        env1 = GbicrTrainingEnvironment(simple_env_config)
        
        # 训练500轮
        rewards1 = self._train_phase(trainer1, env1, 500, "阶段1")
        
        # 阶段2：中等复杂度环境
        logger.info("阶段2：中等复杂度环境")
        medium_env_config = {
            'map_size': (500, 500, 90),
            'num_drones': [8],
            'mobility_models': ['random_waypoint'],
            'traffic_patterns': ['uniform'],
            'simulation_time': config.SIM_TIME,
        }
        
        trainer2 = GbicrTrainer(self.gbicr_config)
        # 继承前一阶段的模型
        if hasattr(trainer1, 'agent'):
            trainer2.agent = trainer1.agent
        
        env2 = GbicrTrainingEnvironment(medium_env_config)
        rewards2 = self._train_phase(trainer2, env2, 800, "阶段2")
        
        # 阶段3：完整环境训练
        logger.info("阶段3：完整环境训练")
        full_env_config = {
            'map_size': (config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT),
            'num_drones': [config.NUMBER_OF_DRONES],
            'mobility_models': ['random_waypoint'],
            'traffic_patterns': ['uniform'],
            'simulation_time': config.SIM_TIME,
        }
        
        trainer3 = GbicrTrainer(self.gbicr_config)
        # 继承前一阶段的模型
        if hasattr(trainer2, 'agent'):
            trainer3.agent = trainer2.agent
        
        env3 = GbicrTrainingEnvironment(full_env_config)
        rewards3 = self._train_phase(trainer3, env3, 1000, "阶段3")
        
        # 保存最终模型
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "gbicr_optimized_model.npy"
        
        if hasattr(trainer3, 'agent'):
            trainer3.agent.save_model(str(model_path))
            logger.info(f"模型已保存到: {model_path}")
        
        # 合并训练历史
        all_rewards = rewards1 + rewards2 + rewards3
        self.training_history['rewards'] = all_rewards
        
        return trainer3, all_rewards
    
    def _train_phase(self, trainer, env, episodes, phase_name):
        """训练一个阶段"""
        logger.info(f"{phase_name}: 开始训练 {episodes} 轮")
        
        rewards = []
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(episodes):
            # 重置环境
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作
                # 为简化训练，使用随机可用动作
                available_actions = list(range(min(10, len(state))))  # 简化的可用动作
                if not available_actions:
                    available_actions = [0]
                action, _ = trainer.agent.select_action(state, available_actions)
                
                # 执行动作
                # 为简化训练，使用随机的无人机ID
                num_drones = env.num_drones
                current_drone_id = random.randint(0, num_drones - 1)
                destination_id = random.randint(0, num_drones - 1)
                while destination_id == current_drone_id and num_drones > 1:
                    destination_id = random.randint(0, num_drones - 1)
                
                next_state, reward, done, info = env.step(action, current_drone_id, destination_id)
                
                # 存储经验
                action_prob = 0.1  # 简化的动作概率
                trainer.agent.store_transition(state, action, reward, next_state, done, action_prob)
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
            
            # 更新智能体
            if episode % 10 == 0:
                trainer.agent.update()
            
            # 记录和评估
            if episode % 50 == 0:
                avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
                logger.info(f"{phase_name} - Episode {episode}: 平均奖励 = {avg_reward:.3f}")
                
                # 早停检查
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter > 100:  # 早停
                    logger.info(f"{phase_name}: 早停于第 {episode} 轮")
                    break
        
        logger.info(f"{phase_name}: 训练完成，最佳平均奖励: {best_reward:.3f}")
        return rewards
    
    def evaluate_model(self, model_path=None):
        """评估训练好的模型"""
        logger.info("开始模型评估")
        
        # 创建评估环境
        eval_env_config = {
            'map_size': (config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT),
            'num_drones': [config.NUMBER_OF_DRONES],
            'mobility_models': ['random_waypoint'],
            'traffic_patterns': ['uniform'],
            'simulation_time': config.SIM_TIME,
        }
        
        eval_env = GbicrTrainingEnvironment(eval_env_config)
        trainer = GbicrTrainer(self.gbicr_config)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            trainer.agent.load_model(model_path)
            logger.info(f"已加载模型: {model_path}")
        
        # 评估多轮
        eval_episodes = 100
        eval_rewards = []
        success_count = 0
        
        for episode in range(eval_episodes):
            state = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 为评估提供可用动作
                available_actions = list(range(min(10, len(state))))
                if not available_actions:
                    available_actions = [0]
                action, _ = trainer.agent.select_action(state, available_actions, exploration=False)
                
                # 为评估添加必要的参数
                num_drones = eval_env.num_drones
                current_drone_id = random.randint(0, num_drones - 1)
                destination_id = random.randint(0, num_drones - 1)
                while destination_id == current_drone_id and num_drones > 1:
                    destination_id = random.randint(0, num_drones - 1)
                
                state, reward, done, info = eval_env.step(action, current_drone_id, destination_id)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            
            # 判断成功（可根据具体指标调整）
            if episode_reward > 0:
                success_count += 1
        
        avg_reward = np.mean(eval_rewards)
        success_rate = success_count / eval_episodes
        
        logger.info(f"评估结果:")
        logger.info(f"  平均奖励: {avg_reward:.3f}")
        logger.info(f"  成功率: {success_rate:.3f}")
        logger.info(f"  奖励标准差: {np.std(eval_rewards):.3f}")
        
        return avg_reward, success_rate
    
    def plot_training_results(self, rewards, save_path="training_results.png"):
        """绘制训练结果"""
        plt.figure(figsize=(12, 8))
        
        # 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(rewards)
        plt.title('训练奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # 移动平均奖励
        plt.subplot(2, 2, 2)
        window_size = 50
        if len(rewards) >= window_size:
            moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards)-window_size+1)]
            plt.plot(moving_avg)
        plt.title(f'移动平均奖励 (窗口={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Moving Average Reward')
        plt.grid(True)
        
        # 奖励分布
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=50, alpha=0.7)
        plt.title('奖励分布')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # 训练进度
        plt.subplot(2, 2, 4)
        if len(rewards) > 100:
            early_rewards = rewards[:len(rewards)//3]
            late_rewards = rewards[-len(rewards)//3:]
            plt.boxplot([early_rewards, late_rewards], labels=['Early', 'Late'])
        plt.title('训练前后期对比')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"训练结果图已保存到: {save_path}")


def main():
    """主函数"""
    print("="*60)
    print("GBICR 优化训练脚本")
    print("="*60)
    
    # 创建优化训练器
    optimizer = OptimizedGbicrTrainer()
    
    try:
        # 执行渐进式训练
        logger.info("开始优化训练过程...")
        start_time = time.time()
        
        trainer, rewards = optimizer.train_progressive()
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time/60:.1f} 分钟")
        
        # 绘制训练结果
        optimizer.plot_training_results(rewards)
        
        # 评估最终模型
        model_path = "./models/gbicr_optimized_model.npy"
        if os.path.exists(model_path):
            avg_reward, success_rate = optimizer.evaluate_model(model_path)
            
            print("\n" + "="*60)
            print("训练完成！")
            print(f"最终评估结果:")
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  成功率: {success_rate:.3f}")
            print(f"  模型保存位置: {model_path}")
            print("="*60)
            
            print("\n使用建议:")
            print("1. 将训练好的模型复制到 routing/gbicr/ 目录")
            print("2. 在 gbicr_config.py 中设置 pretrained_model_path")
            print("3. 重新运行 main.py 查看改善效果")
            print("4. 如果效果仍不理想，可以调整配置参数继续训练")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()