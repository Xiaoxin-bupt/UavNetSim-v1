import simpy
import random
from simulator.log import logger
from phy.phy import Phy
from utils import config


class SimpleMac:
    """
    简化的MAC层实现，跳过复杂的信道竞争机制
    
    主要特点：
    1. 无信道竞争和退避
    2. 使用固定发送速率计算传输时延
    3. 简化的ACK机制
    4. 适用于路由算法性能测试
    

    """

    def __init__(self, drone):
        self.my_drone = drone
        self.simulator = drone.simulator
        self.rng_mac = random.Random(self.my_drone.identifier + self.my_drone.simulator.seed + 5)
        self.env = drone.env
        self.phy = Phy(self)
        self.enable_ack = True
        
        # 简化的ACK处理
        self.wait_ack_process_dict = dict()
        self.wait_ack_process_finish = dict()
        self.wait_ack_process_count = 0
        
        # 固定传输参数
        self.fixed_transmission_rate = config.BIT_RATE  # 使用配置的比特率
        self.fixed_propagation_delay = 10  # 固定传播延迟 (微秒)

    def mac_send(self, pkd):
        """
        简化的数据包发送过程
        
        Parameters:
            pkd: 需要发送的数据包
        """
        
        # 记录首次尝试时间
        if pkd.number_retransmission_attempt[self.my_drone.identifier] == 1:
            pkd.first_attempt_time = self.env.now
            
        logger.info('At time: %s (us) ---- UAV: %s starts simple MAC transmission for packet: %s',
                    self.env.now, self.my_drone.identifier, pkd.packet_id)
        
        # 计算传输时延：数据包长度 / 传输速率
        transmission_delay = pkd.packet_length / self.fixed_transmission_rate * 1e6  # 转换为微秒
        
        # 添加固定的传播延迟
        total_delay = transmission_delay + self.fixed_propagation_delay
        
        # 模拟传输延迟
        yield self.env.timeout(total_delay)
        
        # 发送数据包
        if hasattr(pkd, 'next_hop_id') and pkd.next_hop_id is not None:
            # 单播数据包
            self.phy.unicast(pkd, pkd.next_hop_id)
            logger.info('At time: %s (us) ---- UAV: %s unicasts packet: %s to UAV: %s',
                        self.env.now, self.my_drone.identifier, pkd.packet_id, pkd.next_hop_id)
            
            # 触发接收方的接收过程
            if pkd.next_hop_id < len(self.simulator.drones):
                self.simulator.drones[pkd.next_hop_id].receive()
                
            # 如果需要ACK，启动等待ACK过程
            if self.enable_ack and hasattr(pkd, 'dst_drone'):
                yield self.env.process(self.wait_ack(pkd))
        else:
            # 广播数据包
            self.phy.broadcast(pkd)
            logger.info('At time: %s (us) ---- UAV: %s broadcasts packet: %s',
                        self.env.now, self.my_drone.identifier, pkd.packet_id)
            
            # 触发所有邻居的接收过程
            for drone in self.simulator.drones:
                if drone.identifier != self.my_drone.identifier:
                    drone.receive()

    def wait_ack(self, pkd):
        """
        简化的ACK等待过程
        
        Parameters:
            pkd: 等待ACK的数据包
        """
        
        # 创建等待ACK的进程标识
        self.wait_ack_process_count += 1
        key = ''.join(['wait_ack', str(self.my_drone.identifier), '_', str(pkd.packet_id)])
        
        self.wait_ack_process_finish[key] = 0
        
        try:
            # 等待ACK超时时间（简化为固定值）
            ack_timeout = config.ACK_TIMEOUT if hasattr(config, 'ACK_TIMEOUT') else 1000  # 1ms默认超时
            yield self.env.timeout(ack_timeout)
            
            # ACK超时，调用路由协议的惩罚函数
            if hasattr(self.my_drone.routing_protocol, 'penalize'):
                self.my_drone.routing_protocol.penalize(pkd)
            
            logger.info('At time: %s (us) ---- ACK timeout for packet: %s at UAV: %s',
                        self.env.now, pkd.packet_id, self.my_drone.identifier)
            
            # 检查是否需要重传
            if pkd.number_retransmission_attempt[self.my_drone.identifier] < config.MAX_RETRANSMISSION_ATTEMPT:
                # 重传数据包
                yield self.env.process(self.my_drone.packet_coming(pkd))
            else:
                # 达到最大重传次数，丢弃数据包
                logger.info('At time: %s (us) ---- Packet: %s is dropped after max retransmissions',
                            self.env.now, pkd.packet_id)
                
        except simpy.Interrupt:
            # 收到ACK，传输成功
            logger.info('At time: %s (us) ---- UAV: %s receives ACK for packet: %s',
                        self.env.now, self.my_drone.identifier, pkd.packet_id)
            
        finally:
            self.wait_ack_process_finish[key] = 1

    def get_transmission_delay(self, packet_length):
        """
        计算传输延迟
        
        Parameters:
            packet_length: 数据包长度（比特）
            
        Returns:
            传输延迟（微秒）
        """
        return packet_length / self.fixed_transmission_rate * 1e6 + self.fixed_propagation_delay

    def set_transmission_rate(self, rate):
        """
        设置传输速率
        
        Parameters:
            rate: 传输速率（bps）
        """
        self.fixed_transmission_rate = rate
        logger.info('SimpleMac transmission rate set to: %s bps', rate)

    def set_propagation_delay(self, delay):
        """
        设置传播延迟
        
        Parameters:
            delay: 传播延迟（微秒）
        """
        self.fixed_propagation_delay = delay
        logger.info('SimpleMac propagation delay set to: %s us', delay)