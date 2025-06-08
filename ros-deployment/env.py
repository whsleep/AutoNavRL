import gym
from gym import spaces
import numpy as np
from real_env import REAL_ENV


class RobotNavEnv(gym.Env):
    """
    自定义 Gym 环境，用于封装机器人导航的 REAL_ENV 模拟器。
    
    该环境
    - 将模拟器的输出转换为固定大小的观测空间
    - 为线速度和角速度定义连续的动作空间
    - 处理状态规范化和预处理
    - 管理情节终止条件
    
    属性
        action_space (gym.spaces.Box): 线速度和角速度的连续动作空间
        observation_space (gym.spaces.Box): 固定大小的观测空间
        state_dim (int): 状态向量的尺寸
        sim (REAL_ENV): 真实环境模拟器实例
        time (int): 剧集终止的步骤计数器
    """
    def __init__(self):
        super(RobotNavEnv, self).__init__()
        # 动作空间: [linear_velocity, angular_velocity]
        # 线速度范围: [-0.6, 0.6] m/s
        # 角速度范围: [-1.2, 1.2] rad/s
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]), 
            high=np.array([0.6, 1.2]), 
            dtype=np.float32
        )
        
        # 观测空间：包含 49 维向量：
        # - 分档激光雷达扫描数据（42 维）
        # - 与目标的距离（1 维）
        # 目标方向 cos/sin（2 维）
        # 当前线速度/角速度（2 维）
        # 球门角度差余弦/正弦（2 维）
        self.state_dim = 49
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # 用默认目标初始化模拟器
        self.goal = [0, 0, 0]
        self.sim = REAL_ENV(goal_pose=self.goal)
        self.time = 0

    def prepare_state(self, data):
        """
        将原始环境数据处理为规范化状态向量。
        
        参数
            data(元组): 原始环境数据，包含
                - 激光雷达扫描数据
                - 到目标的距离
                - 目标方向余弦/正弦
                - 碰撞标志
                - 到达目标标志
                - 角度差
                - 最后动作
                - 奖励
        
        返回值
            元组：（归一化状态、终点标志）
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # Handle infinite values in LIDAR data
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 10

        # Bin LIDAR data to reduce dimensionality
        max_bins = self.state_dim - 7
        bin_size = int(np.floor(len(latest_scan) / max_bins))
        min_values = []
        
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 10)
            if len(min_values) >= max_bins:
                break

        # Normalize distance and velocities
        distance /= 10
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4

        # Convert angle difference to cos/sin representation
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)

        # Combine all state components
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel] + [rad_cos, rad_sin]
        assert len(state) == self.state_dim

        terminal = 1 if collision or goal else 0
        return state, terminal

    def reset(self, goal):
        """
        Reset the environment with a new goal.
        
        Args:
            goal (list): [x, y, yaw] target pose
            
        Returns:
            numpy.ndarray: Initial observation
        """
        self.goal = goal
        sim_data = self.sim.reset(goal_pose=self.goal)
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = obs
        self.time = 0
        return obs

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (numpy.ndarray): [linear_velocity, angular_velocity]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        lin_velocity, ang_velocity = action

        # Execute action in simulator
        sim_data = self.sim.step(lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data)
        reward = sim_data[-1]
        
        # Check termination conditions
        done = terminal
        self.time += 1
        if self.time >= 5000:  # Time limit
            done = True
            reward = 0
            print("Episode terminated due to time limit")
        
        info = {}
        self.current_obs = obs
        return obs, reward, done, info

