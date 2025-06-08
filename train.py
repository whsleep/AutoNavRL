import gym
from gym import spaces
import numpy as np
import argparse

from sim import SIM_ENV
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class RobotNavEnv(gym.Env):
    """
    自定义的Gym环境,封装了SIM_ENV模拟器。
    
    该环境将模拟器的输出转换为固定大小的观测值，定义了动作空间，并根据强化学习的需求对动作进行缩放。
    """
    
    def __init__(self, render=True):
        """
        初始化机器人导航环境。
        
        Args:
            render (bool): 是否启用可视化
        """
        # 调用父类gym.Env的构造函数
        super(RobotNavEnv, self).__init__()
        
        # 环境配置
        self.render = render
        self.state_dim = 49  # 观测空间的维度
        self.max_steps = 150  # 每个episode的最大步数
        
        # 定义动作空间(线速度和角速度)
        self.action_space = spaces.Box(
            low=np.array([-0.6, -1.2]),  # [最小线速度, 最小角速度]
            high=np.array([0.6, 1.2]),   # [最大线速度, 最大角速度]
            dtype=np.float32
        )
        
        # 定义观测空间(归一化到[-1, 1])
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # 初始化模拟器
        self.sim = SIM_ENV(render=render)
        
        # 初始化episode跟踪变量
        self._reset_episode_tracking()
        
        # 获取初始观测值
        initial_data = self.sim.reset()
        self.current_obs, _ = self.prepare_state(initial_data)

    def _reset_episode_tracking(self):
        """
        重置所有episode跟踪变量。
        """
        self.time = 0 # 仿真离散时间
        self.last_position = None # 上次位置
        self.total_distance = 0 # 运行的路径长度
        self.total_velocity = 0 # 运行的全部速度标量
        self.steps = 0 

    def _calculate_metrics(self, current_position, action):
        """
        计算并更新episode指标。
        
        Args:
            current_position: 当前机器人位置 [x, y]
            action: 当前动作 [线速度, 角速度]
        """
        if self.last_position is not None:
            # 单次移动距离
            step_distance = np.linalg.norm(current_position - self.last_position)
            # 计算总距离
            self.total_distance += step_distance
            # 计算总的速度矢量
            self.total_velocity += np.linalg.norm(action)
        # 更新上次位置
        self.last_position = current_position
        self.steps += 1

    def _get_episode_info(self, terminal, reward):
        """
        生成episode信息字典。
        
        Args:
            terminal (bool): episode是否终止
            reward (float): 最终奖励
            
        Returns:
            dict: episode信息
        """
        # 计算平均速度
        avg_velocity = self.total_velocity / self.steps if self.steps > 0 else 0
        # 返回信息
        return {
            'success': terminal and reward > 0,
            'collision': terminal and reward < 0,
            'steps': self.steps,
            'total_distance': self.total_distance,
            'average_velocity': avg_velocity,
            'time_limit_reached': self.time >= self.max_steps
        }

    def prepare_state(self, data):
        """
        将原始模拟器数据处理成归一化的观测向量。
        
        Args:
            data: 原始模拟器数据元组
            
        Returns:
            tuple: (处理后的状态, 终止标志)
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # 处理激光扫描中的无穷大值
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 10 # 无穷大值设置为雷达的最远检测距离

        # 下采样激光扫描数据 
        max_bins = self.state_dim - 7
        # 将2D点云按照扇形区域进行划分
        bin_size = int(np.ceil(len(latest_scan) / max_bins))
        min_values = []

        # 创建扇形范围并获取最小值
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # 找到当前扇形中的最小值并将其追加到min_values列表中
            min_values.append(min(bin) / 10)

        # 将值归一化到[0, 1]范围
        distance /= 10
        lin_vel = (action[0] + 0.6) / 1.2
        ang_vel = (action[1] + 1.2) / 2.4
        
        # 将角度差转换为cos/sin表示
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)

        # 将所有特征组合成状态向量
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel] + [rad_cos, rad_sin]

        # 判断是否满足状态维度
        assert len(state) == self.state_dim
        # 抵达目标或者发生碰撞则停止
        terminal = 1 if collision or goal else 0

        return state, terminal

    def reset(self):
        """
        重置环境并返回初始观测值。
        
        Returns:
            numpy.ndarray: 初始观测值
        """
        sim_data = self.sim.reset()
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = obs
        self._reset_episode_tracking()
        return obs

    def step(self, action):
        """
        在环境中执行一步。
        
        Args:
            action: [线速度, 角速度]
            
        Returns:
            tuple: (观测值, 奖励, 终止标志, 信息)
        """
        # 处理带有死区的动作
        lin_velocity = 0 if abs(action[0]) < 0.15 else action[0]
        ang_velocity = 0 if abs(action[1]) < 0.15 else action[1]

        # 执行模拟
        sim_data = self.sim.step(lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        
        obs, terminal = self.prepare_minstate(sim_data)
        reward = sim_data[-1]

        # 更新指标
        current_position = self.sim.env.get_robot_state()[:2]
        self._calculate_metrics(current_position, action)

        # 检查终止条件
        done = terminal
        self.time += 1
        if self.time >= self.max_steps:
            done = True
            reward = -100

        # 生成信息字典
        info = self._get_episode_info(terminal, reward)
        
        self.current_obs = obs
        return obs, reward, done, info


def make_env(render=False):
    """
    创建新实例的RobotNavEnv的工具函数。
    用于创建多个并行环境。
    
    Args:
        render (bool): 是否启用可视化
        
    Returns:
        function: 环境初始化函数
    """
    def _init():
        env = RobotNavEnv(render)
        return env
    return _init


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练TD3模型用于机器人导航')
    parser.add_argument('--num-envs', type=int, default=7,
                       help='并行训练环境的数量')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='训练的总时间步数')
    parser.add_argument('--model-path', type=str, default="models/td3_robot_nav_model",
                       help='保存/加载模型的路径')
    parser.add_argument('--tensorboard-log', type=str, default="./td3_robot_nav_tensorboard/",
                       help='Tensorboard日志目录')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='评估的episode数量')
    parser.add_argument('--render', action='store_true',
                       help='训练期间启用渲染')
    args = parser.parse_args()

    # 创建环境
    env_fns = [make_env() for _ in range(args.num_envs)]
    env_fns.append(make_env(render=args.render))  # 添加一个可选渲染的环境
    env = SubprocVecEnv(env_fns)

    # 创建或加载TD3模型
    try:
        model = TD3.load(args.model_path, env=env)
        print(f"从{args.model_path}加载现有模型")
    except:
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
        print("创建新模型")

    # 训练模型
    model.learn(total_timesteps=args.total_timesteps)

    # 评估训练后的模型
    eval_env = DummyVecEnv([make_env()])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"平均奖励: {mean_reward} +/- {std_reward}")
    
    # 保存训练后的模型
    model.save(args.model_path)