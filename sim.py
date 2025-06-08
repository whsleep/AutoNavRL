import numpy as np
import random
import shapely
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase


class SIM_ENV:
    """
    机器人导航模拟器环境包装器。
    
    该类为机器人导航模拟提供了高级接口、
    处理状态跟踪、奖励计算和环境交互。
    """
    
    def __init__(self, world_file="robot_world.yaml", render=False):
        """
        初始化模拟环境。
        
        参数：
            world_file (str): 世界配置文件的路径
            render (bool): 是否启用可视化
        """
        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        # 获取机器人目标点
        self.robot_goal = self.env.get_robot_info(0).goal
        
        # 初始化跟踪变量
        self._reset_tracking()

    def _reset_tracking(self):
        """重置距离和角度差的跟踪变量。"""
        self.prev_distance = None
        self.prev_diff_rad = None

    def _calculate_robot_metrics(self, robot_state):
        """
        计算与机器人相关的指标，包括目标矢量、距离和方向。
        
        参数
            robot_state: 机器人当前的状态 [x, y, theta]
            
        返回值
            tuple: (goal_vector, distance, cos, sin, diff_rad)
        """
        # 计算目标向量
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        
        # 计算机器人方向与目标方向之间的角度差 [-π, π]
        diff_rad = float(((-robot_state[2] + self.env.robot.goal[2] + np.pi) % (2 * np.pi)) - np.pi)
        
        # 计算距离目标点距离与位置
        distance = np.linalg.norm(goal_vector)
        # 方向矢量
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        # 获取方向矢量和目标向量之间的正弦余弦值
        cos, sin = self._calculate_cossin(pose_vector, goal_vector)
        
        return goal_vector, distance, cos, sin, diff_rad
    
    """
        静态方法的主要用途是将一些与类相关的功能封装在类中，
        但这些功能不需要访问类的实例或类本身。
        静态方法通常用于工具函数或辅助函数。
    """
    @staticmethod
    def _calculate_cossin(vec1, vec2):
        """
        计算两个矢量之间的余弦和正弦。
        
        参数
            vec1: 第一个向量
            vec2: 第二向量
            
        返回值
            元组：向量间夹角的(余弦、正弦)
        笔记
            dot(v1, v2) = |v1||v2|cos(theta)
            v1 x v2 = |v1||v2|sin(theta)
        """
        # 向量归一化
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        # 计算余弦值
        cos = np.dot(vec1, vec2)
        # 计算正弦值
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    def _calculate_reward(self, goal, collision, distance_delta, action, laser_scan, delta_rad):
        """
        根据各种因素计算奖励，包括目标实现情况、
        避免碰撞和运动效率。
        
        参数：
            goal (bool): 是否达到目标
            collision (bool): 是否发生碰撞
            distance_delta (float): 到目标的距离变化
            action (list): [线速度、角速度］
            laser_scan (list): 激光扫描读数
            delta_rad (float): 角度差变化
            
        返回值
            float: 计算出的奖励
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        
        # 奖励组件
        progress_reward = distance_delta * 10  # 对接近目标的奖励
        dir_progress = delta_rad * 1  # 与目标保持一致的奖励
        time_penalty = -0.65  # 每个时间步的罚款
        rotation_penalty = -abs(action[1]) * 0.4  # 过度旋转的处罚
        
        # 避障
        safe_distance = 1.35 # 安全距离
        min_dist = min(laser_scan) # 雷达最近距离
        # 安全距离大于最小距离惩罚项为0，否则为 安全距离减去最小距离
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0
        
        return progress_reward + time_penalty + obstacle_penalty + dir_progress + rotation_penalty

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        执行一个模拟步骤。
        
        参数
            lin_velocity (float): 线速度
            ang_velocity (float): 角速度
            
        返回值
            元组： (laser_scan, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        # 单步仿真
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        if self.env.display:
            self.env.render()

        # 获取传感器数据
        scan = self.env.get_lidar_scan()
        robot_state = self.env.get_robot_state()
        
        # 计算指标
        goal_vector, distance, cos, sin, diff_rad = self._calculate_robot_metrics(robot_state)
        
        # 计算机器人状态和目标的距离和角度偏差
        if self.prev_distance is None:
            distance_delta = delta_rad = 0
        else:
            distance_delta = self.prev_distance - distance # 相邻两次步长差值
            delta_rad = abs(self.prev_diff_rad) - abs(diff_rad) # 向量两次方向角差值
        
        # 更新上次距离和角度差
        self.prev_distance = distance
        self.prev_diff_rad = diff_rad
        
        # 获取状态并计算奖励
        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self._calculate_reward(goal, collision, distance_delta, action, scan["ranges"], delta_rad)
        
        # 判断是否抵达目标点
        if goal:
            print("Goal reached")

        
        return scan["ranges"], distance, cos, sin, collision, goal, diff_rad, action, reward

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=True):
        """
        重置模拟环境
        
        参数：
            robot_state (list): 机器人初始状态 [x、y、theta]
            robot_goal (list): 目标位置[x, y, theta]
            random_obstacles (list): 是否放置随机障碍物
            
        返回值
            tuple: 初始状态信息
        """
        # 初始化机器人状态
        if robot_state is None:
            # 随机刷新在地图范围内
            robot_state = [[random.uniform(0.5, 5.5)], 
                          [random.uniform(0.5, 5.5)], 
                          [0]]
        self.env.robot.set_state(state=np.array(robot_state), init=True)

        # 放置障碍物
        if random_obstacles:
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[6, 6, 3.14],
                ids=list(range(1, 7)),
                non_overlapping=True
            )

        # 设置目标
        if robot_goal is None:
            robot_goal = self._generate_valid_goal()
        
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        self._reset_tracking()
        
        # 获取初始状态
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    def _generate_valid_goal(self):
        """
        生成一个不与障碍物重叠的有效目标位置。
        
        返回值
            列表： 有效目标位置 [x、y、θ]
        """
        while True:
            goal = [[random.uniform(0.5, 5.5)], 
                   [random.uniform(0.5, 5.5)], 
                   [random.uniform(-3.14, 3.14)]]
            
            # 检查目标是否与障碍物重叠
            shape = {"name": "circle", "radius": 0.4}
            state = [goal[0], goal[1], goal[2]]
            # 创建几何对象
            gf = GeometryFactory.create_geometry(**shape)
            geometry = gf.step(np.c_[state])
            
            if not any(shapely.intersects(geometry, obj._geometry) for obj in self.env.obstacle_list):
                return goal