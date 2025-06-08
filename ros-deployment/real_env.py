#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import tf
import time

def reduce_lidar_scan(scan_array):
    """
    通过对 4 组读数取平均值来降低激光雷达扫描数据的维度。
    
    参数：
        scan_array(numpy.ndarray): 原始激光雷达扫描数据
        
    返回值
        numpy.ndarray: 还原的激光雷达扫描数据
    """
    # 确保长度能被 4 整除，否则应修剪多余部分
    if len(scan_array) % 4 != 0:
        scan_array = scan_array[:-(len(scan_array) % 4)]

    # 将数组重塑为 4 个元素的块状结构
    reshaped_array = np.array(scan_array).reshape(-1, 4)

    # 计算每个数据块的平均值，不包括零
    result = []
    for chunk in reshaped_array:
        non_zero_values = chunk[chunk != 0]
        if len(non_zero_values) == 0:
            result.append(10)  # All zeros, use 10
        else:
            result.append(np.mean(non_zero_values))  # Take the mean of non-zero values

    return np.array(result)

def constrain_lidar_scan(bot_pos, yaw, angles, lidar_ranges, box_limits):
    """
    将激光雷达读数限制在指定的方框范围内。
    如果你的竞技场缺乏适当的边界，这将非常有用。
    
    参数：
        bot_pos （元组）： (x、y)机器人位置
        yaw （浮点数）： 机器人方向
        angles (numpy.ndarray): 激光雷达光束角度
        lidar_ranges (numpy.ndarray): 激光雷达测距读数
        box_limits （元组）： (min_x, max_x, min_y, max_y) 环境边界
        
    返回：
        numpy.ndarray: 受约束的激光雷达范围
    """
    bot_x, bot_y = bot_pos
    min_x, max_x, min_y, max_y = box_limits
    
    constrained_ranges = np.empty_like(lidar_ranges)

    for i, (angle, lidar_range) in enumerate(zip(angles, lidar_ranges)):
        adjusted_angle = angle + yaw
        ray_dx = np.cos(adjusted_angle)
        ray_dy = np.sin(adjusted_angle)

        distances = []

        # Check intersections with vertical boundaries
        if ray_dx != 0:
            t1 = (min_x - bot_x) / ray_dx
            t2 = (max_x - bot_x) / ray_dx
            distances.extend([t for t in [t1, t2] if t > 0])

        # Check intersections with horizontal boundaries
        if ray_dy != 0:
            t3 = (min_y - bot_y) / ray_dy
            t4 = (max_y - bot_y) / ray_dy
            distances.extend([t for t in [t3, t4] if t > 0])

        if distances:
            min_boundary_dist = min(distances)
            constrained_ranges[i] = min(lidar_range, min_boundary_dist)
        else:
            constrained_ranges[i] = lidar_range

    return constrained_ranges

class REAL_ENV:
    """
    与 ROS 接口的真实机器人环境类。
    
    该类处理
    - 激光雷达数据处理
    - 机器人运动控制
    - 状态跟踪和目标进度
    - 碰撞检测
    
    属性
        scan_sub (rospy.Subscriber): 激光雷达扫描订阅器
        tf_listener (tf.TransformListener): 用于姿势跟踪的 TF 监听器
        cmd_vel_pub (rospy.Publisher): 速度指令发布器
        latest_scan （列表）： 最新的激光雷达扫描数据
        robot_pose(列表): 当前机器人位置 [x, y]
        robot_yaw(浮点): 当前机器人方向
        collision(bool): 碰撞标志 碰撞标志
        goal_reached (bool): 已达到目标标志： 达到目标标志
        robot_goal(列表): 目标姿势 [x、y、yaw]
    """
    
    def __init__(self, goal_pose=None):
        """
        初始化真实机器人环境
        """
        # Subscribers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.tf_listener = tf.TransformListener()

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # State variables
        self.latest_scan = []
        self.robot_pose = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.collision = False
        self.goal_reached = False
        self.robot_goal = goal_pose
        
        # Performance metrics
        self.start_time = time.time()
        self.path_length = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0
        self.timestep = 0
        self.prev_pose = None

        rospy.sleep(1)  # Initialization wait

    def scan_callback(self, data):
        """
        处理收到的激光雷达扫描数据
        
        参数：
            data (LaserScan): 原始激光雷达扫描信息
        """
        latest_scan = reduce_lidar_scan(data.ranges)
 
        bot_position = self.robot_pose
        bot_yaw = self.robot_yaw
        lidar_offset = 0.15  # 激光雷达位于机器人中心前方 0.15 米处

        # 计算激光雷达位置
        # lidar_x = bot_position[0] + lidar_offset * np.cos(self.robot_yaw)
        # lidar_y = bot_position[1] + lidar_offset * np.sin(self.robot_yaw)
        
        lidar_x = bot_position[0]
        lidar_y = bot_position[1] 

        # 生成激光雷达光束角度按照提取后的个数
        lidar_angles = np.linspace(0, 2 * np.pi, num=len(latest_scan))
        box_limits = (-5, 5, -5, 5)  # 环境边界

        # 将激光雷达读数限制在环境边界内
        latest_scan = constrain_lidar_scan(
            [lidar_x, lidar_y], 
            bot_yaw, 
            lidar_angles, 
            latest_scan, 
            box_limits
        )
        
        # 旋转扫描数据，与机器人方向保持一致
        self.latest_scan = latest_scan
        # self.latest_scan = np.roll(latest_scan, int(len(latest_scan) * 1/2))
    
        # 检查碰撞
        self.collision = min(self.latest_scan) < 0.15  # 15cm collision threshold
        if self.collision:
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_vel_pub.publish(cmd)
            print("Collision detected!")

    def get_robot_pose_from_tf(self):
        """
        从 TF 获取当前机器人姿势。
        """
        self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
        (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
        
        self.robot_pose = [trans[0], trans[1]]
        _, _, self.robot_yaw = euler_from_quaternion(rot)

        print("TF Pose:", self.robot_pose, self.robot_yaw)

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        在环境中执行一个步骤。
        
        参数
            lin_velocity （浮点）： 线速度指令
            ang_velocity （浮点速度）： 角速度指令
            
        返回 返回 返回 返回值值值值
            元组： (scan_data, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        self.timestep += 1
        self.get_robot_pose_from_tf()
        
        if self.robot_pose is None:
            rospy.logwarn("Waiting for AMCL pose...")
            rospy.sleep(0.1)
            return None

        # 发布速度指令
        cmd = Twist()
        cmd.linear.x = lin_velocity
        cmd.angular.z = ang_velocity
        if not self.collision and not self.goal_reached:
            self.cmd_vel_pub.publish(cmd)

        rospy.sleep(0.1)  # Allow time for motion

        # 计算目标向量和进度
        goal_vector = [
            self.robot_goal[0] - self.robot_pose[0],
            self.robot_goal[1] - self.robot_pose[1],
        ]

        # 计算与球门的角度差
        diff_rad = float(((-self.robot_yaw + self.robot_goal[2] + np.pi) % (2 * np.pi)) - np.pi)
        distance = np.linalg.norm(goal_vector)
        goal = (distance < 0.15 and abs(diff_rad) < 0.15)  # 15cm position and 0.15rad angle threshold

        # 更新路径长度
        if self.prev_pose is not None:
            delta = np.sqrt((self.robot_pose[0] - self.prev_pose[0])**2 +
                         (self.robot_pose[1] - self.prev_pose[1])**2)
            self.path_length += delta
        self.prev_pose = self.robot_pose

        # 更新速度总和，以便在度量中求取平均值
        self.linear_vel_sum += abs(lin_velocity)
        self.angular_vel_sum += abs(ang_velocity)

        if goal:
            print(self.robot_yaw, self.robot_goal[2])
            rospy.loginfo("Goal reached!")
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_vel_pub.publish(cmd)
            self.goal_reached = True
            
            # Log performance metrics
            print("Time Taken:", time.time() - self.start_time)
            print("Distance:", distance)
            print("Ang_diff:", diff_rad)
            print("Path Length:", self.path_length)
            print('Avg Linear:', self.linear_vel_sum/self.timestep)
            print('Avg Ang:', self.angular_vel_sum/self.timestep)

        # 计算观测成分
        pose_vector = [np.cos(self.robot_yaw), np.sin(self.robot_yaw)]
        cos, sin = self.cossin(pose_vector, goal_vector)
        action = [lin_velocity, ang_velocity]
        reward = 0  # Made for inference, not used

        return self.latest_scan, distance, cos, sin, self.collision, goal, diff_rad, action, reward

    def reset(self, goal_pose=None):
        """
        用新目标重置环境
        
        参数：
            goal_pose （列表）： [x、y、yaw] 目标姿势
            
        返回 返回 返回 返回值值值值
            元组： 初始环境状态
        """
        # Reset metrics
        self.start_time = time.time()
        self.path_length = 0
        self.timestep = 0
        self.linear_vel_sum = 0
        self.angular_vel_sum = 0
        
        # Get initial pose
        self.get_robot_pose_from_tf()
        self.prev_pose = None

        rospy.loginfo("Manually reset the robot and localization if needed.")
        rospy.sleep(2)  # Wait for manual reset or AMCL reinitialization

        # Set new goal
        self.robot_goal = goal_pose
        self.collision = False
        self.goal_reached = False

        # Take initial step
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    @staticmethod
    def cossin(vec1, vec2):
        """
        Compute cosine and sine of angle between two vectors.
        
        Args:
            vec1 (list): First vector [x, y]
            vec2 (list): Second vector [x, y]
            
        Returns:
            tuple: (cosine, sine) of angle between vectors
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin
    
if __name__ == "__main__":
    # Test the environment
    rospy.init_node('real_robot_env', anonymous=True)
    env = REAL_ENV(goal_pose=[0,0,0])
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        step_result = env.step(0.0, 0.0)
        if step_result:
            scan, distance, cos, sin, collision, goal, diff_rad, action, reward = step_result
            rospy.loginfo(f"Distance: {distance:.2f}, Reward: {reward:.2f}")
            if collision or goal:
                break
        rate.sleep()