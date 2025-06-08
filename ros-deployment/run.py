#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from stable_baselines3 import TD3, PPO, SAC
import tf.transformations as tf_trans
from env import RobotNavEnv
import math
import argparse
# Global variables for goal tracking
goal_received = False
current_goal = None

def goal_callback(msg):
    """
    处理 RViz 目标姿势的回调函数。
    
    参数：
        msg(PoseStamped): 来自 RViz 的目标姿势信息
    """
    global goal_received, current_goal
    if msg.header.frame_id == "map":
        x = msg.pose.position.x
        y = msg.pose.position.y

        # 从四元数中提取偏航值
        orientation_q = msg.pose.orientation
        quaternion = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        _, _, yaw = tf_trans.euler_from_quaternion(quaternion)
        # 获取当前目标
        current_goal = [x, y, yaw]
        rospy.loginfo(f"Received valid goal: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.2f}°")
        goal_received = True
    else:
        rospy.logwarn(f"Ignoring goal from frame '{msg.header.frame_id}' (expecting 'map')")

def run_rl(eval_env, goal):
    """
    执行训练有素的策略，以达到指定目标。
    
    参数：
        eval_env (RobotNavEnv): 环境实例
        目标（列表）： [X、Y、偏航] 目标姿势
    """
    global model
    obs = eval_env.reset(goal)
    done = False
    total_reward = 0

    while not done and not rospy.is_shutdown():
        # 从训练有素的政策中获取行动
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward

    rospy.loginfo(f"Task finished. Total reward: {total_reward}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RL navigation with ROS')
    parser.add_argument('--model-path', type=str, default="models/td3_robot_nav_model.zip",
                       help='Path to the trained model')
    args = parser.parse_args()

    # 初始化ROS节点
    rospy.init_node('rl_goal_runner', anonymous=False)

    # 加载模型
    model = TD3.load(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Subscribe to RViz goal topic
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)

    # 初始化环境和主循环
    rate = rospy.Rate(1)  # 1 Hz
    eval_env = RobotNavEnv()

    while not rospy.is_shutdown():
        if goal_received:
            goal_received = False
            run_rl(eval_env, current_goal)
        rate.sleep()
