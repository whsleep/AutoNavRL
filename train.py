import gym
from gym import spaces
import numpy as np
import argparse
import yaml

from sim import SIM_ENV
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class RobotNavEnv(gym.Env):
    """
    Custom Gym environment that wraps the SIM_ENV simulator.
    
    This environment converts the simulator's outputs into a fixed-size observation,
    defines an action space, and scales actions as required for reinforcement learning.
    """
    
    def __init__(self, render=False):
        """
        Initialize the robot navigation environment.
        
        Args:
            render (bool): Whether to enable visualization
        """
        super(RobotNavEnv, self).__init__()
        
        # Read config.yaml 
        with open('config.yaml', 'r') as file:
            data = yaml.safe_load(file)
        self.linear_vel = data['train_param']['linear_vel']
        self.angular_vel = data['train_param']['angular_vel']
        self.lidar2d_max = data['train_param']['lidar2d_max']

        # Environment configuration
        self.render = render
        self.state_dim = 49  # Dimension of the observation space
        self.max_steps = 150  # Maximum number of steps per episode
        
        # done
        # Define action space (linear and angular velocity)
        self.action_space = spaces.Box(
            low=np.array([self.linear_vel[0], self.angular_vel[0]]),  # [min_linear_vel, min_angular_vel]
            high=np.array([self.linear_vel[1], self.angular_vel[1]]),   # [max_linear_vel, max_angular_vel]
            dtype=np.float32
        )
        
        # Define observation space (normalized to [-1, 1])
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Initialize simulator
        self.sim = SIM_ENV(render=render)
        
        # Initialize episode tracking
        self._reset_episode_tracking()
        
        # Get initial observation
        initial_data = self.sim.reset()
        self.current_obs, _ = self.prepare_state(initial_data)

    def _reset_episode_tracking(self):
        """Reset all episode tracking variables."""
        self.time = 0
        self.last_position = None
        self.total_distance = 0
        self.total_velocity = 0
        self.steps = 0

    def _calculate_metrics(self, current_position, action):
        """
        Calculate and update episode metrics.
        
        Args:
            current_position: Current robot position [x, y]
            action: Current action [linear_vel, angular_vel]
        """
        if self.last_position is not None:
            step_distance = np.linalg.norm(current_position - self.last_position)
            self.total_distance += step_distance
            self.total_velocity += np.linalg.norm(action)
        self.last_position = current_position
        self.steps += 1

    def _get_episode_info(self, terminal, reward):
        """
        Generate episode information dictionary.
        
        Args:
            terminal (bool): Whether episode is terminal
            reward (float): Final reward
            
        Returns:
            dict: Episode information
        """
        avg_velocity = self.total_velocity / self.steps if self.steps > 0 else 0
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
        Process raw simulator data into a normalized observation vector.
        
        Args:
            data: Raw simulator data tuple
            
        Returns:
            tuple: (processed_state, terminal_flag)
        """
        latest_scan, distance, cos, sin, collision, goal, diff_rad, action, reward = data
        latest_scan = np.array(latest_scan)

        # Handle infinite values in laser scan
        inf_mask = np.isinf(latest_scan)
        # done
        latest_scan[inf_mask] = self.lidar2d_max

        # Downsample laser scan data
        max_bins = self.state_dim - 7
        # done
        bin_size = int(np.floor(len(latest_scan) / max_bins))
        min_values = []
        
        # Create bins and get minimum values
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            # done
            min_values.append(min(bin) / self.lidar2d_max)
            if len(min_values) >= max_bins:
                break

        # Normalize values to [0, 1] range
        # done
        distance /= self.lidar2d_max
        # done
        lin_vel = (action[0] + self.linear_vel[1]) / (self.linear_vel[1]*2.0)
        ang_vel = (action[1] + self.angular_vel[1]) / (self.angular_vel[1]*2.0)
        
        # Convert angle difference to cos/sin representation
        rad_cos = np.cos(diff_rad)
        rad_sin = np.sin(diff_rad)

        # Combine all features into state vector
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel] + [rad_cos, rad_sin]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal

    def reset(self):
        """
        Reset the environment and return initial observation.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        sim_data = self.sim.reset()
        obs, _ = self.prepare_state(sim_data)
        self.current_obs = obs
        self._reset_episode_tracking()
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: [linear_velocity, angular_velocity]
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Process actions with deadzone
        lin_velocity = 0 if abs(action[0]) < 0.15 else action[0]
        ang_velocity = 0 if abs(action[1]) < 0.15 else action[1]

        # Step simulation
        sim_data = self.sim.step(lin_velocity=lin_velocity, ang_velocity=ang_velocity)
        obs, terminal = self.prepare_state(sim_data)
        reward = sim_data[-1]

        # Update metrics
        current_position = self.sim.env.get_robot_state()[:2]
        self._calculate_metrics(current_position, action)

        # Check termination conditions
        done = terminal
        self.time += 1
        if self.time >= self.max_steps:
            done = True
            reward = -100

        # Generate info dictionary
        info = self._get_episode_info(terminal, reward)
        
        self.current_obs = obs
        return obs, reward, done, info


def make_env(render=True):
    """
    Utility function for creating new instances of RobotNavEnv.
    This is used to create multiple parallel environments.
    
    Args:
        render (bool): Whether to enable visualization
        
    Returns:
        function: Environment initialization function
    """
    def _init():
        env = RobotNavEnv(render)
        return env
    return _init


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TD3 model for robot navigation')
    parser.add_argument('--num-envs', type=int, default=7,
                       help='Number of parallel training environments')
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='Total timesteps to train for')
    parser.add_argument('--model-path', type=str, default="models/td3_robot_nav_model",
                       help='Path to save/load model')
    parser.add_argument('--tensorboard-log', type=str, default="./td3_robot_nav_tensorboard/",
                       help='Tensorboard log directory')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training')
    args = parser.parse_args()

    # Create environments
    env_fns = [make_env() for _ in range(args.num_envs)]
    env_fns.append(make_env(render=args.render))  # Add one environment with optional rendering
    env = SubprocVecEnv(env_fns)

    # Create or load the TD3 model
    try:
        model = TD3.load(args.model_path, env=env)
        print(f"Loaded existing model from {args.model_path}")
    except:
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
        print("Created new model")

    # Train the model
    model.learn(total_timesteps=args.total_timesteps)

    # Evaluate the trained model
    eval_env = DummyVecEnv([make_env()])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    # Save the trained model
    model.save(args.model_path)
