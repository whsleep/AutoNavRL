# ROS部署

## 几个关键话题

`/move_base_simple/goal` 是`rviz`中手动设置目标点发布的话题，类型为[PoseStamped](https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html)。

`/scan`获取的激光雷达信息，类型为[LaserScan](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html)

`/tf`获取激光雷达在`map`坐标系下的坐标值。

## '__main__'

### 订阅目标发布

```python
# 订阅 /move_base_simple/goal 节点
rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)
```

### 创建虚拟环境

```python
from env import RobotNavEnv
    eval_env = RobotNavEnv()
```

这里的环境主要作用是将`ros`话题传入的信息转换为`model`可以处理的格式。

`env.py`中的`RobotNavEnv`类同样继承自`gym.Env`类，所以需要重写

`__init__(self)`，`reset(self, goal)`和`step(self, action)`函数，内容和训练中使用的内容类似，需要重点注意的是

```python
from real_env import REAL_ENV
    self.sim = REAL_ENV(goal_pose=self.goal)
```
这里从`real_env`中`import`了`REAL_ENV`，这个类才是与实车交互的实际接口，该类位于`real_env.py`
文件内，接下来查看该类内容.

#### `REAL_ENV`
##### `__init__`

- 订阅话题

订阅`/scan`和`/tf`变换。

```python
self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
self.tf_listener = tf.TransformListener()
```

- 速度指令发布
```python
self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
```

##### `scan_callback()`

调用 `reduce_lidar_scan` 对点云信息进行处理

>
>首先将二维点云数量修剪为`4`的倍数
>
>```python
>if len(scan_array) % 4 != 0:
>    scan_array = scan_array[:-(len(scan_array) % 4)]
>```
>
>将一维点云数组重塑为每行为4个元素的二维数组，并提取4个元素的平均值(不包括0)
>
>```python
># 将数组重塑为 4 个元素的块状结构
>reshaped_array = np.array(scan_array).reshape(-1, 4)
>
># 计算每个数据块的平均值，不包括零
>result = []
>for chunk in reshaped_array:
>    non_zero_values = chunk[chunk != 0]
>    if len(non_zero_values) == 0:
>        result.append(10)  # All zeros, use 10
>    else:
>        result.append(np.mean(non_zero_values))
>```

再调用`constrain_lidar_scan()`对雷达数据进行限制

限制信息如下

```python
lidar_x = bot_position[0]
lidar_y = bot_position[1] 

# 生成激光雷达光束角度按照提取后的个数
lidar_angles = np.linspace(0, 2 * np.pi, num=len(latest_scan))
box_limits = (-5, 5, -5, 5)  # 环境边界
```

其中`bot_position`为智能体在全局坐标系下的坐标，这里我假设`lidar`和智能体坐标重合。

根据限制后点云数量，在$[0,2\pi]$内均匀生成对应数量的方位角。

`constrain_lidar_scan()`将二维点云限制在(min_x, max_x, min_y, max_y) 环境边界内。

##### `get_robot_pose_from_tf()`

订阅`tf`变换，获得`robot_pose`和`robot_yaw`信息。

##### `step()`

