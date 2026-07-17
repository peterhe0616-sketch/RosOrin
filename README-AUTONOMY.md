# ROSOrin 连续自主导航

这一版复用树莓派 `MentorPi` 容器中已安装的 ROS 2 Humble、Nav2、slam_toolbox 和 robot_localization，不重复下载 ROS。Windows 电脑运行本地 Qwen3-VL-4B INT4 和网页控制台，机器人端执行确定性的建图、规划、避障与恢复。

## 已实现功能

- `slam_toolbox` 使用 `/scan_raw` 实时生成二维占据栅格，并发布 `map -> odom` 定位变换
- 现有 `robot_localization` 融合轮式里程计与 IMU，Nav2 使用 `map / odom / base_footprint` 完成实时定位
- NavFn A* 全局规划、DWB 局部规划、20 Hz 速度平滑和连续绕障
- 前、后、左、右均使用 `0.16 m` 的最终碰撞停止边界；局部和全局代价地图膨胀半径为 `0.16 m`
- Nav2 标准恢复树包含重新规划、旋转和受控倒车；控制台也可单独请求倒车 `0.15 m`
- 通过地图点击或输入 `x / y / yaw` 发送地图坐标目标
- 紧急停止、暂停、继续、取消目标、旋转、保存地图
- 电脑端每秒采样一帧；若一次推理超过一秒，会把等待中的连续帧合并到下一轮，而不是逐帧积压
- Qwen3-VL 多轮 Chat 保留场景记忆和最近 12 条用户提示词；每 12 轮用模型生成的短记忆压缩上下文，避免无限增长，并把压缩记忆保存到 `state/autonomy_memory.json` 供下次启动恢复
- 模型同时看到连续相机帧、slam_toolbox 拼接的 LiDAR 二维地图、LiDAR 方向距离和 Nav2 状态
- 界面实时显示“场景、可核验观察、风险、决策、简短理由、压缩记忆”。这是模型给出的可审计决策依据，不声称展示模型内部隐藏思维链
- AI 自动下发高层相对目标默认关闭；开启后仍只给 Nav2 目标，不直接发轮速
- 状态中包含实际线速度/转弯角速度、IMU 横滚/俯仰、振动 RMS、卡住和疑似打滑；可用于模型下一轮判断

## 数据与控制分层

```text
Camera 1 Hz ───────────────┐
LiDAR /map + sectors ──────┼─ Qwen3-VL（连续语义、记忆、相对目标）
Nav/IMU/odom status ───────┘                    │
                                                ▼
地图点击/数值目标/实时提示词 ─────────────── /vla/nav_goal
                                                │
                                                ▼
slam_toolbox → Nav2 A* + DWB → velocity_smoother
                                                │
                                                ▼
                         Collision Monitor 0.16 m → 电机
```

模型是低频任务规划器；地图、定位、路径和障碍物约束都在 ROS 2 高频闭环中连续运行，所以小车不会等待每一次模型推理后才走一步。

## 部署机器人端

先确认电脑和机器人处于同一局域网，且 `ssh ros-robot` 可连接。然后在 PowerShell 运行：

```powershell
cd D:\RosOrin
.\deploy_robot_autonomy.ps1 -Start
```

脚本只把 `robot/rosorin_autonomy` 上传到现有容器、执行单包 `colcon build` 并启动 launch，不安装 ROS 软件包。机器人端日志位于：

```text
/home/pi/docker/tmp/rosorin_autonomy.log
```

若只想更新和编译而不启动：

```powershell
.\deploy_robot_autonomy.ps1
```

## 启动电脑端

首次新增界面依赖时可复用现有环境：

```powershell
.\.venv-vla\Scripts\python.exe -m pip install -r .\requirements-vla.txt
```

启动本地 Qwen3-VL 和控制台：

```powershell
.\start_autonomy_console.ps1
```

如果机器人 IP 变化：

```powershell
.\start_autonomy_console.ps1 -RobotHost 192.168.1.17
```

只验证建图和 Nav2 界面、不加载模型：

```powershell
.\start_autonomy_console.ps1 -NoVlm
```

浏览器默认打开 `http://127.0.0.1:7860`。选择地图点后先检查坐标，再点击“发送目标”。“允许 AI 自动下发高层目标”必须由操作者明确勾选。

## ROS 接口

| 接口 | 类型 | 作用 |
| --- | --- | --- |
| `/vla/nav_goal` | `geometry_msgs/PoseStamped` | 发送 `map` 坐标目标 |
| `/vla/nav_command` | `std_msgs/String` JSON | STOP、PAUSE、RESUME、CANCEL、BACKUP、SPIN、SAVE_MAP |
| `/vla/navigation_status` | `std_msgs/String` JSON | 位姿、目标、距离、速度、IMU、卡住、打滑和地图状态 |
| `/autonomy/cmd_vel_smoothed` | `geometry_msgs/Twist` | Nav2 平滑后的速度，仅供碰撞监控输入 |
| `/controller/cmd_vel` | `geometry_msgs/Twist` | 碰撞监控批准后的最终底盘命令 |

## 验证

电脑端测试：

```powershell
.\.venv-vla\Scripts\python.exe -m unittest discover -s tests -v
```

部署后可做不移动小车的检查：

```powershell
ssh ros-robot "docker exec MentorPi bash -lc 'source /opt/ros/humble/setup.bash; source /home/ubuntu/ros2_ws/install/setup.bash; ros2 node list; ros2 topic echo /vla/navigation_status --once'"
```

首次实车测试建议先不勾选 AI 自动下发，只用很近的地图目标验证定位方向、停止按钮和倒车方向。虽然场地封闭且车小，0.16 m 碰撞边界仍应保留为最终硬约束。
