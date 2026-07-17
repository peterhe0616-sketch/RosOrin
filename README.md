# ROSOrin 机器人多传感器融合与实时三维感知

ROSOrin 是一个运行在 Windows 电脑端的机器人感知与控制原型。它不要求电脑安装 ROS 2，而是通过 rosbridge WebSocket 和 HTTP/MJPEG 接收机器人数据，将相机、LD19 LiDAR、IMU 与里程计融合为二维监控界面和实时彩色三维点云。

## 截至目前的成果

- 接入相机、LiDAR、IMU、相机标定信息和 `/odom` 里程计数据
- 完成带畸变矫正的相机画面与 LiDAR 俯视图实时仪表盘
- 支持全局 `W/A/S/D` 组合键发布 `/controller/cmd_vel` 控制机器人
- 集成 Depth Anything V2 Small，实现单目相对深度估计
- 使用 LD19 实测距离对单目深度进行鲁棒尺度与偏移校正
- 将 RGB-D 点云、LiDAR 点和里程计位姿统一到机器人/固定场景坐标系
- 实现滚动局部点云、关键帧增量体素地图和 Open3D 实时三维显示
- 支持 Intel Arc GPU 上的 OpenVINO FP16 推理，并可切换 CPU/GPU
- 提供 YAML 参数化配置、无窗口性能测试和 RGB/深度/LiDAR 调试预览

## 技术栈

| 方向 | 技术 |
| --- | --- |
| 语言与平台 | Python 3、Windows、PowerShell |
| 机器人通信 | ROS 2 topics、rosbridge、roslibpy、WebSocket、HTTP/MJPEG |
| 视觉与数值计算 | OpenCV、NumPy |
| 单目深度 | Depth Anything V2 Small、ONNX FP16 |
| AI 推理加速 | OpenVINO、Intel Arc GPU / CPU |
| 三维处理与可视化 | Open3D、点云反投影、体素降采样、关键帧地图 |
| 多传感器融合 | 相机内外参、LiDAR 投影、鲁棒仿射深度标定、里程计坐标变换 |
| 配置 | YAML |

## 数据流

```text
Robot ROS 2
 ├─ CameraInfo ───────────────┐
 ├─ LD19 LaserScan ───────────┼─ rosbridge ─┐
 ├─ IMU / Odometry ───────────┘             │
 └─ Camera Image ─────────────── HTTP/MJPEG ┤
                                            ▼
                         OpenCV 畸变矫正 + Depth Anything V2
                                            │
                              LiDAR 公制尺度/偏移校正
                                            │
                         RGB 点云 + LiDAR + Odometry 坐标变换
                                            │
                         Open3D 局部点云 / 固定场景体素地图
```

## 项目结构

- `receive_robot_sensors.py`：二维传感器仪表盘、键盘控制及三维模块入口
- `realtime_sparse_3d.py`：深度推理、LiDAR 校正、点云融合和 Open3D 显示
- `fusion_config.yaml`：模型、深度、点云、地图及传感器外参配置
- `README-3D.md`：三维融合的运行方式、参数和标定说明
- `README-VLA.md`：Qwen3-VL INT4 安装、基准、语义驾驶与安全模式
- `requirements.txt` / `requirements-3d.txt`：基础与三维环境依赖
- `models/depth-anything-v2-small/`：Depth Anything V2 Small 配置；ONNX FP16 权重为本地依赖，不纳入 Git

## Qwen3-VL 语义驾驶

电脑端可通过 OpenVINO 运行 Qwen3-VL-4B INT4，把相机画面、LiDAR 扇区距离和中文任务转换成受限的高级动作建议。先运行 `setup_vla.ps1` 安装并转换模型，再按照 `README-VLA.md` 从观察模式逐步验证到封闭场地低速自动模式。

实时稀疏三维视线模型见 `README-3D.md`。

这个程序在 Windows 上直接接收机器人三路实时数据，无需安装 ROS 2：

- 摄像头：HTTP/MJPEG，显示在 OpenCV 窗口左侧
- LiDAR：rosbridge WebSocket，转换成俯视散点图显示在窗口右侧
- IMU：rosbridge WebSocket，topic `/imu`
- 控制：全局 `W/A/S/D` 按键发布 `/controller/cmd_vel`

## 安装

在 PowerShell 中进入本目录：

```powershell
cd D:\RosOrin
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可不激活环境，直接执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py
```

雷达图以机器人前方朝上显示，包含米制距离环、左右方向、距离着色和最近障碍物标记。按窗口中的 `Esc` 或终端中的 `Ctrl+C` 退出。

默认控制速度：

- `W`：以 `0.2 m/s` 前进
- `S`：以 `0.2 m/s` 后退
- `A`：`angular.z = +0.69 rad/s`
- `D`：`angular.z = -0.69 rad/s`
- 支持 `W+A`、`W+D`、`S+A`、`S+D` 组合

松开按键时发布一次零速度。程序遵照当前配置，不处理窗口失焦、空格停车或退出停车。

摄像头默认订阅 `/ascamera/camera_publisher/rgb0/camera_info`，使用相机的 `K`、`D` 参数实时矫正桶形畸变。

窗口顶部提供 `Distortion` 滑块，范围为 `-10 倍` 到 `+10 倍`。滑块显示值为 `0–2000`：

- `1000`：不做畸变矫正
- `1100`：使用相机标定参数的标准 1 倍矫正（默认）
- `2000`：同方向 10 倍强矫正
- `0`：反方向 10 倍强矫正，用于枕形/桶形反向调节

也可以从命令行指定滑块初值：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --distortion-strength 0.75
```

修改移动速度：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --linear-speed 0.15 --angular-speed 0.5
```

关闭键盘控制：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --no-control
```

显示未矫正的原始图像：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --no-undistort
```

不打开视频，只显示 LiDAR：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --no-video
```

调整雷达显示半径，例如显示 10 米：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --lidar-range 10
```

运行 10 秒后自动退出：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --headless --no-video --duration 10
```

机器人 IP 改变时：

```powershell
.\.venv\Scripts\python.exe .\receive_robot_sensors.py --host 192.168.3.17
```
