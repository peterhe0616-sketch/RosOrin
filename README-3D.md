# ROSOrin 实时稀疏三维视线模型

该程序在 Windows 电脑上组合以下数据：

- USB 单目摄像头：Depth Anything V2 Small 单目深度
- LD19：为单目深度提供公制尺度与偏移校正
- `/odom`：对齐最近 1.5 秒的点云
- Open3D：显示以车体为中心、向前滚动的稀疏彩色点云

完整界面默认还会订阅 `/odom`，把关键帧点云融合到固定场景坐标系中：地面和世界坐标轴保持不动，蓝色小车模型按照里程计位姿移动。

模型使用 OpenVINO 在 Intel Arc GPU 上运行。

## 文件

- `realtime_sparse_3d.py`：实时融合与三维显示
- `fusion_config.yaml`：模型、点云密度和传感器外参
- `requirements-3d.txt`：独立三维环境依赖
- `models/depth-anything-v2-small/onnx/model_fp16.onnx`：深度模型

## 运行

完整界面（畸变矫正滑块、二维激光地图、WASD 和当前帧稠密彩色三维点图）：

```powershell
cd D:\RosOrin
.\.venv3d\Scripts\python.exe .\receive_robot_sensors.py
```

不需要三维窗口时添加 `--no-3d`。

原来的滚动稀疏三维程序仍可单独运行：

```powershell
cd D:\RosOrin
.\.venv3d\Scripts\python.exe .\realtime_sparse_3d.py
```

Open3D 窗口操作：

- 默认视角固定为以小车为中心的俯后视角，地面保持在画面下方
- 鼠标拖动：沿地平面平移视野，不旋转相机
- 鼠标滚轮：缩放
- 蓝色车体、黄色箭头和 `ROBOT` 标签：小车当前位置及前进方向
- 关闭窗口或在终端按 `Ctrl+C`：退出

同时显示 RGB、预测公制深度和 LD19 投影点：

```powershell
.\.venv3d\Scripts\python.exe .\realtime_sparse_3d.py --preview
```

仅测试融合和性能，不打开三维窗口：

```powershell
.\.venv3d\Scripts\python.exe .\realtime_sparse_3d.py --no-window --max-frames 30
```

强制选择推理设备：

```powershell
.\.venv3d\Scripts\python.exe .\realtime_sparse_3d.py --device GPU
.\.venv3d\Scripts\python.exe .\realtime_sparse_3d.py --device CPU
```

## 当前默认参数

- 推理尺寸：518 × 392
- 单帧最大采样点：7,000
- Open3D 点大小：8 px（可在 `fusion_config.yaml` 的 `cloud.point_size_px` 调整）
- 当前帧点图采样步长：4 px，通常约 1.6万–1.9万个 RGB 点
- 固定场景地图体素：5 cm
- 地图关键帧阈值：平移 6 cm 或旋转 3°
- 地图点数上限：250,000
- 深度范围：0.3–8 m
- 体素尺寸：8 cm
- 历史窗口：1.5 秒
- 尺度平滑系数：0.18
- 推理设备：Intel Arc GPU

可在 `fusion_config.yaml` 中修改这些参数。

## 外参说明

当前 `camera_to_base` 和 `lidar_to_camera` 来自机器人 URDF 的安装位置：

- LiDAR：`base_link` 前方约 1.15 cm、高约 13.64 cm
- 相机：`base_link` 前方约 5.74 cm、高约 9.19 cm

USB 摄像头没有发布完整的相机—雷达 TF，因此这些数值只是初始估计。调试窗口中的红点是 LD19 投影位置：若红点没有落在激光实际命中的物体边缘，应先标定并替换 `lidar_to_camera`，否则单目尺度校正和三维轮廓都会产生系统误差。

## 输出指标

终端每秒输出：

```text
fps=7.6 inference=63ms lidar_in_view=86 matches=86 residual=0.115m points=3883
```

- `inference`：单目深度推理耗时
- `lidar_in_view`：投影进相机视野的 LD19 点数
- `matches`：用于深度校正的匹配数
- `residual`：雷达与校正后深度的中位拟合误差
- `points`：体素降采样后的滚动点云数量

如果 `matches` 长期低于 10 或 `residual` 明显高于 0.2 m，优先检查外参和传感器时间差。
