# Qwen3-VL 电脑端语义驾驶

该模块在 Windows 电脑上运行 Qwen3-VL-4B-Instruct INT4，通过 OpenVINO 使用 Intel CPU/GPU。它低频理解相机画面和自然语言任务，现有 LiDAR/深度模块继续负责高频几何感知。

Qwen3-VL 的输出是不可信的高级动作建议。`semantic_supervisor.py` 会检查结果时效、置信度和 LiDAR 距离，并把速度限制在 `vla_config.yaml` 的安全范围内。

新版连续自主导航将模型改为 1 Hz 场景采样和有界多轮 Chat 记忆，并由 Nav2 执行连续路径。入口为 `autonomy_console.py`，部署方式见 [README-AUTONOMY.md](README-AUTONOMY.md)。

## 模式

- `--vla`：仅观察，显示模型输出，不控制小车。
- `--assist`：显示经过安全策略过滤后的建议，不自动发布速度。
- `--auto`：发布经过安全策略过滤的低速命令；按住任意 `W/A/S/D` 会立即人工覆盖。

只有显式传入 `--auto` 才允许模型参与控制。配置文件中的 `mode` 不会自动打开控制。

## 安装模型

在 PowerShell 中运行：

```powershell
cd D:\RosOrin
.\setup_vla.ps1
```

脚本执行以下步骤：

1. 使用 Python 3.11 创建 `.venv-vla`。
2. 安装 OpenVINO、OpenVINO GenAI、Optimum Intel 和转换依赖。
3. 使用 `hf download` 下载 `Qwen/Qwen3-VL-4B-Instruct`。
4. 将权重转换成 OpenVINO INT4 IR，写入 `models/qwen3-vl-4b-int4-ov`。

若源模型已经下载，可使用 `-SkipDownload`。若转换模型已经存在，可使用 `-SkipConvert`。

## 本机基准

先用合成道路画面测试 GPU：

```powershell
.\.venv-vla\Scripts\python.exe .\benchmark_vla.py --device GPU
```

再比较 CPU：

```powershell
.\.venv-vla\Scripts\python.exe .\benchmark_vla.py --device CPU
```

也可传入真实相机截图：

```powershell
.\.venv-vla\Scripts\python.exe .\benchmark_vla.py --image .\scene.jpg --device GPU
```

根据基准结果修改 `vla_config.yaml` 的 `device`、`interval_s`、`image_max_side` 和 `max_new_tokens`。

### 当前电脑实测

Core Ultra 5 125H、32 GB 内存、Intel Arc 核显，OpenVINO 2026.2.1：

| 设备 | 输入/输出限制 | 加载 | 单次推理 |
| --- | --- | ---: | ---: |
| Intel Arc GPU | 448×308 / 96 token | 约 9.24 s（有编译缓存） | 约 6.39 s（首次严格决策测试） |
| CPU | 448×308 / 96 token | 约 4.67 s | 约 14.23 s |

接入真实相机后，GPU 热态决策约 3.3–5.3 秒；同时摄像头约 10.7 FPS、LiDAR 约 9.7 FPS、IMU 约 42 FPS。默认因此选择 `GPU` 和 `image_max_side: 448`。

VLA 与稠密三维点图可以同时运行，但会争用同一块 Intel GPU。联合测试中深度推理延迟波动明显增大；低速自动实验优先使用 `--no-3d`，需要三维观察或建图时再启用完整界面。

## 运行顺序

### 1. 被动观察

```powershell
.\.venv-vla\Scripts\python.exe .\receive_robot_sensors.py --vla
```

若只验证 VLM、希望减少 GPU 争用，可额外传入 `--no-3d`。

### 2. 辅助建议

```powershell
.\.venv-vla\Scripts\python.exe .\receive_robot_sensors.py --assist --no-3d `
  --vla-instruction "沿走廊慢速前进，遇到人停车"
```

### 3. 封闭场地低速自动模式

确认观察与辅助模式输出稳定、急停可用且小车架空测试通过后，才运行：

```powershell
.\.venv-vla\Scripts\python.exe .\receive_robot_sensors.py --auto --no-3d `
  --vla-instruction "持续沿可通行区域流畅行驶，接近障碍时从更开阔的一侧绕行，只有无法通过时停车"
```

自动模式下：

- 任意 WASD 键优先于模型命令。
- 模型加载、报错、结果超时或置信度不足时速度为零。
- LiDAR 前方扇区没有有效数据时，禁止任何自动运动。
- 前方小于 `stop_distance_m` 时，前进和带前进分量的转向会被否决。
- 左/右空间不足时，对应转向会被否决。
- 最近一次通过安全检查的非零动作会连续保持到下一次 VLM 更新，最长 `max_command_duration_s`，当前为 12 秒；LiDAR 安全检查仍以约 20 Hz 持续运行并可立即归零。
- 松开人工 WASD 后有 1 秒自动控制冷却期。

当前前方停车距离 `stop_distance_m` 和左右转向净空 `turn_clearance_m` 均为 `0.16 m`。
- 程序退出会发布一次零速度。

## 输出协议

模型必须返回一个 JSON 对象：

```json
{
  "scene": "室内走廊，前方有纸箱",
  "hazards": ["前方纸箱"],
  "action": "SLOW_RIGHT",
  "target_heading_deg": -15,
  "max_speed_mps": 0.05,
  "confidence": 0.82,
  "reason": "右侧空间更宽"
}
```

允许动作：`STOP`、`HOLD`、`FORWARD`、`SLOW_FORWARD`、`TURN_LEFT`、`TURN_RIGHT`、`SLOW_LEFT`、`SLOW_RIGHT`。

## 测试

```powershell
.\.venv-vla\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖输出解析、异步工作线程、结果超时、置信度阈值、LiDAR 扇区、前方障碍否决和速度边界。

## 文件

- `qwen_vl_runtime.py`：OpenVINO VLM 后端、提示词、异步队列和状态。
- `semantic_supervisor.py`：输出验证、LiDAR 扇区和确定性安全策略。
- `vla_config.yaml`：模型、推理周期、指令和安全阈值。
- `benchmark_vla.py`：CPU/GPU 本机性能基准。
- `setup_vla.ps1`：下载并转换模型。
- `tests/test_semantic_supervisor.py`：不依赖真实小车和大模型的安全测试。

## 限制

Qwen3-VL 是通用视觉语言模型，不是经过本车数据训练的端到端 VLA。当前实现适合语义观察、低频任务理解和有硬安全约束的实验控制，不是无人驾驶安全认证系统。单前置相机也无法观察侧后方盲区。
