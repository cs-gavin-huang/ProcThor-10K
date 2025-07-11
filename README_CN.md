# ProcTHOR 数据收集与 VLM 评估框架

这是一个用于在 [AI2-THOR](https://ai2thor.allenai.org/) 模拟环境中，利用 [ProcTHOR-10k](https://prior.allenai.org/datasets/procthor) 数据集，进行数据收集和视觉语言模型（VLM）评估的框架。

## 核心功能

- **数据采集**: 从 ProcTHOR-10k 数据集的程序生成场景中采集图像和元数据。此功能由 `main.py` 中的 `run_sampling_loc` 函数驱动。
- **可配置采集**: 支持自定义要处理的房屋数量、每个房屋的采样点数量等。这些参数在 `main.py` 的 `collect` 命令中定义。
- **断点续采**: 能够从上次中断的地方继续采集，或从最后一个采集点重新开始 (`--resume`, `--restart_at_max` 标志)。
- **VLM 评估框架**: 提供了一个评估 VLM 在特定任务上表现的框架，例如：
  - **相对方向判断**: `main.py` 中的 `evaluate_door_direction_task` 函数。
  - **逐步导航**: `main.py` 中的 `evaluate_vlm_navigation_task` 函数。
- **模拟环境**: 基于 AI2-THOR 模拟器，提供逼真的室内环境。

## 项目结构

```
.
├── main.py                     # 主程序入口。定义 `collect` 和 `vlm_eval` 命令，并分别调用 `run_sampling_loc` 和 `run_custom_vlm_evaluation` 函数。
├── requirements.txt            # 项目依赖列表。
├── logs/                       # 日志文件输出目录。
├── experiment_sampling_loc/    # `collect` 命令采集的数据默认输出目录。
├── experiment_VLM_eval_object_affordance/ # VLM 评估结果的输出目录 (由 `src/vlm_eval/` 内的脚本使用)。
└── src/
    ├── config/                 # 配置文件。
    │   ├── experiment_config.py # 定义实验级配置，如数据采集的输出目录 `OUTPUT_DIR`。
    │   └── settings.py          # 定义 VLM 模型、Prompts、Affordance 真值等详细配置。
    ├── experiments/            # 实验逻辑。
    │   └── house_collect.py    # `collect` 命令的核心实现，定义了 `HouseCollectExperiment` 类。
    ├── utils/                  # 通用工具函数。
    └── vlm_eval/               # VLM 评估相关模块。注意：此目录下的代码未被 main.py 的 vlm_eval 命令直接调用。
        ├── main.py            # 一个独立的 VLM 评估入口点。
        ├── models/            # VLM 模型客户端实现。
        ├── data/              # 数据处理逻辑。
        └── utils/             # 评估工具函数。
```

## 安装

1.  **先决条件**:
    *   Python 3.8+
    *   在 Linux 环境下，`ai2thor` 可能需要一个 X server。如果在无头服务器上运行，可以使用 `Xvfb`。

2.  **安装依赖**:
    在项目根目录下运行以下命令来安装所有必需的 Python 包：
    ```bash
    pip install -r requirements.txt
    ```

## 使用流程

工作流程主要分为两步：**数据采集** 和 **VLM 评估**。

### 第一步：数据采集 (`collect`)

此步骤从 ProcTHOR-10k 数据集中加载场景，并采集图像及元数据。

**命令示例**:
```bash
# 采集5个房屋的数据，每个房屋内采样10个随机视点
python main.py collect --num_houses 5 --samples_per_house 10
```

**命令行参数**:
- `--num_houses [整数]`: 要处理的房屋数量。`0` 表示 `train` 分割中的所有房屋。
- `--samples_per_house [整数]`: 每个房屋中采集的随机视角数量。
- `--resume`: 从下一个房屋继续采集。
- `--restart_at_max`: 从已存在的最高索引房屋重新开始采集。

**输出**:
采集的数据保存在 `experiment_sampling_loc/` 目录下。该路径由 `src/config/experiment_config.py` 文件中的 `OUTPUT_DIR` 变量定义。

### 第二步：VLM 评估 (`vlm_eval`)

此步骤利用第一步采集的数据来评估 VLM。

**重要提示**:
> `vlm_eval` 命令当前调用的是 `main.py` 中的 `run_custom_vlm_evaluation` 函数，这是一个**演示性**实现。此函数包含**硬编码的示例路径**并且调用了一个**模拟的（Mocked）VLM**，需要根据实际需求进行修改。

**如何运行**:
```bash
python main.py vlm_eval
```

**自定义评估的修改点**:

1.  **替换模拟的VLM**:
    模拟的 VLM 逻辑位于 `main.py` 的 `get_vlm_response` 函数中。需要将此函数的实现替换为真实的 VLM 模型 API 调用。

2.  **修改评估逻辑**:
    `main.py` 的 `run_custom_vlm_evaluation` 函数中硬编码了评估用的房屋 ID（变量 `example_house_id`）。需要修改此函数以遍历 `collect` 命令生成的实际数据集，并对每个任务运行评估。

**注意**:
`vlm_eval` 命令通过 `argparse` 定义了 `--models` 和 `--sample_mode` 等参数，但这些参数在 `main.py` 的 `run_custom_vlm_evaluation` 函数中**未被实际使用**。可按需在修改该函数时利用这些参数。


