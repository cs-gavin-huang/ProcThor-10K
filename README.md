# ProcTHOR Data Collection and VLM Evaluation Framework

This is a framework for data collection and Vision Language Model (VLM) evaluation in the [AI2-THOR](https://ai2thor.allenai.org/) simulation environment, specifically using the [ProcTHOR-10k](https://prior.allenai.org/datasets/procthor) dataset.

## Core Features

- **Data Collection**: Collects images and metadata from procedurally generated scenes in the ProcTHOR-10k dataset. This functionality is driven by the `run_sampling_loc` function in `main.py`.
- **Configurable Collection**: Supports customizing the number of houses to process, the number of sample points per house, etc. These parameters are defined in the `collect` command in `main.py`.
- **Resumable Collection**: Can resume collection from the last interruption or restart from the last collected point (using `--resume` and `--restart_at_max` flags).
- **VLM Evaluation Framework**: Provides a framework for evaluating VLM performance on specific tasks, such as:
  - **Relative Direction Judgment**: The `evaluate_door_direction_task` function in `main.py`.
  - **Step-by-Step Navigation**: The `evaluate_vlm_navigation_task` function in `main.py`.
- **Simulation Environment**: Provides a realistic indoor environment based on the AI2-THOR simulator.

## Project Structure

```
.
├── main.py                     # Main entry point. Defines `collect` and `vlm_eval` commands, which call `run_sampling_loc` and `run_custom_vlm_evaluation` respectively.
├── requirements.txt            # Project dependency list.
├── logs/                       # Output directory for log files.
├── experiment_sampling_loc/    # Default output directory for data collected by the `collect` command.
├── experiment_VLM_eval_object_affordance/ # Output directory for VLM evaluation results (used by scripts within `src/vlm_eval/`).
└── src/
    ├── config/                 # Configuration files.
    │   ├── experiment_config.py # Defines experiment-level configurations, such as the output directory `OUTPUT_DIR` for data collection.
    │   └── settings.py          # Defines detailed configurations for VLM models, prompts, affordance ground truth, etc.
    ├── experiments/            # Experiment logic.
    │   └── house_collect.py    # Core implementation of the `collect` command, defines the `HouseCollectExperiment` class.
    ├── utils/                  # Common utility functions.
    └── vlm_eval/               # VLM evaluation related modules. Note: The code in this directory is not directly called by the vlm_eval command in main.py.
        ├── main.py            # A separate entry point for VLM evaluation.
        ├── models/            # VLM model client implementations.
        ├── data/              # Data processing logic.
        └── utils/             # Evaluation utility functions.
```

## Installation

1.  **Prerequisites**:
    *   Python 3.8+
    *   On Linux, `ai2thor` may require an X server. If running on a headless server, you can use `Xvfb`.

2.  **Install Dependencies**:
    Run the following command in the project root directory to install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Workflow

The workflow is primarily divided into two steps: **Data Collection** and **VLM Evaluation**.

### Step 1: Data Collection (`collect`)

This step loads scenes from the ProcTHOR-10k dataset and collects images and metadata.

**Example Command**:
```bash
# Collect data from 5 houses, sampling 10 random viewpoints in each.
python main.py collect --num_houses 5 --samples_per_house 10
```

**Command-line Arguments**:
- `--num_houses [integer]`: The number of houses to process. `0` means all houses in the `train` split.
- `--samples_per_house [integer]`: The number of random viewpoints to sample in each house.
- `--resume`: Resume collection from the next house.
- `--restart_at_max`: Restart collection from the highest-indexed existing house.

**Output**:
The collected data is saved in the `experiment_sampling_loc/` directory. This path is defined by the `OUTPUT_DIR` variable in the `src/config/experiment_config.py` file.

### Step 2: VLM Evaluation (`vlm_eval`)

This step uses the data collected in the first step to evaluate a VLM.

**Important Note**:
> The `vlm_eval` command currently calls the `run_custom_vlm_evaluation` function in `main.py`, which is a **demonstrative** implementation. This function contains **hard-coded example paths** and calls a **mocked VLM**. It needs to be modified for actual use.

**How to Run**:
```bash
python main.py vlm_eval
```

**Points for Customization**:

1.  **Replace the Mocked VLM**:
    The mocked VLM logic is located in the `get_vlm_response` function in `main.py`. The implementation of this function needs to be replaced with actual VLM model API calls.

2.  **Modify the Evaluation Logic**:
    The `run_custom_vlm_evaluation` function in `main.py` has a hard-coded house ID for evaluation (the `example_house_id` variable). This function needs to be modified to iterate over the actual dataset generated by the `collect` command and run the evaluation for each task.

**Note**:
The `vlm_eval` command defines parameters such as `--models` and `--sample_mode` via `argparse`, but these are **not actually used** in the current `run_custom_vlm_evaluation` function in `main.py`. They can be utilized when modifying the function as needed. 