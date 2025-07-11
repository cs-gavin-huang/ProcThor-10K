import os
import subprocess
import datetime

# Experiment configuration
EXPERIMENT_NAME = "house_collect"
OUTPUT_DIR = "experiment_sampling_loc"
NUM_SAMPLES = 10
IMG_WIDTH = 640
IMG_HEIGHT = 640
IMG_QUALITY = "High WebGL"
VLM_EVAL_OBJECT_AFFORDANCE = "VLM_eval_object_affordance"

def get_env_info():
    """Get environment information for logging."""
    info = []
    info.append(f"Experiment name: {EXPERIMENT_NAME}")
    info.append(f"Output directory: {OUTPUT_DIR}")
    info.append(f"Number of samples: {NUM_SAMPLES}")
    info.append(f"Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    info.append(f"Image quality: {IMG_QUALITY}")
    info.append(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        conda_env = subprocess.check_output("conda info --envs | grep '*' | awk '{print $1}'", shell=True).decode().strip()
        info.append(f"Conda environment: {conda_env}")
    except:
        info.append("Conda environment: Unknown")
    return "\n".join(info) 