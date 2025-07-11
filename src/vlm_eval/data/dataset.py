from pathlib import Path
import json
from typing import Dict, List, Any
import random
from src.config.settings import EXPERIMENT_CONFIG

class AffordanceDataset:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        
    def get_house_paths(self) -> List[Path]:
        """获取所有房屋路径"""
        return [p for p in self.base_dir.glob("*") if p.is_dir()]
    
    def get_location_paths(self, house_path: Path) -> List[Path]:
        """获取指定房屋的所有位置路径"""
        return [p for p in house_path.glob("*") if p.is_dir()]
    
    def get_view_files(self, loc_path: Path) -> List[Path]:
        """获取指定位置的所有包含场景实体列表的JSON文件"""
        return [p for p in loc_path.glob("view_*.json")]
    
    def load_view_data(self, view_file: Path) -> List[Dict[str, Any]]:
        """加载视角数据 (现在是实体列表)"""
        try:
            with open(view_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"Warning: Data in {view_file} is not a list as expected. Returning empty list.")
                    return []
                return data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {view_file}. Returning empty list.")
            return []
        except Exception as e:
            print(f"Error loading view data from {view_file}: {e}. Returning empty list.")
            return []
    
    def get_image_path(self, view_file: Path, view_id: int) -> Path:
        """获取对应的图片路径 (view_file is view_*.json)"""
        return view_file.parent / f"view_{view_id}.png"
    
    def get_agent_state_for_view(self, view_file: Path) -> Dict:
        """从对应的meta.json文件中加载并返回agent_state。"""
        # view_*.json -> _meta.json
        meta_filename = view_file.name.replace("view_", "").replace(".json", "_meta.json")
        meta_filepath = view_file.parent / meta_filename
        
        if not meta_filepath.exists():
            # Fallback for older data structures if needed
            # For now, just warn and return empty
            print(f"Warning: Corresponding meta file not found at {meta_filepath}")
            return {}
            
        try:
            with open(meta_filepath, "r") as f:
                meta_data = json.load(f)
                return meta_data.get("agent_state", {})
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading or parsing agent state from {meta_filepath}: {e}")
            return {}
    
    def sample_objects(self, objects: List[Dict], sample_size: int) -> List[Dict]:
        """采样指定数量的物品"""
        if sample_size == "full":
            return objects
        return random.sample(objects, min(sample_size, len(objects)))
    
    def format_prompt(self, obj: Dict, prop: str, version: str, agent_type: str = None) -> str:
        """格式化prompt"""
        prompt_template = EXPERIMENT_CONFIG["prompt_versions"][version]
        return prompt_template.format(
            distance=obj.get("distance", "N/A"),
            object_type=obj["objectType"],
            property=prop,
            agent_type=agent_type or random.choice(EXPERIMENT_CONFIG["agent_types"])
        ) 