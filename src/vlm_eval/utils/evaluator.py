import datetime
import json
import random
from pathlib import Path
from typing import Dict, List, Any

from src.config.settings import EXPERIMENT_CONFIG, OUTPUT_CONFIG
from src.vlm_eval.models.auto_vlm import get_vlm_client


class AffordanceEvaluator:
    def __init__(self, model_name: str, output_dir: Path):
        self.model = get_vlm_client(model_name)
        self.output_dir = Path(output_dir)

    def evaluate_object(
        self,
        obj_info: Dict,
        image_path: Path,
        house_name: str,
        loc_name: str,
        view_id: str,
    ) -> Dict[str, Any]:
        eval_start_time = datetime.datetime.now().isoformat()
        evaluations_data = {}

        for group_name, group_data in EXPERIMENT_CONFIG["prompt_groups"].items():
            for prop_to_eval in group_data["properties"]:
                if prop_to_eval not in obj_info:
                    # This warning is helpful if a property is defined in a group 
                    # but its ground truth is missing for a specific object.
                    print(
                        f"Warning: Ground truth for property '{prop_to_eval}' not found "
                        f"in object_info for {obj_info.get('objectId', obj_info['objectType'])}. "
                        f"Skipping this property in group '{group_name}'."
                    )
                    continue

                ground_truth_for_prop = obj_info[prop_to_eval]

                for prompt_key, prompt_template_str in group_data["prompts"].items():
                    # --- Prepare format arguments ---
                    agent_type = random.choice(EXPERIMENT_CONFIG["agent_types"])
                    distance = obj_info.get("distance", "N/A")
                    object_type = obj_info["objectType"]

                    format_args = {
                        "distance": distance,
                        "object_type": object_type,
                        "agent_type": agent_type,
                        "property": prop_to_eval, # Included by default
                    }

                    # Task-specific argument adjustments
                    if group_name == "identification":
                        format_args.pop("object_type", None)
                        format_args.pop("property", None)
                    elif group_name == "state":
                        format_args.pop("property", None)

                    # Only format with keys present in the prompt string
                    prompt_text = prompt_template_str.format(**{k: v for k, v in format_args.items() if f"{{{k}}}" in prompt_template_str})

                    # --- VLM Call and Error Handling ---
                    model_raw_prediction = "ERROR_IN_EVALUATION"
                    error_message = None
                    try:
                        model_raw_prediction = self.model.chat_completion(prompt_text, str(image_path))
                        model_raw_prediction = model_raw_prediction.strip().lower()
                    except Exception as e:
                        error_message = str(e)
                        print(
                            f"Error during VLM evaluation for object {obj_info.get('objectId', obj_info['objectType'])} "
                            f"with prompt {group_name}_{prompt_key}: {error_message}"
                        )

                    # --- Store Evaluation Result ---
                    # The unique key for results now combines group, prompt, and property
                    current_eval_key = f"{group_name}_{prompt_key}_{prop_to_eval}"
                    
                    evaluations_data[current_eval_key] = {
                        "prompt_group": group_name,
                        "prompt_key": prompt_key,
                        "property_evaluated": prop_to_eval,
                        "prompt_text_generated": prompt_text,
                        "model_answer": model_raw_prediction,
                        "ground_truth": ground_truth_for_prop,
                    }
                    if error_message:
                        evaluations_data[current_eval_key]["error_message"] = error_message

        eval_end_time = datetime.datetime.now().isoformat()

        evaluation_result = {
            "model_id": self.model.model_id
            if hasattr(self.model, "model_id")
            else str(self.model.__class__.__name__),
            "house": house_name,
            "location": loc_name,
            "view_id": view_id,
            "object_id": obj_info.get("objectId", "OBJECT_ID_MISSING"),
            "object_type": obj_info["objectType"],
            "image_path": str(image_path),
            "evaluation_start_time": eval_start_time,
            "evaluation_end_time": eval_end_time,
            "evaluations": evaluations_data,
        }
        return evaluation_result

    def save_result(self, result_data: Dict, output_file_path: Path):
        """Saves a single object's evaluation result to a JSON file."""
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(result_data, f, indent=4)

