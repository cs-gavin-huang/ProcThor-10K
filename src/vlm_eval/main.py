from pathlib import Path
from typing import List, Dict, Set
import json
import random
from tqdm import tqdm
from collections import defaultdict
import re
import numpy as np
from src.config.settings import (
    VLM_MODELS,
    EXPERIMENT_CONFIG,
    OUTPUT_CONFIG,
    TARGET_INTERACTABLE_OBJECT_TYPES,
    GROUND_TRUTH_AFFORDANCES
)
from .data.dataset import AffordanceDataset
from .utils.evaluator import AffordanceEvaluator
from .models.auto_vlm import get_vlm_client
from .models.vlm_client import OllamaClient

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    # Remove or replace invalid characters
    name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    # Ensure it's not too long (optional, depending on filesystem limits)
    return name[:100] # Limit length

def are_agent_and_object_adjacent(agent_pos: dict, object_bbox_corners: list, grid_size: float) -> bool:
    """
    Checks if the agent's grid cell is adjacent to any grid cell occupied by the object's bounding box.
    """
    if not agent_pos or not object_bbox_corners:
        return False

    # Convert agent position to grid coordinates
    agent_grid_x = int(agent_pos['x'] / grid_size)
    agent_grid_y = int(agent_pos['y'] / grid_size)
    agent_grid_z = int(agent_pos['z'] / grid_size)

    # Determine the grid cells occupied by the object's bounding box
    bbox_points = np.array([[p['x'], p['y'], p['z']] for p in object_bbox_corners])
    min_coords = np.min(bbox_points, axis=0)
    max_coords = np.max(bbox_points, axis=0)

    obj_min_grid_x = int(min_coords[0] / grid_size)
    obj_max_grid_x = int(max_coords[0] / grid_size)
    obj_min_grid_y = int(min_coords[1] / grid_size)
    obj_max_grid_y = int(max_coords[1] / grid_size)
    obj_min_grid_z = int(min_coords[2] / grid_size)
    obj_max_grid_z = int(max_coords[2] / grid_size)

    # Check for adjacency. The agent is adjacent if its grid cell is within 1 unit
    # of the object's bounding box grid space.
    # Check X dimension
    closest_x = max(obj_min_grid_x, min(agent_grid_x, obj_max_grid_x))
    # Check Y dimension
    closest_y = max(obj_min_grid_y, min(agent_grid_y, obj_max_grid_y))
    # Check Z dimension
    closest_z = max(obj_min_grid_z, min(agent_grid_z, obj_max_grid_z))

    distance_vector = np.array([agent_grid_x - closest_x, agent_grid_y - closest_y, agent_grid_z - closest_z])
    
    # Using Chebyshev distance (max coordinate difference) for grid adjacency
    grid_distance = np.max(np.abs(distance_vector))

    # Adjacent if distance is 0 (overlapping) or 1 (touching)
    return grid_distance <= 1

def main(args):
    dataset = AffordanceDataset(args.input_dir)
    all_samples = []

    print("Starting to collect and process samples...")
    for house_path in tqdm(dataset.get_house_paths(), desc="Processing Houses"):
        for loc_path in tqdm(dataset.get_location_paths(house_path), desc=f"  Locs in {house_path.name}", leave=False):
            for view_file in tqdm(dataset.get_view_files(loc_path), desc=f"    Views in {loc_path.name}", leave=False):
                try:
                    view_id_str = view_file.stem.split('_')[-1]
                    view_id_for_eval = int(view_id_str)
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse view ID from filename {view_file.name}. Skipping this file.")
                    continue

                image_path = dataset.get_image_path(view_file, view_id_for_eval)
                scene_entities_list = dataset.load_view_data(view_file)

                if not scene_entities_list: # Potentially empty if load_view_data returned [] due to error/structure
                    # print(f"Debug: No scene entities loaded from {view_file.name}")
                    continue

                for entity_index, entity_data in enumerate(scene_entities_list):
                    if not isinstance(entity_data, dict):
                        print(f"Warning: Entity at index {entity_index} in {view_file.name} is not a dictionary. Skipping.")
                        continue

                    entity_name = entity_data.get("name")
                    if not entity_name:
                        # Allow processing unnamed entities if TARGET_INTERACTABLE_OBJECT_TYPES is empty and user wants to define a default obj type
                        # For now, skipping unnamed as objectType inference relies on it.
                        # print(f"Debug: Entity at index {entity_index} in {view_file.name} has no 'name'. Skipping.")
                        continue
                        
                    object_type = entity_name.split('|')[0]

                    # Filter by TARGET_INTERACTABLE_OBJECT_TYPES if the set is not empty
                    if TARGET_INTERACTABLE_OBJECT_TYPES and object_type not in TARGET_INTERACTABLE_OBJECT_TYPES:
                        # print(f"Debug: Skipping entity {entity_name} (type: {object_type}) as it's not in TARGET_INTERACTABLE_OBJECT_TYPES.")
                        continue
                    
                    # Construct obj_data for evaluation
                    obj_for_eval = {
                        "objectId": entity_name, # Using the full name as a unique ID for now
                        "objectType": object_type,
                        "original_entity_data": entity_data, # Store original data for reference if needed
                        "agent_state_at_view": dataset.get_agent_state_for_view(view_file) # Pass agent state
                    }

                    # Populate ground truth affordances
                    # Get affordances for this specific object_type, or fall back to "Default"
                    gt_affordances_for_type = GROUND_TRUTH_AFFORDANCES.get(
                        object_type, GROUND_TRUTH_AFFORDANCES.get("Default", {})
                    )
                    
                    found_any_gt = False
                    for prop in EXPERIMENT_CONFIG["affordance_properties"]:
                        if prop.startswith("reachable_"):
                            instance_data = obj_for_eval.get("original_entity_data", {})
                            agent_state = obj_for_eval.get("agent_state_at_view", {})
                            bbox_corners = instance_data.get("objectOrientedBoundingBox", {}).get("cornerPoints")
                            
                            if agent_state and bbox_corners:
                                # Using a grid size of 0.25 as defined in the experiment
                                grid_size = 0.25 
                                obj_for_eval[prop] = are_agent_and_object_adjacent(agent_state.get("position"), bbox_corners, grid_size)
                            else:
                                # Fallback if critical data is missing
                                obj_for_eval[prop] = False
                            found_any_gt = True
                        elif prop == "identification":
                            # For identification, the ground truth is the object's type itself.
                            obj_for_eval[prop] = object_type
                            found_any_gt = True
                        elif prop == "state_isOpen":
                            # For state, prioritize the real-time state from the instance data.
                            instance_data = obj_for_eval.get("original_entity_data", {})
                            if "isOpen" in instance_data and instance_data["isOpen"] is not None:
                                obj_for_eval[prop] = instance_data["isOpen"]
                            else:
                                # Fallback to the static ground truth dictionary if not available in instance data
                                obj_for_eval[prop] = gt_affordances_for_type.get(prop, False)
                            found_any_gt = True
                        elif prop in gt_affordances_for_type:
                            obj_for_eval[prop] = gt_affordances_for_type[prop]
                            found_any_gt = True
                        else:
                            # If a specific affordance property is not in GROUND_TRUTH_AFFORDANCES for this type,
                            # use the "Default" type's value for that property, or False if even that is missing.
                            obj_for_eval[prop] = GROUND_TRUTH_AFFORDANCES.get("Default", {}).get(prop, False)
                            print(f"Warning: Ground truth for property '{prop}' not found for objectType '{object_type}'. Used default value: {obj_for_eval[prop]}.")
                            print(f"         Please define it in GROUND_TRUTH_AFFORDANCES in settings.py for meaningful evaluation.") 
                    
                    if not found_any_gt and TARGET_INTERACTABLE_OBJECT_TYPES: # Only warn if we intended to process this type
                         print(f"Warning: No ground truth affordances found for objectType '{object_type}' ('{entity_name}') in GROUND_TRUTH_AFFORDANCES.")
                         print(f"         Results for this object might not be meaningful. Please update settings.py.")

                    all_samples.append({
                        "house": house_path.name,
                        "loc": loc_path.name,
                        "view": view_id_for_eval,
                        "image_path": image_path,
                        "obj": obj_for_eval
                    })

    if not all_samples:
        print("Critical: No samples were collected after processing all input files. Check data paths, format, and TARGET_INTERACTABLE_OBJECT_TYPES in settings.")
        return

    if args.sample_mode == "full":
        samples_to_process = all_samples
    else:
        n_samples = int(args.sample_mode)
        if n_samples > len(all_samples):
            print(f"Warning: Requested {n_samples} samples, but only {len(all_samples)} were collected. Processing all available.")
            samples_to_process = all_samples
        elif len(all_samples) == 0:
            print("Error: No samples collected, cannot proceed with sampling.")
            return
        else:
            samples_to_process = random.sample(all_samples, n_samples)
    
    print(f"Collected {len(all_samples)} total potential samples. Processing {len(samples_to_process)} samples based on sample_mode: '{args.sample_mode}'.")

    for model_name in args.models:
        print(f"\n--- Evaluating Model: {model_name} ---")
        # output_dir for model is created by AffordanceEvaluator.save_result if needed
        # but base_dir for model is good for context.
        model_output_base_dir = Path(OUTPUT_CONFIG["base_dir"]) / model_name
        
        # Pass VLM_MODEL_CONFIG and VLM_TYPE_TO_CLASS from settings
        # This assumes AffordanceEvaluator will use get_vlm_client internally or we pass it.
        # For explicit unload, we need access to the client instance used by the evaluator.
        # Let's assume AffordanceEvaluator creates and holds the client.
        # A better approach might be to create client here and pass to evaluator.
        # For now, to enable unload, we'll fetch the client again. This is not ideal if evaluator has state.

        evaluator = AffordanceEvaluator(model_name, model_output_base_dir) # Evaluator will create its own client
        
        model_all_object_results = []
        if not samples_to_process:
            print(f"No samples to process for model {model_name}. Skipping metric calculation.")
        else:
            for sample in tqdm(samples_to_process, desc=f"Evaluating {model_name}"):
                result_per_object = evaluator.evaluate_object(
                    sample["obj"],
                    sample["image_path"],
                    sample["house"],
                    sample["loc"],
                    str(sample["view"])
                )
                model_all_object_results.append(result_per_object)

                # Sanitize components for filename
                s_house = sanitize_filename(sample["house"])
                s_loc = sanitize_filename(sample["loc"])
                s_view = sanitize_filename(str(sample["view"]))
                s_obj_id = sanitize_filename(result_per_object["object_id"])

                output_path_per_object = (
                    model_output_base_dir
                    / s_house
                    / s_loc
                    / f"view_{s_view}_obj_{s_obj_id}.json"
                )
                evaluator.save_result(result_per_object, output_path_per_object)

        # Metric Calculation and Printing for the current model
        metrics_per_model_prompt = defaultdict(lambda: {"correct": 0, "total": 0})
        
        if not model_all_object_results:
            print(f"No results to calculate metrics for model {model_name}.")
            continue

        current_model_id = model_all_object_results[0]["model_id"] # Get from first result

        for single_object_result in model_all_object_results:
            for eval_key, eval_data in single_object_result.get("evaluations", {}).items():
                model_answer_str = str(eval_data["model_answer"])
                ground_truth_val = eval_data["ground_truth"]
                
                if model_answer_str == "ERROR_IN_EVALUATION":
                    continue # Skip errors for metric calculation

                is_correct = (model_answer_str.lower() == str(ground_truth_val).lower())
                
                # The unique identifier for metrics is now simply the key from the evaluations dictionary,
                # as it already contains all necessary context (group_prompt_property).
                unique_prompt_identifier = eval_key

                metrics_per_model_prompt[unique_prompt_identifier]["correct"] += 1 if is_correct else 0
                metrics_per_model_prompt[unique_prompt_identifier]["total"] += 1

        print(f"\nMetrics for model: {current_model_id}")
        model_overall_accuracies_for_avg = []

        sorted_prompt_keys = sorted(metrics_per_model_prompt.keys())

        for prompt_id in sorted_prompt_keys:
            counts = metrics_per_model_prompt[prompt_id]
            if counts["total"] > 0:
                accuracy = (counts["correct"] / counts["total"])
                print(f"  Prompt '{prompt_id}': Accuracy = {accuracy*100:.2f}% ({counts['correct']}/{counts['total']})")
                model_overall_accuracies_for_avg.append(accuracy)
            else:
                print(f"  Prompt '{prompt_id}': No valid predictions.")

        if model_overall_accuracies_for_avg:
            average_model_accuracy = (sum(model_overall_accuracies_for_avg) / len(model_overall_accuracies_for_avg)) * 100
            print(f"  Average accuracy for model {current_model_id} (across all evaluated prompts): {average_model_accuracy:.2f}%")
        else:
            print(f"  No data to calculate average accuracy for model {current_model_id}")

        # --- Attempt to unload Ollama model after evaluation ---
        # We need to get the VLM client instance. 
        # AffordanceEvaluator has its own client, let's try to get a fresh one for unload command.
        # This assumes get_vlm_client returns a client that can be used for this purpose.
        # A more robust solution would be for AffordanceEvaluator to expose its client or for the client to be managed here.
        
        # Assuming settings are available here or imported if needed for get_vlm_client
        from src.config import settings # Ensure settings is available
        
        # Call get_vlm_client with only model_name as it expects
        vlm_client_for_unload = get_vlm_client(model_name)

        if isinstance(vlm_client_for_unload, OllamaClient):
            print(f"INFO: Attempting to unload Ollama model: {model_name} by setting keep_alive to 0.")
            try:
                # Make a dummy call with keep_alive: 0 to unload the model
                # The prompt "unload" is arbitrary, just to make a call.
                # The OllamaClient's chat_completion method has been updated to handle keep_alive.
                vlm_client_for_unload.chat_completion(prompt="unload", image_path=None, keep_alive=0)
                print(f"INFO: Unload signal sent for model {model_name}.")
            except Exception as e:
                print(f"WARNING: Error trying to unload model {model_name}: {e}")
        # --- End of unload attempt ---

    print("\n--- VLM Evaluation Finished ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM Affordance Evaluation")
    current_vlm_choices = list(VLM_MODELS.keys())
    parser.add_argument("--models", type=str, nargs="+", default=[current_vlm_choices[0]] if current_vlm_choices else [], choices=current_vlm_choices)
    parser.add_argument("--sample_mode", type=str, default="10", choices=["full", "100", "10"])
    parser.add_argument("--input_dir", type=str, default="experiment_sampling_loc")
    args = parser.parse_args()
    main(args) 