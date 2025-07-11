import os
import json
import glob
import datetime
import shutil
import argparse
from pathlib import Path
from src.utils.common import ensure_dir
from src.vlm_eval.main import main as vlm_eval_main
from src.experiments.house_collect import HouseCollectExperiment
from src.config import OUTPUT_DIR, get_env_info, VLM_EVAL_OBJECT_AFFORDANCE
import prior
from tqdm import tqdm
import random
import re
import time
from ai2thor.controller import Controller

def run_sampling_loc(args):
    ensure_dir(OUTPUT_DIR)
    print("Loading procthor-10k dataset...")
    dataset = prior.load_dataset("procthor-10k")
    print("Dataset loaded.")
    
    existing_house_indices = []
    if os.path.exists(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            if name.isdigit(): # Considering directories named by index
                try:
                    existing_house_indices.append(int(name))
                except ValueError:
                    pass # Ignore if not an integer name
            elif name.startswith("house_"):
                # Attempt to parse index if dirs are like "house_0", "house_1_procid..."
                # or "house_idx_0"
                parts = name.split('_')
                if len(parts) >= 2:
                    potential_index = parts[-1] # Check the last part after underscore
                    if potential_index.isdigit():
                        existing_house_indices.append(int(potential_index))
                    elif len(parts) >=3 and parts[1] == "idx" and parts[2].isdigit(): # for house_idx_N
                        existing_house_indices.append(int(parts[2]))

    current_house_idx: int
    num_processed_for_resume = 0

    if not existing_house_indices:
        print("No existing house folders found (or no folders named with parseable numerical indices). Starting collection from house index 0.")
        current_house_idx = 0
    else:
        max_idx = max(existing_house_indices)
        print(f"Found existing house data. Max index detected: {max_idx}.")
        if args.resume:
            current_house_idx = max_idx + 1
            num_processed_for_resume = current_house_idx
            print(f"Resuming from house index {current_house_idx}.")
        elif args.restart_at_max:
            current_house_idx = max_idx
            print(f"Restarting collection, will clear and reprocess data starting from house index {current_house_idx}.")
        else: # Default behavior: start from scratch (index 0)
            current_house_idx = 0
            if max_idx >= 0 : # only print if there was data and it wasn't a fresh start
                print(f"Starting collection from house index 0. Existing data up to index {max_idx} will be effectively ignored unless overwritten.")

    num_total_houses_in_split = 0
    target_split = 'train' 
    if hasattr(dataset, target_split) and dataset[target_split]:
        num_total_houses_in_split = len(dataset[target_split])
    
    if num_total_houses_in_split == 0:
        print(f"No houses found in the '{target_split}' dataset split. Exiting collection.")
        return

    print(f"Starting data collection for '{target_split}' split. Total houses in split: {num_total_houses_in_split}.")
    if args.num_houses > 0:
        print(f"Will process a maximum of {args.num_houses} houses based on user input.")

    houses_actually_processed_count = 0

    # --- MODIFICATION FOR TESTING ---
    # Force processing only the 11th house (index 10)
    print("INFO: MODIFICATION ACTIVE - Forcing processing of house index 10 only.")
    current_house_idx = 10
    args.num_houses = 1 # Ensure only one house is processed
    args.resume = False # Disable resume logic for this specific test
    args.restart_at_max = False # Disable restart logic for this specific test
    # --- END OF MODIFICATION ---

    while current_house_idx < num_total_houses_in_split:
        if args.num_houses > 0 and houses_actually_processed_count >= args.num_houses:
            print(f"Reached specified limit of {args.num_houses} houses to process. Stopping.")
            break
        
        print(f"\nProcessing dataset index: {current_house_idx} (Attempt {houses_actually_processed_count + 1} for this run)")
        try:
            house_data_from_dataset = dataset[target_split][current_house_idx]
            
            # Determine a representative id for directory naming, primarily for messages here.
            # HouseCollectExperiment will do its own robust ID generation for the actual folder name.
            prelim_house_id_for_log = house_data_from_dataset.get('proceduralParameters',{}).get('id') or \
                                  house_data_from_dataset.get('id') or \
                                  house_data_from_dataset.get('scene') or \
                                  f"house_idx_{current_house_idx}"
            
            # The actual directory path will be determined inside HouseCollectExperiment using its sanitized ID.
            # For clearing purposes, we can try to predict it, or rely on the user to manage if names are very dynamic.
            # Construct a potential directory name for clearing based on the same logic as in HouseCollectExperiment.
            # This is a bit of duplicated logic but helps for --restart_at_max.
            potential_house_id_for_dir_name = prelim_house_id_for_log # Start with the best guess
            # Simplified sanitation for prediction, actual one is in common.py used by the experiment
            from src.utils.common import sanitize_filename_for_path # Local import for this prediction
            predicted_house_dir = os.path.join(OUTPUT_DIR, sanitize_filename_for_path(str(potential_house_id_for_dir_name)))

            if os.path.exists(predicted_house_dir) and not args.resume: 
                print(f"Clearing existing directory: {predicted_house_dir} as resume is not active or this is a restart point.")
                shutil.rmtree(predicted_house_dir)
            elif os.path.exists(predicted_house_dir) and args.resume and current_house_idx < num_processed_for_resume:
                print(f"Warning: Directory {predicted_house_dir} exists for an index that should have been skipped on resume. Check logic.")
            
            experiment = HouseCollectExperiment(
                house_dict=house_data_from_dataset, 
                output_dir=OUTPUT_DIR, 
                current_house_dataset_idx=current_house_idx, # Pass the dataset index
                num_samples_per_house=args.samples_per_house,
                # grid_size, rotation_increment, verbose can be passed if made args
            )
            experiment.run()
            print(f"Successfully processed dataset index {current_house_idx} (Log ID: {prelim_house_id_for_log}).")
            houses_actually_processed_count += 1

        except StopIteration: 
            print("Dataset iterator exhausted.")
            break
        except IndexError: 
            print(f"IndexError: House index {current_house_idx} is out of bounds for '{target_split}' split (size {num_total_houses_in_split}).")
            break
        except Exception as e:
            log_id_for_error = str(current_house_idx)
            if 'house_data_from_dataset' in locals(): 
                 log_id_for_error = house_data_from_dataset.get('proceduralParameters',{}).get('id') or \
                                   house_data_from_dataset.get('id') or \
                                   house_data_from_dataset.get('scene') or \
                                   f"house_idx_{current_house_idx}"
            print(f"Error processing dataset index {current_house_idx} (Log ID: {log_id_for_error}): {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for better debugging
            print("Skipping to the next house.")
        
        current_house_idx += 1
        if current_house_idx < num_total_houses_in_split and (args.num_houses <=0 or houses_actually_processed_count < args.num_houses):
            # print(f"Moving to next dataset index: {current_house_idx}") # Less verbose
            pass
        elif args.num_houses > 0 and houses_actually_processed_count >= args.num_houses:
            pass 
        else:
            print(f"\nAll houses in the '{target_split}' dataset split have been processed or attempted up to index {current_house_idx-1}.")
    
    print(f"Data collection finished. Total houses processed in this run: {houses_actually_processed_count}.")


def main():
    parser = argparse.ArgumentParser(description="AI2THOR Data Collection and VLM Affordance Evaluation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # --- House Collection Command ---
    collect_parser = subparsers.add_parser("collect", help="Run house data collection from procthor-10k")
    collect_parser.add_argument("--num_houses", type=int, default=0, help="Number of houses to process. 0 for all in train split. (Default: 0)")
    collect_parser.add_argument("--samples_per_house", type=int, default=10, help="Number of random camera positions to sample per house. (Default: 10)")
    collect_parser.add_argument("--resume", action="store_true", help="Resume collection from the next house after the highest indexed existing house folder.")
    collect_parser.add_argument("--restart_at_max", action="store_true", help="Restart collection from the highest indexed existing house folder, reprocessing that house.")

    # --- VLM Evaluation Command ---
    vlm_parser = subparsers.add_parser("vlm_eval", help="Run VLM evaluation on collected data")
    vlm_parser.add_argument("--models", type=str, nargs='+', default=["gpt4v"], choices=["gpt4v", "llava", "qwen_vl"], help="Select one or more VLM models for evaluation.")
    vlm_parser.add_argument("--sample_mode", type=str, default="full", choices=["full", "100", "10"], help="Sampling mode for evaluation: full, 100 (random 100 houses), 10 (random 10 houses).")
    vlm_parser.add_argument("--input_dir", type=str, default=OUTPUT_DIR, help=f"Directory containing the collected house data (Default: {OUTPUT_DIR})")
    
    args = parser.parse_args()

    if args.resume and args.restart_at_max:
        print("Error: --resume and --restart_at_max flags cannot be used simultaneously. Choose one or neither.")
        parser.print_help()
        return
    
    if args.command == "collect":
        print("Running data collection...")
        run_sampling_loc(args)
    elif args.command == "vlm_eval":
        print("Running VLM evaluation...")
        run_custom_vlm_evaluation(args)

if __name__ == "__main__":
    main()

# Mock VLM function - To be replaced with an actual VLM API call
def get_vlm_response(image_path_or_frame, prompt_text, task_type="direction"):
    """
    Mocks a VLM call. 
    For direction, it randomly chooses a direction.
    For navigation, it randomly chooses a move.
    This should be replaced with an actual VLM API integration.
    `image_path_or_frame` can be a file path or an image frame (e.g., numpy array).
    """
    print(f"---- MOCK VLM CALL ----")
    print(f"  Prompt: {prompt_text}")
    # print(f"  Image: {image_path_or_frame if isinstance(image_path_or_frame, str) else '[frame_data]'}")
    
    if task_type == "direction":
        directions = ["left", "right", "ahead", "behind"]
        chosen_direction = random.choice(directions)
        print(f"  Mock VLM Response (Direction): {chosen_direction}")
        return chosen_direction
    elif task_type == "navigation":
        moves = ["move_ahead", "move_left", "move_right"]
        chosen_move = random.choice(moves)
        print(f"  Mock VLM Response (Navigation): {chosen_move}")
        return chosen_move
    else:
        return "Error: Unknown task type for VLM."

def evaluate_door_direction_task(task_summary: dict, first_image_path: str):
    """
    Evaluates the VLM's ability to determine the direction of a colored door.
    """
    print(f"\n--- Evaluating Door Direction Task for: {task_summary.get('door_id_sanitized_for_filename', 'UnknownDoor')} ---")
    
    door_color = task_summary.get("modified_door_color")
    if not door_color:
        print("  WARN: No modified door color found in task summary. Skipping direction task.")
        return None

    prompt = f"The door in front of you has been painted {door_color}. What is its direction relative to you? Answer with only one of 'left', 'right', 'ahead', or 'behind'."
    
    # In a real scenario, you might pass the image data directly if not a path
    vlm_response = get_vlm_response(first_image_path, prompt, task_type="direction")
    
    # TODO: Add logic to compare VLM response with ground truth if available
    # For now, we just print the response.
    print(f"  VLM Response for door direction: {vlm_response}")
    return vlm_response


def evaluate_vlm_navigation_task(
    controller: Controller, 
    task_summary: dict, 
    path_to_follow_xz: list, 
    initial_agent_state: tuple, # (position_dict, rotation_y)
    max_steps=40, 
    vlm_prompt_version="A"
    ):
    """
    Evaluates VLM-guided navigation to a target.
    Returns True if successful, False otherwise.
    """
    print(f"\n--- Evaluating VLM Navigation Task for: {task_summary.get('door_id_sanitized_for_filename', 'UnknownDoor')} ---")
    print(f"  Path to follow (XZ coords): {path_to_follow_xz}")
    print(f"  Max steps: {max_steps}, VLM Prompt Version: {vlm_prompt_version}")

    if not path_to_follow_xz:
        print("  ERROR: No path nodes provided for navigation. Aborting.")
        return False

    # Teleport agent to the start of the path
    start_pos, start_rot = initial_agent_state
    event = controller.step(
        action="Teleport",
        position=start_pos,
        rotation=start_rot,
        forceAction=True
    )
    if not event.metadata['lastActionSuccess']:
        print(f"  ERROR: Failed to teleport agent to start position {start_pos}. Aborting navigation task.")
        return False
    
    print(f"  Agent teleported to start: {start_pos}, rotation: {start_rot}")

    target_xz = path_to_follow_xz[-1]
    # Define a small tolerance for reaching the target
    target_tolerance = controller.initialization_parameters.get("gridSize", 0.25) * 0.75 

    for step_count in range(max_steps):
        current_agent_pos = event.metadata['agent']['position']
        current_agent_pos_xz = (current_agent_pos['x'], current_agent_pos['z'])

        # Check if target is reached
        dist_to_target = ((current_agent_pos_xz[0] - target_xz[0])**2 + (current_agent_pos_xz[1] - target_xz[1])**2)**0.5
        if dist_to_target < target_tolerance:
            print(f"  SUCCESS: Reached target {target_xz} in {step_count + 1} steps!")
            return True

        print(f"  Step {step_count + 1}/{max_steps}. Current pos: {current_agent_pos_xz}, Target: {target_xz}, Dist: {dist_to_target:.2f}")

        # Get current agent view (frame)
        current_frame = event.frame # This is a numpy array

        # Formulate prompt based on version
        if vlm_prompt_version == "A":
            prompt = "You need to reach the destination. Which way should you move? Choose one: move_ahead, move_left, move_right."
        elif vlm_prompt_version == "B":
            prompt = "You are navigating a house. Your goal is to reach the destination. Based on the current view, what is the best single step to take? Options: move_ahead, move_left, move_right."
        else: # Default to A
            prompt = "You need to reach the destination. Which way should you move? Choose one: move_ahead, move_left, move_right."

        vlm_action_str = None
        for attempt in range(3): # Retry VLM call up to 3 times
            response = get_vlm_response(current_frame, prompt, task_type="navigation")
            # Regex to robustly extract one of the three actions
            match = re.search(r'\b(move_ahead|move_left|move_right)\b', response, re.IGNORECASE)
            if match:
                vlm_action_str = match.group(1).lower()
                break
            else:
                print(f"    WARN: VLM attempt {attempt+1} did not yield a valid action. Raw response: '{response}'. Retrying...")
                time.sleep(0.5) # Small delay before retry

        if not vlm_action_str:
            print(f"  ERROR: VLM failed to provide a valid action after 3 attempts. Aborting navigation.")
            return False

        # Map VLM action string to AI2-THOR action
        ai2thor_action = None
        if vlm_action_str == "move_ahead":
            ai2thor_action = "MoveAhead"
        elif vlm_action_str == "move_left":
            ai2thor_action = "MoveLeft" # Note: AI2THOR uses MoveLeft/Right without underscores
        elif vlm_action_str == "move_right":
            ai2thor_action = "MoveRight"
        
        if not ai2thor_action: # Should not happen if regex is correct
            print(f"  ERROR: Could not map VLM action '{vlm_action_str}' to AI2-THOR action. Aborting.")
            return False

        print(f"    VLM suggests: {vlm_action_str} -> Executing: {ai2thor_action}")
        event = controller.step(action=ai2thor_action)
        
        if not event.metadata['lastActionSuccess']:
            print(f"    WARN: Action {ai2thor_action} failed. Error: {event.metadata.get('errorMessage')}")
            # Decide whether to stop or let the VLM try again from the new state

    print(f"  Max steps ({max_steps}) reached. Navigation failed.")
    return False


def run_custom_vlm_evaluation(args):
    # --- Configuration ---
    # Path to the output directory of house_collect.py
    collected_data_base_dir = args.input_dir # Use input_dir from the command line arguments
    
    # Example: pick a specific house and task to evaluate
    # This part should ideally be made more robust to iterate through houses/tasks
    # or use args.sample_mode if implemented for this custom eval.
    # For now, keeping it simple to find one example.
    example_house_id = "house_idx_0" # This is hardcoded for a specific test case.
    example_task_dir_name_pattern = "start_" # This is hardcoded for a specific test case.
    
    print(f"Starting Custom VLM Evaluation Script")
    print(f"Looking for collected data in: {collected_data_base_dir}")

    # Find a task summary to work with
    # This is a simplified way to find one task. A more robust iteration method is needed for general use.
    task_summary_path = None
    nav_plan_path = None
    first_image_unmasked_path = None # Path to the first unmasked image of the task

    house_path = os.path.join(collected_data_base_dir, example_house_id)
    if not os.path.isdir(house_path):
        print(f"ERROR: House directory not found: {house_path}")
        return

    for task_dir_name in os.listdir(house_path):
        if task_dir_name.startswith(example_task_dir_name_pattern):
            current_task_path = os.path.join(house_path, task_dir_name)
            if os.path.isdir(current_task_path):
                summary_file = os.path.join(current_task_path, "navigation_task_summary.json")
                plan_file = os.path.join(current_task_path, "navigation_plan.txt")
                # Look for the first unmasked image (e.g., start_unmasked_idx0_rgb.png)
                # This part should be robust to the naming convention from collect_data_at_viewpoint
                # Assumes a common pattern for the first image of the primary path
                path_data_s1_dir = os.path.join(current_task_path, "path_data_s1_to_proxy")
                if os.path.isdir(path_data_s1_dir):
                    # The exact filename depends on the output of collect_data_at_viewpoint
                    potential_image_name = "start_unmasked_idx0_rgb.png" 
                    img_path_candidate = os.path.join(path_data_s1_dir, potential_image_name)
                    if os.path.exists(img_path_candidate):
                         first_image_unmasked_path = img_path_candidate
                    else:
                        # Fallback: try to find any unmasked image at step 0 if naming is different
                        for f_name in os.listdir(path_data_s1_dir):
                            if "unmasked" in f_name and "_idx0_" in f_name and f_name.endswith("_rgb.png"):
                                first_image_unmasked_path = os.path.join(path_data_s1_dir, f_name)
                                break
                
                if os.path.exists(summary_file) and os.path.exists(plan_file) and first_image_unmasked_path:
                    task_summary_path = summary_file
                    nav_plan_path = plan_file
                    print(f"Found task: {current_task_path}")
                    break
    
    if not task_summary_path or not nav_plan_path or not first_image_unmasked_path:
        print(f"ERROR: Could not find a suitable task summary, navigation plan, and first image in {house_path} matching pattern '{example_task_dir_name_pattern}'")
        print(f"  Searched for summary: navigation_task_summary.json")
        print(f"  Searched for plan: navigation_plan.txt")
        print(f"  Searched for image")
        return

    with open(task_summary_path, 'r') as f:
        task_summary = json.load(f)

    # --- Initialize AI2-THOR Controller ---
    # It's important to use settings compatible with how data was collected
    # For simplicity, using some defaults. The scene will be set by task_summary.
    controller = Controller(
        scene=task_summary.get('house_id'), # Load the correct house/scene
        gridSize=task_summary.get('grid_size', 0.25),
        rotateStepDegrees=task_summary.get('rotation_increment', 90),
        renderInstanceSegmentation=True, # Assuming it might be needed by VLM or for debug
        width=800, # Match the data collection settings
        height=600, # Match the data collection settings
        # Add other params from HouseCollectExperiment if necessary
    )
    # Initial pass to load the scene if needed
    controller.step(action="Pass") 

    # 1. Evaluate Door Direction Task
    evaluate_door_direction_task(task_summary, first_image_unmasked_path)

    # 2. Evaluate VLM Navigation Task
    # Get path and initial state from summary or plan
    # Assuming 'recorded_teleport_commands' has the (pos_dict, rot_y) for each step of the primary path
    recorded_commands = task_summary.get('recorded_teleport_commands')
    path_nodes_xz_for_vlm_nav = task_summary.get('path_s1_nodes_xz') # Or the full path if S2 is relevant

    if recorded_commands and path_nodes_xz_for_vlm_nav:
        initial_agent_state_for_vlm_nav = recorded_commands[0] # (pos_dict, rot_y) for the first step
        
        # Run with prompt version A
        print("\nRunning VLM Navigation with Prompt Version A")
        evaluate_vlm_navigation_task(
            controller, 
            task_summary, 
            path_nodes_xz_for_vlm_nav, 
            initial_agent_state_for_vlm_nav,
            max_steps=40,
            vlm_prompt_version="A"
        )
        
        # Reset agent position for the next run if needed (or controller will be reset)
        # controller.reset(scene=task_summary.get('house_id')) # Or just re-teleport

        # Run with prompt version B
        print("\nRunning VLM Navigation with Prompt Version B")
        evaluate_vlm_navigation_task(
            controller, 
            task_summary, 
            path_nodes_xz_for_vlm_nav, 
            initial_agent_state_for_vlm_nav, # Agent will be re-teleported by the function
            max_steps=40,
            vlm_prompt_version="B"
        )
    else:
        print("WARN: Could not find recorded teleport commands or path nodes in summary for VLM navigation.")

    # --- Cleanup ---
    print("\nEvaluation finished. Stopping controller.")
    controller.stop()
