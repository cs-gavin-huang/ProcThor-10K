import os
import json
import copy
import numpy as np
import shutil
import random # For selecting random doors and generating points
import math # For agent orientation
import typing
from ai2thor.controller import Controller
from PIL import Image, ImageDraw 

# Imports from src.utils
from src.utils.sampling import uniform_sample_positions
from src.utils.common import ensure_dir, save_png, save_json, sanitize_filename_for_path
from src.utils.segmentation_utils import get_floor_semantic_colors
from src.utils.navigation_planning_utils import PathPlanner, find_valid_path_to_target
from src.utils.constants import TARGET_OBJECT_TYPES_FOR_MASKING
from src.utils.door_target_handler import get_door_target_xz
from src.utils.navigation_helpers import calculate_agent_orientation_for_next_step
from src.utils.data_collection_utils import save_waypoint_data
from src.utils.visualization_utils import save_path_topdown_visualization, create_transparent_highlight
from src.utils.object_helpers import find_closest_visible_object_by_type, find_closest_object_by_position

# Imports from src.experiments
from src.experiments.base import ExperimentBase

# Imports from src.config
from src.config import EXPERIMENT_NAME, OUTPUT_DIR, NUM_SAMPLES, IMG_WIDTH, IMG_HEIGHT, IMG_QUALITY, get_env_info

class HouseCollectExperiment(ExperimentBase):
    def __init__(self, house_dict: dict, output_dir: str, current_house_dataset_idx: int, num_samples_per_house=10, grid_size=0.25, rotation_increment=90, verbose=True):
        super().__init__(house_dict, output_dir) 
        self.house = house_dict 
        
        proc_params = self.house.get('proceduralParameters')
        if isinstance(proc_params, dict) and proc_params.get('id'):
            self.house_id = str(proc_params['id'])
        elif self.house.get('id'):
            self.house_id = str(self.house.get('id'))
        elif self.house.get('scene'): 
            self.house_id = str(self.house.get('scene'))
        else:
            self.house_id = f"house_idx_{current_house_dataset_idx}"
        
        self.house_dir = os.path.join(self.output_dir, sanitize_filename_for_path(self.house_id)) 
        ensure_dir(self.house_dir)
        
        self.num_samples_per_house = num_samples_per_house 
        self.grid_size = grid_size
        self.rotation_increment = rotation_increment
        self.verbose = verbose
        self.target_object_types_for_masking = TARGET_OBJECT_TYPES_FOR_MASKING

        self.current_path_masked_object_id = None
        self.mask_applied_this_path = False
        self.path_retest_count = 5
        self.current_path_door_modification_info = None

        controller_args = {
            'scene': self.house, 
            'quality': IMG_QUALITY,
            'width': IMG_WIDTH,
            'height': IMG_HEIGHT,
            'renderSemanticSegmentation': True,
            'renderInstanceSegmentation': True, 
            'renderDepthImage': True,
            'snapToGrid': True, 
            'gridSize': self.grid_size, 
            'rotateStepDegrees': self.rotation_increment, 
        }
        
        self.controller = Controller(**controller_args)
        self.path_planner = None 
        self.agent_start_y = 0.900999 

        self.floor_semantic_colors_set = set()
        try:
            event = self.controller.step(action="Pass") 
            if event and event.metadata:
                color_map = getattr(event, 'color_to_object_type', None)
                self.floor_semantic_colors_set = get_floor_semantic_colors(event.metadata, color_map)
                if 'agent' in event.metadata and 'position' in event.metadata['agent']:
                    self.agent_start_y = event.metadata['agent']['position'].get('y', self.agent_start_y)
                
                if self.floor_semantic_colors_set:
                    if self.verbose: print(f"House {self.house_id}: Identified semantic colors for 'Floor': {self.floor_semantic_colors_set}")
                else:
                    print(f"House {self.house_id}: WARNING - NO semantic colors for 'Floor' FOUND.")
            else:
                print(f"House {self.house_id}: WARNING - Initial Pass action failed or returned no metadata.")
        except Exception as e:
            print(f"House {self.house_id}: WARNING - Error during initial Pass or fetching floor colors: {e}.")

    def _teleport_agent(self, position_dict, rotation_y=None, force_action=True):
        action_params = {
            "action": "Teleport",
            "x": position_dict['x'],
            "y": position_dict.get('y', self.agent_start_y), 
            "z": position_dict['z'],
            "forceAction": force_action
        }
        if rotation_y is not None:
            action_params["rotation"] = rotation_y
        
        return self.controller.step(**action_params)

    def _collect_along_path(self, path_nodes_xz: list, task_dir: str, task_summary: dict):
        """
        Navigates along a path, collects standard data at each waypoint, and updates the task_summary.
        """
        if not path_nodes_xz:
            return task_summary, [], False

        waypoint_dir = os.path.join(task_dir, "path_waypoints")
        ensure_dir(waypoint_dir)
        
        executed_teleport_commands = [] 
        collection_success = True

        for i, node_xz in enumerate(path_nodes_xz):
            # Determine agent orientation for this step
            next_node_xz = path_nodes_xz[i + 1] if i + 1 < len(path_nodes_xz) else node_xz
            current_agent_rot_y = self.controller.last_event.metadata['agent']['rotation']['y']
            
            rotation_y = calculate_agent_orientation_for_next_step(
                current_pos_xz=node_xz, 
                next_pos_xz=next_node_xz,
                current_agent_rotation_y=current_agent_rot_y,
                rotation_increment=self.rotation_increment
            )

            # Teleport to the waypoint
            event = self._teleport_agent({'x': node_xz[0], 'z': node_xz[1]}, rotation_y=rotation_y)
            if not (event and event.metadata['lastActionSuccess']):
                print(f"  ERROR: Teleport failed at step {i}. Aborting collection for this task.")
                collection_success = False
                break
            
            # Record the actual state after teleport
            agent_state = event.metadata['agent']
            executed_teleport_commands.append((
                {'x': agent_state['position']['x'], 'y': agent_state['position']['y'], 'z': agent_state['position']['z']},
                agent_state['rotation']['y']
            ))

            # Save all standard data for this waypoint
            visible_objects = save_waypoint_data(
                controller=self.controller,
                output_dir=waypoint_dir,
                waypoint_index=i,
                floor_semantic_colors_set=self.floor_semantic_colors_set,
                verbose=self.verbose
            )
            
            # Update the central object inventory
            if visible_objects:
                for obj in visible_objects:
                    obj_id = obj.get('objectId')
                    if obj_id and obj_id not in task_summary['object_inventory']:
                        task_summary['object_inventory'][obj_id] = {
                            "objectType": obj.get("objectType"),
                            "movable": obj.get("movable"),
                            "pickupable": obj.get("pickupable"),
                            "isopen": obj.get("isOpen"),
                            "openable": obj.get("openable"),
                            "initial_position": obj.get("position"),
                            "initial_rotation": obj.get("rotation"),
                        }

        return task_summary, executed_teleport_commands, collection_success

    def _validate_structural_doors(self, structural_doors: list, scene_metadata: dict) -> list:
        """
        Cross-references structural doors with live scene objects to filter out non-operable doors/doorframes.
        Returns a list of rich objects containing both structural and scene metadata.
        """
        validated_doors = []
        objects_by_id = {o['objectId']: o for o in scene_metadata.get('objects', [])}
        objects_by_asset_id = {o['assetId']: o for o in scene_metadata.get('objects', []) if 'assetId' in o}

        for d_struct in structural_doors:
            if not d_struct.get('openable'):
                    continue
                        
            scene_obj = None
            if d_struct.get('assetId') and d_struct['assetId'] in objects_by_asset_id:
                scene_obj = objects_by_asset_id[d_struct['assetId']]
            elif d_struct.get('id') and d_struct['id'] in objects_by_id:
                scene_obj = objects_by_id[d_struct['id']]
            
            if scene_obj:
                obj_type = scene_obj.get("objectType", "")
                if "Door" in obj_type and "Doorframe" not in obj_type:
                    # Append a richer dictionary with both parts
                    validated_doors.append({'struct': d_struct, 'scene': scene_obj})
        
        if self.verbose:
            print(f"  Validated {len(validated_doors)} of {len(structural_doors)} structural doors as usable targets.")
        return validated_doors

    def _collect_object_affordance_probes(self, task_dir: str, task_summary: dict, executed_teleport_commands: list):
        """
        At the pre-target waypoint, collect object affordance data by creating highlight images.
        """
        if len(executed_teleport_commands) < 2:
            if self.verbose: print("  Skipping object affordance probes: path too short.")
            return task_summary

        # 1. Go to the second-to-last waypoint
        pre_target_pose = executed_teleport_commands[-2]
        event = self._teleport_agent(pre_target_pose[0], rotation_y=pre_target_pose[1])
        if not (event and event.metadata['lastActionSuccess']):
            if self.verbose: print("  Failed to teleport to pre-target pose for affordance probes.")
            return task_summary
        
        # 2. Prepare directories and initial data structures
        stimuli_dir = os.path.join(task_dir, "object_affordance_stimuli")
        ensure_dir(stimuli_dir)
        task_summary['affordance_probes']['object_affordance'] = []

        # 3. Find the main target door ID to avoid highlighting it
        target_door_id = None
        target_door_actual_xz = task_summary.get('path_info', {}).get('target_actual_xz')
        if target_door_actual_xz:
            closest_door_obj = find_closest_object_by_position(event.metadata, target_door_actual_xz, {"Door"})
            if closest_door_obj:
                target_door_id = closest_door_obj.get('objectId')

        # 4. Find and process key visible objects
        if self.verbose: print("  Collecting object affordance probes...")
        rgb_frame = event.frame
        instance_masks = event.instance_masks

        for obj in event.metadata['objects']:
            obj_id = obj.get('objectId')
            if (obj.get('visible') and 
                obj.get('objectType') in self.target_object_types_for_masking and 
                obj_id != target_door_id and 
                obj_id in instance_masks):

                # We have a valid, visible, non-target key object
                if self.verbose: print(f"    - Found probe target: {obj['objectType']} ({obj_id})")
                
                mask = instance_masks[obj_id]
                highlight_img = create_transparent_highlight(
                    rgb_image=rgb_frame,
                    instance_mask=mask,
                    highlight_color=(0, 0, 255), # Blue for probes
                    alpha=0.4,
                    border_width=2
                )

                img_filename = f"obj_{obj['objectType']}_{obj_id}_highlight.png"
                img_path = os.path.join(stimuli_dir, img_filename)
                highlight_img.save(img_path)

                probe_record = {
                    "object_id": obj_id,
                    "object_type": obj.get('objectType'),
                    "stimulus_image_path": os.path.relpath(img_path, self.output_dir),
                    "ground_truth_attributes": {
                        "movable": obj.get("movable"),
                        "pickupable": obj.get("pickupable"),
                        "openable": obj.get("openable"),
                        "isOpen": obj.get("isOpen"),
                        "isToggled": obj.get("isToggled"),
                    }
                }
                task_summary['affordance_probes']['object_affordance'].append(probe_record)
        
        return task_summary

    def _collect_effect_affordance_probe(self, task_dir: str, task_summary: dict, executed_teleport_commands: list):
        """
        Places an obstacle mid-path and collects data on the blocked scene.
        If the obstacle is immovable, it also calculates a detour.
        """
        path_len = len(executed_teleport_commands)
        if path_len < 5: # Need a reasonably long path to block
            if self.verbose: print("  Skipping effect affordance probe: path too short.")
            return task_summary

        # 1. Select a midpoint to block
        midpoint_idx = path_len // 2
        pre_obstacle_pose = executed_teleport_commands[midpoint_idx - 1]
        obstacle_placement_pos_dict = executed_teleport_commands[midpoint_idx][0]

        # 2. Find a suitable object to act as an obstacle
        inventory = task_summary.get('object_inventory', {})
        
        # Try to find a movable chair first
        obstacle_obj_id = None
        obstacle_type = "movable"
        movable_candidates = [oid for oid, props in inventory.items() if props.get('objectType') == "Chair" and props.get('movable')]
        if movable_candidates:
            obstacle_obj_id = random.choice(movable_candidates)
                            else:
            # Fallback to a large, immovable object
            obstacle_type = "immovable"
            immovable_candidates = [oid for oid, props in inventory.items() if props.get('objectType') in {"Sofa", "DiningTable", "Dresser"} and not props.get('movable')]
            if immovable_candidates:
                obstacle_obj_id = random.choice(immovable_candidates)

        if not obstacle_obj_id:
            if self.verbose: print("  Skipping effect affordance probe: no suitable obstacle object found in inventory.")
            return task_summary
        
        if self.verbose: print(f"  Collecting effect affordance probe: placing {obstacle_type} obstacle '{obstacle_obj_id}'.")

        # 3. Place the obstacle
        original_obstacle_pos = inventory[obstacle_obj_id]['initial_position']
        original_obstacle_rot = inventory[obstacle_obj_id]['initial_rotation']
        
        place_obstacle_pose = {"objectId": obstacle_obj_id, "position": obstacle_placement_pos_dict, "rotation": original_obstacle_rot}
        event = self.controller.step(action="SetObjectPoses", objectPoses=[place_obstacle_pose], forceAction=True)
        
        if not event.metadata['lastActionSuccess']:
            print(f"  WARN: Could not place obstacle {obstacle_obj_id}. Skipping effect probe.")
            return task_summary

        # 4. Teleport to pre-obstacle view and collect data
        self._teleport_agent(pre_obstacle_pose[0], rotation_y=pre_obstacle_pose[1])
        
        stimuli_dir = os.path.join(task_dir, "effect_affordance_stimulus")
        ensure_dir(stimuli_dir)
        
        blocked_view_objects = save_waypoint_data(
            controller=self.controller, output_dir=stimuli_dir, waypoint_index=0,
            floor_semantic_colors_set=self.floor_semantic_colors_set, verbose=self.verbose
        )

        # 5. Store probe info in summary
        probe_record = {
            "obstacle_id": obstacle_obj_id,
            "obstacle_type": obstacle_type,
            "obstacle_is_movable_gt": inventory[obstacle_obj_id].get('movable', False),
            "placed_at_waypoint_idx": midpoint_idx,
            "blocked_view_stimulus_dir": os.path.relpath(stimuli_dir, self.output_dir),
            "detour_path": None
        }

        # 6. If immovable, calculate detour
        if obstacle_type == "immovable":
            if self.verbose: print("    Obstacle is immovable, calculating detour...")
            event_rp = self.controller.step(action="GetReachablePositions")
            if event_rp and event_rp.metadata['lastActionSuccess']:
                detour_planner = PathPlanner(event_rp.metadata['actionReturn'], self.grid_size)
                start_node = detour_planner.find_closest_graph_node((pre_obstacle_pose[0]['x'], pre_obstacle_pose[0]['z']))
                end_node = detour_planner.find_closest_graph_node(task_summary['path_info']['target_proxy_node_xz'])
                if start_node and end_node:
                    detour_path_nodes = detour_planner.plan_path(start_node, end_node)
                    if detour_path_nodes:
                        probe_record['detour_path'] = detour_path_nodes
                        if self.verbose: print(f"      - Found detour path with {len(detour_path_nodes)} steps.")

        task_summary['affordance_probes']['effect_affordance'] = probe_record

        # 7. Cleanup: Move the object back to its original position
        reset_obstacle_pose = {"objectId": obstacle_obj_id, "position": original_obstacle_pos, "rotation": original_obstacle_rot}
        self.controller.step(action="SetObjectPoses", objectPoses=[reset_obstacle_pose], forceAction=True)
        if self.verbose: print(f"    Cleaned up: Moved obstacle {obstacle_obj_id} back to original position.")

        return task_summary

    def run(self):
        """
        Main experiment execution function, rewritten for the unified collection pipeline.
        """
        print(f"Starting Unified Data Collection for house: {self.house_id}")

        # 1. Initialization
        event = self.controller.step(action="GetReachablePositions")
        if not event or not event.metadata['lastActionSuccess']:
            print("  CRITICAL: Could not get reachable positions. Aborting house.")
            return

        initial_reachable_positions = event.metadata['actionReturn']
        self.path_planner = PathPlanner(initial_reachable_positions, self.grid_size, self.verbose)

        # 2. Main loop to find and process tasks
        num_successful_tasks = 0
        for i in range(self.num_samples_per_house * 5): # Try more times to find valid paths
            if num_successful_tasks >= self.num_samples_per_house:
                print(f"  Successfully collected {num_successful_tasks} samples, limit reached.")
                break
            
            print(f"\nAttempt {i+1} to find a valid navigation task...")

            # We need live metadata for door validation
            live_metadata = self.controller.last_event.metadata
            structural_doors = self.house.get('doors', [])
            validated_doors = self._validate_structural_doors(structural_doors, live_metadata)
            all_graph_nodes = self.path_planner.get_graph_nodes()

            # 3. Find a valid navigation task
            path_data = find_valid_path_to_target(
                path_planner=self.path_planner,
                valid_targets=validated_doors,
                all_reachable_nodes=all_graph_nodes,
                min_path_len=8, # Increased for more interesting paths
                verbose=self.verbose
            )

            if not path_data:
                if self.verbose: print("  No suitable path found in this attempt.")
                                    continue
                                
            num_successful_tasks += 1
            task_name = f"task_{num_successful_tasks}_start_{path_data['start_node_xz'][0]:.2f}_{path_data['start_node_xz'][1]:.2f}_to_door_{path_data['target_info']['struct']['id']}"
            task_dir = os.path.join(self.house_dir, sanitize_filename_for_path(task_name))
            ensure_dir(task_dir)
            print(f"  Found valid task, processing and saving to: {task_dir}")

            # --- UNIFIED COLLECTION PIPELINE ---

            # 4. Initialize task_summary.json (in memory)
            task_summary = {
                "house_id": self.house_id,
                "task_name": task_name,
                "target_object": path_data['target_info']['struct'],
                "path_info": {
                    "original_path": path_data['path_nodes_xz'],
                    "start_node_xz": path_data['start_node_xz'],
                    "target_proxy_node_xz": path_data['target_proxy_node_xz'],
                    "target_actual_xz": path_data['target_actual_xz'],
                },
                "object_inventory": {},
                "affordance_probes": {}
            }

            # 5. Collect standard data along path (Navigation Affordance)
            task_summary, executed_commands, collection_success = self._collect_along_path(
                path_nodes_xz=path_data['path_nodes_xz'],
                task_dir=task_dir,
                task_summary=task_summary
            )

            if not collection_success:
                print(f"  Collection failed for task {task_name}. Skipping to next attempt.")
                shutil.rmtree(task_dir) # Clean up failed task directory
                num_successful_tasks -= 1 # Decrement success counter
                continue

            task_summary['path_info']['executed_teleport_commands'] = executed_commands

            # 6. Collect Object Affordance data
            task_summary = self._collect_object_affordance_probes(
                task_dir=task_dir,
                task_summary=task_summary,
                executed_teleport_commands=executed_commands
            )
            
            # 7. Collect Effect Affordance data
            task_summary = self._collect_effect_affordance_probe(
                task_dir=task_dir,
                task_summary=task_summary,
                executed_teleport_commands=executed_commands
            )

            # 8. Save the final task_summary.json
            summary_path = os.path.join(task_dir, "task_summary.json")
            save_json(task_summary, summary_path)
            print(f"  Saved task summary to {summary_path}")

            # 9. Save Top-Down View Visualization
            save_path_topdown_visualization(
                controller=self.controller, base_dir=task_dir, 
                path_nodes_xz=path_data['path_nodes_xz'], 
                start_node_xz=path_data['start_node_xz'], 
                door_actual_xz=path_data['target_actual_xz'],
                door_proxy_node_xz=path_data['target_proxy_node_xz'],
                verbose=self.verbose
            )

            # 10. Reset controller state for the next task to ensure independence
            if num_successful_tasks < self.num_samples_per_house:
                if self.verbose: print("  Resetting house state for next task...")
                self.controller.reset(scene=self.house)
                event_rp = self.controller.step('GetReachablePositions')
                if not (event_rp and event_rp.metadata['lastActionSuccess']):
                    print("  CRITICAL: Could not get reachable positions after reset. Aborting house.")
                    break
                self.path_planner = PathPlanner(event_rp.metadata['actionReturn'], self.grid_size, self.verbose)

        print(f"\nFinished processing for house {self.house_id}. Collected {num_successful_tasks} tasks.")
        self.controller.stop()
