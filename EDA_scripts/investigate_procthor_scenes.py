import sys
import os
import random

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Controller is no longer needed for this script
    # from ai2thor.controller import Controller
    # from ai2thor.platform import CloudRendering 
    import prior
    # from procthor.configs.paths import PROCTHOR_PATH # User removed this, prior should handle path
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that prior and procthor are correctly installed and configured,")
    # print("and that PROCTHOR_PATH is accessible via procthor.configs.paths.") # User removed PROCTHOR_PATH
    sys.exit(1)

def investigate_scenes_from_dataset(num_scenes_to_sample=50):
    """
    Investigates a sample of scenes from the procthor-10k dataset by reading
    metadata directly from the loaded dataset to find object types and specifically
    checks for the presence of data under the 'doors' key.
    """
    print("--- Starting Procthor Scene Investigation (from dataset files) ---")

    print("Loading procthor-10k dataset...")
    dataset_collection = None
    try:
        dataset_collection = prior.load_dataset("procthor-10k") 
    except Exception as e:
        print(f"Error loading procthor-10k dataset: {e}")
        return

    if dataset_collection is None:
        print("prior.load_dataset returned None. Cannot proceed.")
        return

    print(f"Dataset loaded. Type: {type(dataset_collection)}.")

    all_scene_data_list = [] # Changed from dict to list
    expected_splits = ['train', 'val', 'test']

    for split_name in expected_splits:
        split_dataset_obj = None
        print(f"\nAttempting to access and process split: '{split_name}'...")
        if hasattr(dataset_collection, split_name):
            try:
                split_dataset_obj = getattr(dataset_collection, split_name)
                if split_dataset_obj is not None:
                    expected_size = -1
                    if hasattr(split_dataset_obj, '__len__'):
                        try: expected_size = len(split_dataset_obj)
                        except TypeError: print(f"  Note: len() raised TypeError for split '{split_name}', possibly lazy loading.")
                    
                    print(f"  Processing split '{split_name}' (type: {type(split_dataset_obj)}). Reported size: {expected_size if expected_size != -1 else 'unknown'}.")
                    
                    items_added_from_split = 0
                    items_iterated_in_split = 0
                    for scene_data_item in split_dataset_obj: # scene_data_item is the house data dict
                        items_iterated_in_split += 1
                        if isinstance(scene_data_item, dict):
                            all_scene_data_list.append(scene_data_item) # Add dict directly to the list
                            items_added_from_split += 1
                        else:
                            print(f"    Warning: Item in split '{split_name}' (item index in iterable: {items_iterated_in_split-1}) is not a dictionary (type: {type(scene_data_item)}). Skipping item.")
                    print(f"  Finished iterating split '{split_name}'. Iterated {items_iterated_in_split} items. Added {items_added_from_split} scene data dictionaries to the list.")
                else:
                    print(f"  Attribute '{split_name}' exists but is None.")
            except Exception as e_split_proc:
                print(f"  Error processing split '{split_name}': {e_split_proc}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Split '{split_name}' not found as an attribute in the loaded dataset collection (type: {type(dataset_collection)}).")

    if not all_scene_data_list:
        print("\nNo scene data was collected into the list. Check processing of splits.")
        return
    
    print(f"\nTotal scene data dictionaries consolidated from all splits: {len(all_scene_data_list)}")

    if num_scenes_to_sample > len(all_scene_data_list):
        print(f"Requested sample size ({num_scenes_to_sample}) is larger than total scenes ({len(all_scene_data_list)}). Sampling all {len(all_scene_data_list)} scenes.")
        sampled_scene_data_items = all_scene_data_list
    else:
        print(f"Randomly sampling {num_scenes_to_sample} scene data dictionaries from {len(all_scene_data_list)} total...")
        sampled_scene_data_items = random.sample(all_scene_data_list, num_scenes_to_sample)

    all_general_object_asset_ids = set()
    scenes_with_defined_doors = 0 
    first_door_example_printed = False # We will print details for all doors in sampled scenes with doors

    for idx, scene_data_item_from_sample in enumerate(sampled_scene_data_items):
        # Process general objects from 'objects' key
        general_objects = scene_data_item_from_sample.get('objects') 
        if isinstance(general_objects, list):
            for obj in general_objects:
                if isinstance(obj, dict):
                    # In procthor, 'assetId' is the primary identifier for these objects
                    asset_id = obj.get('assetId') 
                    if asset_id:
                        all_general_object_asset_ids.add(str(asset_id))
                    # objectType might be less common or more generic for these
                    obj_type = obj.get('objectType')
                    if obj_type:
                        all_general_object_asset_ids.add(str(obj_type)) 
        
        # Specifically check for doors under the 'doors' key
        doors_list = scene_data_item_from_sample.get('doors')
        if isinstance(doors_list, list) and len(doors_list) > 0:
            scenes_with_defined_doors += 1
            
            # Get scene ID for context
            proc_params = scene_data_item_from_sample.get('proceduralParameters')
            scene_identifier_for_log = f"(Scene ID not found in sample {idx})"
            if isinstance(proc_params, dict) and proc_params.get('id'):
                scene_identifier_for_log = str(proc_params.get('id'))
            elif scene_data_item_from_sample.get('id'): # Fallback to root ID if procedural one isn't there
                scene_identifier_for_log = str(scene_data_item_from_sample.get('id'))

            print(f"\n  --- Doors in Scene: {scene_identifier_for_log} (Sample Index {idx}) ---")
            print(f"    Found {len(doors_list)} door(s) in this scene's metadata.")

            for door_idx, door_data in enumerate(doors_list):
                if not isinstance(door_data, dict):
                    print(f"      Door entry {door_idx} is not a dict. Skipping.")
                    continue

                door_id = door_data.get('id', 'N/A')
                door_asset_id = door_data.get('assetId', 'N/A')
                door_openable = door_data.get('openable', 'N/A')
                door_room0 = door_data.get('room0', 'N/A')
                door_room1 = door_data.get('room1', 'N/A')
                door_open_state = door_data.get('openState', 'N/A') # 0.0 to 1.0
                door_hole_polygon = door_data.get('holePolygon') # List of {'x': float, 'z': float}

                print(f"    Door [{door_idx + 1}/{len(doors_list)}]:")
                print(f"      ID: {door_id}")
                print(f"      AssetID: {door_asset_id}")
                print(f"      Openable: {door_openable}")
                print(f"      OpenState: {door_open_state}")
                print(f"      Connects Room0: '{door_room0}' and Room1: '{door_room1}'")
                
                if isinstance(door_hole_polygon, list) and len(door_hole_polygon) > 0:
                    print(f"      Hole Polygon Vertices ({len(door_hole_polygon)} points):")
                    min_x, max_x = float('inf'), float('-inf')
                    min_z, max_z = float('inf'), float('-inf')
                    sum_x, sum_z = 0.0, 0.0
                    valid_points = 0
                    for pt_idx, point in enumerate(door_hole_polygon):
                        if isinstance(point, dict) and 'x' in point and 'z' in point:
                            print(f"        Point {pt_idx}: x={point['x']:.3f}, z={point['z']:.3f}")
                            px, pz = point['x'], point['z']
                            sum_x += px
                            sum_z += pz
                            min_x = min(min_x, px)
                            max_x = max(max_x, px)
                            min_z = min(min_z, pz)
                            max_z = max(max_z, pz)
                            valid_points +=1
                        else:
                            print(f"        Point {pt_idx}: Invalid format {point}")
                    
                    if valid_points > 0:
                        center_x = sum_x / valid_points
                        center_z = sum_z / valid_points
                        print(f"      Hole Approx. Center: x={center_x:.3f}, z={center_z:.3f}")
                        print(f"      Hole BBox: x_min={min_x:.3f}, x_max={max_x:.3f}, z_min={min_z:.3f}, z_max={max_z:.3f}")
                    else:
                        print("      Hole Polygon points were invalid, cannot calculate center/bbox.")
                else:
                    print(f"      Hole Polygon: Not available or empty (data: {door_hole_polygon})")
            print(f"  --- End of doors for Scene: {scene_identifier_for_log} ---")
            # We removed first_door_example_printed logic to see all doors in sampled scenes with doors.

    print("\n--- Investigation Summary ---")
    print(f"Processed metadata for {len(sampled_scene_data_items)} sampled scenes.")
    
    print("\nUnique assetIDs/objectTypes from the 'objects' list (sample may vary):")
    if all_general_object_asset_ids:
        for item_id in sorted(list(all_general_object_asset_ids)):
            print(f"  - {item_id}")
    else:
        print("  No assetIDs or objectTypes were identified in the 'objects' list of processed scenes.")

    print("\nDoor presence based on 'doors' key:")
    print(f"Found {scenes_with_defined_doors} scenes (out of {len(sampled_scene_data_items)} sampled) with a non-empty 'doors' list in their metadata.")
    
    print("\n--- End of Procthor Scene Investigation (from dataset files) ---")

if __name__ == "__main__":
    investigate_scenes_from_dataset(num_scenes_to_sample=1000) # Increased sample to 1000 for better coverage 