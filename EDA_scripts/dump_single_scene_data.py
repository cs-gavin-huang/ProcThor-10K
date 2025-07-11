import sys
import os
import json

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import prior
except ImportError as e:
    print(f"Error importing prior: {e}")
    print("Please ensure that prior is correctly installed.")
    sys.exit(1)

OUTPUT_FILENAME = "dumped_scene_data.json"

def dump_first_scene_data():
    """
    Loads the procthor-10k dataset, takes the first scene from the 'train' split,
    and dumps its full data to a JSON file.
    """
    print("--- Dumping First Scene Data from Procthor-10k ---")

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

    train_split_data = None
    if hasattr(dataset_collection, 'train'):
        try:
            train_split_data = getattr(dataset_collection, 'train')
            if train_split_data is None:
                print("Error: 'train' split data is None.")
                return
            # train_split_data should be a Dataset object (iterable, with __len__)
            if not hasattr(train_split_data, '__iter__') or not hasattr(train_split_data, '__len__'):
                print(f"Error: 'train' split data (type: {type(train_split_data)}) is not iterable or does not have len.")
                return
            if len(train_split_data) == 0:
                print("Error: 'train' split is empty.")
                return
        except Exception as e:
            print(f"Error accessing 'train' split: {e}")
            return
    else:
        print("Error: Dataset collection does not have a 'train' attribute.")
        return

    print(f"Accessing first scene from 'train' split (total items in train: {len(train_split_data)}).")
    first_scene_data = None
    try:
        # Dataset objects are iterable, yielding scene data dicts directly
        for i, scene_data in enumerate(train_split_data): # Iterate to get the first item
            if i == 0: # We only need the first one
                first_scene_data = scene_data
                break 
    except Exception as e:
        print(f"Error iterating through train_split_data to get the first scene: {e}")
        return

    if first_scene_data is None:
        print("Could not retrieve the first scene data from the train split.")
        return

    if not isinstance(first_scene_data, dict):
        print(f"The first item from train split is not a dictionary (type: {type(first_scene_data)}). Cannot dump as JSON dict.")
        return

    # Try to get an ID for this scene for logging purposes
    scene_id_for_log = first_scene_data.get('id') # Check for 'id' at root
    if not scene_id_for_log:
        proc_params = first_scene_data.get('proceduralParameters')
        if isinstance(proc_params, dict) and proc_params.get('id'):
            scene_id_for_log = proc_params.get('id')
    
    scene_id_display = str(scene_id_for_log) if scene_id_for_log else "(ID not found in scene data)"
    print(f"Successfully retrieved data for the first scene. Scene ID (if found): {scene_id_display}")
    
    try:
        # Ensure the output path is relative to the script's directory
        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
        print(f"Dumping scene data to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(first_scene_data, f, indent=4)
        print(f"Successfully dumped data for scene '{scene_id_display}' to {output_path}")
    except Exception as e:
        print(f"Error dumping scene data to JSON: {e}")

if __name__ == "__main__":
    dump_first_scene_data()
