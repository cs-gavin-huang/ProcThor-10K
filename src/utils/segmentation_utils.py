import numpy as np

def get_floor_semantic_colors(event_metadata: dict, color_to_object_type_map: dict = None) -> set:
    """
    Identifies semantic colors for 'Floor' objects from controller event metadata.

    Args:
        event_metadata: The metadata dictionary from a controller event.
        color_to_object_type_map: Optional. The event.color_to_object_type mapping.

    Returns:
        A set of RGB tuples representing floor colors.
    """
    floor_semantic_colors_set = set()

    if color_to_object_type_map:
        # print("[DEBUG get_floor_semantic_colors] Using color_to_object_type_map.") # Keep for critical debug if needed
        for color_tuple, obj_type_str in color_to_object_type_map.items():
            if obj_type_str == "Floor":
                floor_semantic_colors_set.add(tuple(int(c) for c in color_tuple))
    
    if not floor_semantic_colors_set and 'colors' in event_metadata and event_metadata['colors']:
        # print("[DEBUG get_floor_semantic_colors] color_to_object_type_map empty or not provided, trying event_metadata['colors'].")
        color_mapping_list = event_metadata['colors']
        for item in color_mapping_list:
            color_val = item.get('color')
            if isinstance(color_val, list) and len(color_val) == 3:
                color_tuple_int = tuple(int(c) for c in color_val)
            else:
                continue
            
            full_name_str = item.get('name', '')
            # Example names: "Floor_Concrete_002", "Floor"
            base_obj_type_str = full_name_str.split('_')[0] 
            if base_obj_type_str == "Floor" or full_name_str == "Floor":
                floor_semantic_colors_set.add(color_tuple_int)
    
    # if not floor_semantic_colors_set:
        # print("[WARN get_floor_semantic_colors] NO semantic colors for 'Floor' FOUND.")
    return floor_semantic_colors_set

def get_door_semantic_colors(event_metadata: dict, color_to_object_type_map: dict = None) -> set:
    """
    Identifies semantic colors for 'Door' objects from controller event metadata.

    Args:
        event_metadata: The metadata dictionary from a controller event.
        color_to_object_type_map: Optional. The event.color_to_object_type mapping.

    Returns:
        A set of RGB tuples representing door colors.
    """
    door_semantic_colors_set = set()

    if color_to_object_type_map:
        for color_tuple, obj_type_str in color_to_object_type_map.items():
            if obj_type_str == "Door":
                door_semantic_colors_set.add(tuple(int(c) for c in color_tuple))
    
    if not door_semantic_colors_set and 'colors' in event_metadata and event_metadata['colors']:
        color_mapping_list = event_metadata['colors']
        for item in color_mapping_list:
            color_val = item.get('color')
            if isinstance(color_val, list) and len(color_val) == 3:
                color_tuple_int = tuple(int(c) for c in color_val)
            else:
                continue
            
            full_name_str = item.get('name', '')
            base_obj_type_str = full_name_str.split('_')[0] 
            if base_obj_type_str == "Door" or full_name_str == "Door":
                door_semantic_colors_set.add(color_tuple_int)
    
    return door_semantic_colors_set

def create_ground_mask_image(rgb_frame: np.ndarray, semantic_frame: np.ndarray, floor_colors_set: set) -> np.ndarray:
    """
    Creates an RGBA image where only ground pixels (based on floor_colors_set) are visible, 
    and other pixels are transparent.

    Args:
        rgb_frame: The original RGB image (H, W, 3).
        semantic_frame: The semantic segmentation image (H, W, 3).
        floor_colors_set: A set of RGB tuples representing floor colors.

    Returns:
        An RGBA image (H, W, 4) with non-ground pixels transparent.
    """
    if not floor_colors_set or semantic_frame is None:
        # Return a fully transparent RGBA image if no floor colors or no semantic frame
        return np.zeros((rgb_frame.shape[0], rgb_frame.shape[1], 4), dtype=np.uint8)

    combined_mask = np.zeros((semantic_frame.shape[0], semantic_frame.shape[1]), dtype=bool)

    for floor_color_tuple in floor_colors_set:
        floor_color_np = np.array(floor_color_tuple, dtype=np.uint8)
        current_floor_mask = np.all(semantic_frame == floor_color_np.reshape(1, 1, 3), axis=2)
        combined_mask = np.logical_or(combined_mask, current_floor_mask)

    ground_only_image_rgba = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1], 4), dtype=np.uint8)
    
    ground_only_image_rgba[combined_mask, :3] = rgb_frame[combined_mask]
    ground_only_image_rgba[combined_mask, 3] = 255 # Opaque for ground
    
    return ground_only_image_rgba 