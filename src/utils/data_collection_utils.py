# Copyright (c) guoli huang.

import os
import json
import copy
import random
import numpy as np
from PIL import Image, ImageDraw
from ai2thor.controller import Controller # For type hinting
import typing

from src.utils.common import save_png

def collect_data_at_viewpoint(
    controller: Controller, 
    base_dir: str, 
    step_idx: int, 
    view_idx: int, 
    target_object_types_for_masking: set,
    floor_semantic_colors_set: set,
    specific_object_id_to_mask: typing.Optional[str] = None,
    verbose: bool = False,
    viewpoint_label: str = ""
) -> bool:
    """
    Collects RGB, semantic, instance, and metadata at a given agent viewpoint.
    Saves data to disk and returns True if successful, False otherwise.
    """
    event = controller.last_event
    if not (event and event.metadata['lastActionSuccess']):
        print(f"  Cannot collect data, last event was unsuccessful for {base_dir}, step {step_idx}")
        return False

    rgb_arr = event.frame
    semantic_arr = event.semantic_segmentation_frame
    instance_masks = event.instance_masks 
    object_metadata = event.metadata["objects"]

    agent_state_at_view = copy.deepcopy(event.metadata['agent'])
    visible_objects_meta = [o for o in object_metadata if o.get("visible")]

    visible_doors_metadata = []
    if 'doors' in event.metadata:
        for door_meta in event.metadata['doors']:
            if door_meta.get('visible'): 
                visible_doors_metadata.append(door_meta)

    actual_pos = agent_state_at_view['position']
    actual_rot = agent_state_at_view['rotation']
    file_prefix = f"step_{step_idx:02d}_view_{view_idx:01d}_x{actual_pos['x']:.2f}_y{actual_pos['y']:.2f}_z{actual_pos['z']:.2f}_rot{actual_rot['y']:.0f}"
    if viewpoint_label: 
        file_prefix = f"{viewpoint_label}_{file_prefix}"

    save_png(rgb_arr, os.path.join(base_dir, f"{file_prefix}_rgb.png"))
    
    summary_objs = []
    candidate_target_objects_for_masking_ids = []
    for obj_item in visible_objects_meta: 
        summary_objs.append({
            "name": obj_item.get("name", ""), "objectId": obj_item.get("objectId", ""),
            "objectType": obj_item.get("objectType", ""), 
            "position": obj_item.get("position", {}), "rotation": obj_item.get("rotation", {}),
            "distance": obj_item.get("distance"),
            "visible": obj_item.get("visible", False),
            "objectOrientedBoundingBox": obj_item.get("objectOrientedBoundingBox"),
            "isOpen": obj_item.get("isOpen"),
            "isPickedUp": obj_item.get("isPickedUp"),
            "isToggled": obj_item.get("isToggled"),
        })
        if obj_item.get("objectType") in target_object_types_for_masking and obj_item.get("objectId") in instance_masks:
            candidate_target_objects_for_masking_ids.append(obj_item.get("objectId"))
    
    view_metadata = {
        "agent_state": agent_state_at_view, 
        "visible_objects": summary_objs,
        "visible_doors": visible_doors_metadata 
        }
    with open(os.path.join(base_dir, f"{file_prefix}_meta.json"), "w") as f:
        json.dump(view_metadata, f, indent=2)

    base_rgb_pil = Image.fromarray(rgb_arr).convert("RGBA")
    overlay_pil = Image.new("RGBA", base_rgb_pil.size, (0,0,0,0)) 
    draw = ImageDraw.Draw(overlay_pil)

    # Draw floor mask (green) using semantic segmentation
    if semantic_arr is not None and floor_semantic_colors_set:
        for r_idx in range(semantic_arr.shape[0]): 
            for c_idx in range(semantic_arr.shape[1]): 
                pixel_color_tuple = tuple(semantic_arr[r_idx, c_idx])
                if pixel_color_tuple in floor_semantic_colors_set:
                    draw.point((c_idx, r_idx), fill=(0, 255, 0, 128))

    # Draw door mask (magenta) using instance segmentation for better accuracy
    if instance_masks:
        for obj_meta in visible_objects_meta:
            obj_id = obj_meta.get('objectId')
            obj_type = obj_meta.get('objectType')
            if obj_type == 'Door' and obj_id in instance_masks:
                mask_array = instance_masks[obj_id]
                for r_idx in range(mask_array.shape[0]):
                    for c_idx in range(mask_array.shape[1]):
                        if mask_array[r_idx, c_idx]:
                            draw.point((c_idx, r_idx), fill=(255, 0, 255, 128)) # Magenta for doors
    
    # --- MASKING LOGIC FOR A SPECIFIC OBJECT (e.g., Chair) ---
    # This logic handles the `specific_object_id_to_mask` passed from the experiment.
    if specific_object_id_to_mask:
        if specific_object_id_to_mask in instance_masks:
            # The specified object is visible in this frame, so we mask it in blue.
            mask_array = instance_masks[specific_object_id_to_mask]
            for r_idx in range(mask_array.shape[0]):
                for c_idx in range(mask_array.shape[1]):
                    if mask_array[r_idx, c_idx]:
                        # Overwrite any existing color with the specific object's color.
                        draw.point((c_idx, r_idx), fill=(0, 0, 255, 128)) # Blue for the specific target object
            if verbose: 
                print(f"    Overlay: Successfully applied mask for specific object {specific_object_id_to_mask}")
        else:
            # The specified object is NOT visible in this frame. Do nothing.
            if verbose:
                print(f"    Overlay: Specific object {specific_object_id_to_mask} is not visible in this frame. No mask applied.")

    # Fallback to general candidates if no specific ID was given (for backward compatibility)
    elif candidate_target_objects_for_masking_ids:
        object_id_to_mask_for_overlay = random.choice(candidate_target_objects_for_masking_ids)
        if verbose: 
            print(f"    Overlay: No specific_object_id given. Randomly selected general candidate: {object_id_to_mask_for_overlay}.")

        if object_id_to_mask_for_overlay in instance_masks:
            mask_array = instance_masks[object_id_to_mask_for_overlay] 
            for r_idx in range(mask_array.shape[0]):
                for c_idx in range(mask_array.shape[1]):
                    if mask_array[r_idx, c_idx]: 
                        draw.point((c_idx, r_idx), fill=(0, 0, 255, 128))
    
    final_masked_image = Image.alpha_composite(base_rgb_pil, overlay_pil)
    save_png(np.array(final_masked_image.convert("RGB")), os.path.join(base_dir, f"{file_prefix}_overlay_mask.png"))

    return True 