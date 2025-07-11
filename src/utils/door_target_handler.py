# Copyright (c) guoli huang.

import math
import numpy as np
import typing

# Offset distances in meters
DOOR_OFFSET_DEFAULT = 0.5
DOORFRAME_OFFSET = 1.0

def get_door_target_xz(door_struct_item: dict, 
                           scene_object_for_door: typing.Optional[dict] = None, 
                           object_type_str: typing.Optional[str] = None,
                           verbose: bool = False) -> typing.Tuple[typing.Optional[tuple], str]:
    """
    Calculates the world-space XZ center for a door, then applies a fixed Z-axis offset.
    Priority for base position:
    1. Center of the 'objectOrientedBoundingBox' of the live scene object.
    2. Raw 'position' from the live scene object.
    3. Raw 'assetPosition' from the structural door item.
    A fixed offset is then added to the Z-coordinate:
    - DOORFRAME_OFFSET if object_type_str indicates a Doorframe.
    - DOOR_OFFSET_DEFAULT otherwise.
    """
    door_id_str = door_struct_item.get('id', 'UnknownDoor')
    base_xz = None
    pos_source = "failure"

    # --- Determine Base Position ---
    # Priority 1: Center of ObjectOrientedBoundingBox
    if scene_object_for_door and 'objectOrientedBoundingBox' in scene_object_for_door:
        bbox = scene_object_for_door['objectOrientedBoundingBox']
        if bbox and 'cornerPoints' in bbox and isinstance(bbox['cornerPoints'], list) and len(bbox['cornerPoints']) == 8:
            try:
                corner_points = np.array([
                    [p['x'], p['y'], p['z']] for p in bbox['cornerPoints']
                ])
                center_of_bbox = np.mean(corner_points, axis=0)
                base_xz = (round(float(center_of_bbox[0]), 4), round(float(center_of_bbox[2]), 4))
                pos_source = "OOBB_center"
                if verbose: 
                    print(f"    DEBUG Door Target: Door {door_id_str}: Base position from OOBB center {base_xz}.")
            except Exception as e:
                if verbose:
                    print(f"    DEBUG Door Target: Door {door_id_str}: Error calculating OOBB center: {e}.")
        elif verbose:
            print(f"    DEBUG Door Target: Door {door_id_str}: OOBB data present but invalid or incomplete. Skipped OOBB for base.")

    # Priority 2: Fallback to live scene object position (if base_xz not found yet)
    if base_xz is None and scene_object_for_door and 'position' in scene_object_for_door:
        live_pos = scene_object_for_door['position']
        base_xz = (round(live_pos['x'], 4), round(live_pos['z'], 4))
        pos_source = "live_object_position"
        if verbose: print(f"    DEBUG Door Target: Door {door_id_str}: Base position from LIVE object position {base_xz}.")

    # Priority 3: Fallback to structural assetPosition (if base_xz not found yet)
    if base_xz is None and 'assetPosition' in door_struct_item:
        struct_pos = door_struct_item['assetPosition']
        if 'x' in struct_pos and 'z' in struct_pos:
            base_xz = (round(struct_pos['x'], 4), round(struct_pos['z'], 4))
            pos_source = "structural_assetPosition"
            if verbose: print(f"    DEBUG Door Target: Door {door_id_str}: Base position from STRUCTURAL assetPosition {base_xz}.")

    if base_xz is None:
        if verbose: print(f"    DEBUG Door Target: Door {door_id_str}: All methods to get base door XZ failed.")
        return None, "failure"

    # --- Apply Fixed Offset to Z-coordinate based on object type ---
    current_offset = DOOR_OFFSET_DEFAULT
    object_type_description = "default_door"
    if object_type_str and "Doorframe" in object_type_str:
        current_offset = DOORFRAME_OFFSET
        object_type_description = "doorframe"
    
    final_xz = (base_xz[0], round(base_xz[1] + current_offset, 4))
    pos_source += f"_offset_fixed_Z_{object_type_description}_{current_offset}m"
    if verbose: 
        print(f"    DEBUG Door Target: Door {door_id_str} (Type: {object_type_str if object_type_str else 'N/A'}, DeterminedType: {object_type_description}): Applied FIXED +{current_offset}m Z-offset. Base: {base_xz}, Final: {final_xz}")

    return final_xz, pos_source
