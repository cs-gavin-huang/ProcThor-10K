# Copyright (c) guoli huang.

import math
import typing

def calculate_agent_orientation_for_next_step(
    current_pos_xz: tuple, 
    next_pos_xz: tuple, 
    current_agent_rotation_y: float,
    rotation_increment: int,
    final_target_xz: typing.Optional[tuple] = None
) -> float:
    """
    Calculates the snapped agent orientation (yaw) to face the next waypoint or a final target.
    """
    target_xz_for_orientation = next_pos_xz
    if final_target_xz is not None:
            target_xz_for_orientation = final_target_xz

    delta_x = target_xz_for_orientation[0] - current_pos_xz[0]
    delta_z = target_xz_for_orientation[1] - current_pos_xz[1]

    if math.isclose(delta_x, 0) and math.isclose(delta_z, 0):
        # If current and next/target are the same, maintain current orientation
        return current_agent_rotation_y
    
    target_yaw = math.degrees(math.atan2(delta_x, delta_z))
    
    # Snap to the closest rotation_increment
    snapped_yaw = round(target_yaw / rotation_increment) * rotation_increment
    
    # Normalize to [0, 360)
    return (snapped_yaw % 360 + 360) % 360 