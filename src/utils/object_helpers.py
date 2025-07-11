import numpy as np
import typing
import math

def find_closest_visible_object_by_type(event: typing.Dict, object_type: str) -> typing.Optional[typing.Dict]:
    """
    Finds the closest visible object of a given type from the agent's perspective.

    Args:
        event: The AI2-THOR event metadata dictionary.
        object_type: The string of the objectType to search for (e.g., "Chair").

    Returns:
        The full metadata dictionary of the closest visible object of the specified type, 
        or None if no such object is visible.
    """
    if not event or not event.get('objects'):
        return None

    agent_pos_dict = event['agent']['position']
    agent_pos = np.array([agent_pos_dict['x'], agent_pos_dict['y'], agent_pos_dict['z']])
    
    visible_objects_of_type = []
    for obj in event['objects']:
        if obj.get('objectType') == object_type and obj.get('visible'):
            visible_objects_of_type.append(obj)
    
    if not visible_objects_of_type:
        return None

    closest_object = None
    min_dist = float('inf')

    for obj in visible_objects_of_type:
        obj_pos_dict = obj['position']
        obj_pos = np.array([obj_pos_dict['x'], obj_pos_dict['y'], obj_pos_dict['z']])
        dist = np.linalg.norm(agent_pos - obj_pos)
        if dist < min_dist:
            min_dist = dist
            closest_object = obj
            
    return closest_object

def find_closest_object_by_position(
    event: typing.Dict, 
    target_xz: tuple,
    object_types: typing.Set[str]
) -> typing.Optional[typing.Dict]:
    """
    Finds the closest object of a given set of types to a target XZ coordinate.
    This is useful for matching a known coordinate to a dynamic object instance.

    Args:
        event: The AI2-THOR event metadata dictionary.
        target_xz: A tuple (x, z) of the target position.
        object_types: A set of strings of object types to consider (e.g., {"Door"}).

    Returns:
        The full metadata dictionary of the closest object, or None if no such object is found.
    """
    if not event or not event.get('objects'):
        return None
        
    closest_object = None
    min_dist_sq = float('inf')

    for obj in event['objects']:
        # Also check objectId for door-like objects that might not have the right type
        is_type_match = obj.get('objectType') in object_types
        is_id_match = any(ot.lower() in obj.get('objectId', '').lower() for ot in object_types)
        
        if is_type_match or is_id_match:
            obj_pos = obj['position']
            dist_sq = (obj_pos['x'] - target_xz[0])**2 + (obj_pos['z'] - target_xz[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_object = obj
    
    return closest_object 