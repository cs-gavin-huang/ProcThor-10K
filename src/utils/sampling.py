import numpy as np
import random

def uniform_sample_positions(scene_bounds: dict, num_points: int) -> list:
    """
    Samples points uniformly from the bounding box defined by scene_bounds.
    
    Args:
        scene_bounds: A dictionary with 'center' and 'size' keys for the scene.
        num_points: The number of points to sample.

    Returns:
        A list of (x, z) tuples.
    """
    if not scene_bounds or 'center' not in scene_bounds or 'size' not in scene_bounds:
        return []

    center = scene_bounds['center']
    size = scene_bounds['size']
    
    min_x = center['x'] - size['x'] / 2.0
    max_x = center['x'] + size['x'] / 2.0
    min_z = center['z'] - size['z'] / 2.0
    max_z = center['z'] + size['z'] / 2.0

    sampled_points = []
    for _ in range(num_points):
        x = random.uniform(min_x, max_x)
        z = random.uniform(min_z, max_z)
        sampled_points.append((x, z))
        
    return sampled_points

def random_perturb_position(pos, radius=0.2):
    """Add random perturbation to a position within given radius."""
    dx, dz = random.uniform(-radius, radius), random.uniform(-radius, radius)
    return {"x": pos["x"]+dx, "y": pos.get("y", 0.9), "z": pos["z"]+dz} 