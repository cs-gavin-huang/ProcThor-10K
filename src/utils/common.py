# Copyright (c) guoli huang.

import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_png(img_arr, path):
    Image.fromarray(img_arr).save(path)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def print_visible_objects_summary(controller):
    print("-" * 20)
    print("Current visible movable objects (summary):")
    visible_objects_list = []
    last_event = getattr(controller, 'last_event', None)
    if last_event and last_event.metadata and 'objects' in last_event.metadata:
        objects_metadata = last_event.metadata['objects']
        if isinstance(objects_metadata, list):
            for obj in objects_metadata:
                if obj.get('visible', False) and obj.get('moveable', False):
                    obj_id = obj.get('objectId', 'N/A')
                    obj_type = obj.get('objectType', 'N/A')
                    pos = obj.get('position', {})
                    dist = obj.get('distance', -1.0)
                    pos_x = pos.get('x', '?')
                    pos_z = pos.get('z', '?')
                    pos_str = f"({pos_x:.2f}, {pos_z:.2f})" if isinstance(pos_x, (int, float)) and isinstance(pos_z, (int, float)) else f"({pos_x}, {pos_z})"
                    dist_str = f"{dist:.2f}m" if isinstance(dist, (int, float)) else f"{dist}"
                    summary_str = f"- {obj_type:<12} | {obj_id:<20} | {pos_str} | {dist_str}"
                    visible_objects_list.append(summary_str)
            if not visible_objects_list:
                print("  (No visible movable objects)")
            else:
                visible_objects_list.sort()
                for line in visible_objects_list:
                    print(line)
        else:
            print("  (Error: 'objects' metadata is not a list)")
    elif last_event:
        print("  (Last event has no 'objects' metadata)")
    else:
        print("  (Controller has no last event)")
    print("-" * 20)

def plot_locations_on_topdown(topdown_img, locations, save_path, scene_bounds, labels=None, colors=None):
    img = topdown_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    center = scene_bounds["center"]
    size = scene_bounds["size"]
    min_x = center["x"] - size["x"] / 2
    max_x = center["x"] + size["x"] / 2
    min_z = center["z"] - size["z"] / 2
    max_z = center["z"] + size["z"] / 2
    def to_img_coords(x, z):
        px = int((x - min_x) / (max_x - min_x) * (w-1))
        pz = int((z - min_z) / (max_z - min_z) * (h-1))
        return px, h-1-pz
    for i, pos in enumerate(locations):
        px, pz = to_img_coords(pos['x'], pos['z'])
        color = colors[i % len(colors)] if colors else 'red'
        r = 5
        draw.ellipse([px-r, pz-r, px+r, pz+r], outline=color, width=2)
        label = labels[i] if labels else str(i)
        draw.text((px+r+2, pz-r), label, fill=color)
    img.save(save_path)

def average_images(image_list):
    if not image_list:
        return None
    arrs = [np.array(img.convert('RGBA')).astype(np.float32) for img in image_list]
    mask = [np.any(a[:, :, :3] < 250, axis=2) for a in arrs]
    count = np.zeros(arrs[0].shape[:2], dtype=np.float32)
    sum_arr = np.zeros_like(arrs[0])
    for a, m in zip(arrs, mask):
        for c in range(4):
            sum_arr[:, :, c][m] += a[:, :, c][m]
        count[m] += 1
    count[count == 0] = 1
    avg_arr = sum_arr.copy()
    for c in range(4):
        avg_arr[:, :, c] = sum_arr[:, :, c] / count
    avg_arr[:, :, 3] = np.clip(avg_arr[:, :, 3], 0, 255)
    return Image.fromarray(avg_arr.astype(np.uint8), 'RGBA')

def sanitize_filename_for_path(name_str: str) -> str:
    """Sanitizes a string to be used as a valid filename/directory name part."""
    # Replace common problematic characters with underscores
    name_str = name_str.replace("|", "_").replace(":", "_").replace(" ", "_")
    name_str = name_str.replace("(", "_").replace(")", "_")
    name_str = name_str.replace("[", "_").replace("]", "_")
    name_str = name_str.replace("{", "_").replace("}", "_")
    name_str = name_str.replace("/", "_").replace("\\", "_") # Slashes
    name_str = name_str.replace("<", "_").replace(">", "_")
    name_str = name_str.replace("?", "_").replace("*", "_")
    
    # Keep alphanumeric, underscore, hyphen, period.
    # Filter out any other characters that might still be problematic or just not ideal.
    # This is a bit restrictive but aims for safety.
    valid_chars = []
    for char in name_str:
        if char.isalnum() or char in ['_', '-', '.']:
            valid_chars.append(char)
    
    sanitized = "".join(valid_chars)
    
    # Prevent names that are just dots or empty after sanitization
    if not sanitized.strip("._"):
        return "default_sanitized_name"
        
    # Limit length (many filesystems have limits around 255 for the whole path)
    # This limit is for a single component.
    max_len = 60 
    if len(sanitized) > max_len:
        # Simple truncation, could be smarter (e.g. hash suffix)
        sanitized = sanitized[:max_len]
        
    return sanitized 