import os
import copy
from PIL import Image, ImageDraw, ImageOps
import traceback
from ai2thor.controller import Controller # For type hinting
import typing
import numpy as np
import cv2 # For contour finding

def create_highlight_overlay(rgb_image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> Image.Image:
    """
    Overlays a color highlight onto an RGB image where a mask is true.

    Args:
        rgb_image: The base RGB image (H, W, 3).
        mask: The boolean mask (H, W) where True indicates the area to highlight.
        color: The RGB tuple for the highlight color (e.g., (0, 255, 0)).
        alpha: The transparency of the overlay (0.0 to 1.0).

    Returns:
        A PIL Image object of the combined image.
    """
    pil_img = Image.fromarray(rgb_image).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert mask to an image to draw on
    mask_pil = Image.fromarray(mask)
    
    # Create a colored version of the mask
    colored_mask = ImageOps.colorize(mask_pil.convert("L"), black=(0,0,0), white=color)
    
    # Put alpha into the colored mask
    colored_mask.putalpha(int(alpha * 255))

    # Composite the images
    combined = Image.alpha_composite(pil_img, colored_mask)
    return combined.convert("RGB")

def create_transparent_highlight(rgb_image: np.ndarray, instance_mask: np.ndarray, highlight_color: tuple, alpha: float = 0.3, border_width: int = 2) -> Image.Image:
    """
    Creates a 'transparent highlight' on an object in an image.

    This involves a semi-transparent fill and a solid border.

    Args:
        rgb_image: The base RGB image (H, W, 3).
        instance_mask: A boolean mask (H, W) for the specific object instance.
        highlight_color: The RGB tuple for the highlight (e.g., (0, 255, 0)).
        alpha: Transparency of the inner fill (0.0 to 1.0).
        border_width: Width of the solid border in pixels.

    Returns:
        A PIL Image object with the highlight.
    """
    pil_img = Image.fromarray(rgb_image).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Ensure mask is a binary format (0 or 255) for OpenCV
    mask_u8 = instance_mask.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fill_color = highlight_color + (int(alpha * 255),)
    
    # Draw the filled contour on the overlay
    cv2.drawContours(mask_u8, contours, -1, (255), thickness=cv2.FILLED)
    fill_mask_pil = Image.fromarray(mask_u8, mode='L')
    
    # Create the colored fill
    fill = Image.new("RGBA", pil_img.size, fill_color)
    
    # Use the mask to paste the fill onto the overlay
    overlay.paste(fill, (0,0), mask=fill_mask_pil)

    # Draw the solid border on top
    if border_width > 0 and contours:
        # Convert contours to a list of tuples for PIL
        for contour in contours:
            pil_contour = [tuple(point[0]) for point in contour]
            if len(pil_contour) > 1: # Need at least 2 points to draw a line
                 draw.line(pil_contour, fill=highlight_color, width=border_width, joint="curve")

    # Composite the overlay with the original image
    combined = Image.alpha_composite(pil_img, overlay)
    return combined.convert("RGB")

def save_path_topdown_visualization(
    controller: Controller,
    base_dir: str, 
    path_nodes_xz: list, 
    start_node_xz: tuple, 
    door_actual_xz: typing.Optional[tuple], # Can be None if not available
    door_proxy_node_xz: typing.Optional[tuple], # Can be None
    verbose: bool = False
):
    """
    Generates and saves a top-down visualization of the navigation path.
    """
    try:
        event_map_props = controller.step(action="GetMapViewCameraProperties")
        if not event_map_props.metadata['lastActionSuccess']:
            print("  WARN: Could not get map view camera properties for topdown viz.")
            return
            
        pose = copy.deepcopy(event_map_props.metadata["actionReturn"])
        scene_bounds_meta = event_map_props.metadata.get("sceneBounds", {}).get("size")
        if not scene_bounds_meta: 
                scene_bounds_meta = controller.last_event.metadata.get("sceneBounds",{}).get("size")
        
        if not scene_bounds_meta or 'x' not in scene_bounds_meta or 'z' not in scene_bounds_meta:
                print("  WARN: Could not get scene bounds for topdown viz. Using default values.")
                world_width, world_depth = 10, 10 # Default placeholder values
        else:
                world_width = scene_bounds_meta['x']
                world_depth = scene_bounds_meta['z']

        map_camera_y = pose.get('position',{}).get('y', 5.0)
        if 'y' in scene_bounds_meta: 
                map_camera_y = max(map_camera_y, scene_bounds_meta['y'] + 1.0) 
        else: 
                map_camera_y = max(map_camera_y, max(world_width, world_depth) * 0.75)

        ortho_size_factor = 0.55 
        if 'position' not in pose: pose['position'] = {} 
        pose['position']['y'] = map_camera_y 
        if 'rotation' not in pose: pose['rotation'] = {} 
        pose['rotation']['x'] = 90 
        pose['rotation']['y'] = 0  
        pose['rotation']['z'] = 0
        pose['orthographic'] = True 
        pose['orthographicSize'] = max(world_width, world_depth) * ortho_size_factor
        pose['farClippingPlane'] = map_camera_y + (max(world_width, world_depth) * 2) 
        pose['nearClippingPlane'] = map_camera_y - (max(world_width, world_depth) * 2) 
        if pose['nearClippingPlane'] <=0: pose['nearClippingPlane'] = 0.01

        pose.pop('farClipPlane', None)
        pose.pop('nearClipPlane', None)

        if 'fieldOfView' in pose: del pose['fieldOfView']

        event_third_party_cam = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white"
        )

        if not event_third_party_cam.metadata['lastActionSuccess']:
            print("  WARN: Could not add third party camera for topdown viz.")
            return
        
        if not event_third_party_cam.third_party_camera_frames or len(event_third_party_cam.third_party_camera_frames) == 0:
            print("  WARN: AddThirdPartyCamera call successful but no third_party_camera_frames were returned for topdown viz. Skipping viz.")
            return

        top_down_frame_arr = event_third_party_cam.third_party_camera_frames[0]
        top_down_frame = Image.fromarray(top_down_frame_arr).convert("RGBA")
        draw = ImageDraw.Draw(top_down_frame)

        img_width, img_height = top_down_frame.size
        cam_world_x_center = pose['position']['x']
        cam_world_z_center = pose['position']['z']
        
        world_units_img_height = pose['orthographicSize'] * 2
        world_units_img_width = world_units_img_height * (img_width / img_height) 

        def world_xz_to_pixel_xy(world_x, world_z):
            delta_x_world = world_x - cam_world_x_center
            delta_z_world = world_z - cam_world_z_center 
            norm_x = delta_x_world / world_units_img_width
            norm_z = delta_z_world / world_units_img_height 
            pixel_x = (norm_x + 0.5) * img_width
            pixel_y = (-norm_z + 0.5) * img_height 
            return int(round(pixel_x)), int(round(pixel_y))

        point_radius = max(3, int(min(img_width, img_height) * 0.008))

        if path_nodes_xz: # Ensure path_nodes_xz is not None or empty
            for i, node_xz in enumerate(path_nodes_xz):
                px, py = world_xz_to_pixel_xy(node_xz[0], node_xz[1])
                draw.ellipse([(px - point_radius, py - point_radius), (px + point_radius, py + point_radius)], fill='cyan', outline='black')
                if i > 0 and len(path_nodes_xz) > 1: # Check if there is a previous node
                    prev_px, prev_py = world_xz_to_pixel_xy(path_nodes_xz[i-1][0], path_nodes_xz[i-1][1])
                    draw.line([(prev_px, prev_py), (px, py)], fill='cyan', width=max(1, int(point_radius/2)))

        if start_node_xz: # Ensure start_node_xz is not None
            start_px, start_py = world_xz_to_pixel_xy(start_node_xz[0], start_node_xz[1])
            draw.ellipse([(start_px - point_radius, start_py - point_radius), (start_px + point_radius, start_py + point_radius)], fill='green', outline='black')

        if door_actual_xz:
            door_actual_px, door_actual_py = world_xz_to_pixel_xy(door_actual_xz[0], door_actual_xz[1])
            draw.ellipse([(door_actual_px - point_radius, door_actual_py - point_radius), (door_actual_px + point_radius, door_actual_py + point_radius)], fill='red', outline='black')

        if door_proxy_node_xz:
            door_proxy_px, door_proxy_py = world_xz_to_pixel_xy(door_proxy_node_xz[0], door_proxy_node_xz[1])
            draw.ellipse([(door_proxy_px - point_radius, door_proxy_py - point_radius), (door_proxy_px + point_radius, door_proxy_py + point_radius)], fill='magenta', outline='black')
        
        viz_path = os.path.join(base_dir, "topdown_path_viz.png")
        top_down_frame.save(viz_path)
        if verbose: print(f"    Saved top-down visualization to {viz_path}")

    except Exception as e:
        print(f"  ERROR: Failed to save top-down visualization: {e}")
        traceback.print_exc() 