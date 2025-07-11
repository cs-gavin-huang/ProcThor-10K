import os
import math
import prior
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

import sys
# Add the project root to sys.path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.plot_utils import AgentVisualizer 

def setup_controller_for_test():
    """Initializes and returns an AI2-THOR controller, mimicking userful.py setup."""
    print("Loading procthor-10k dataset (mimicking userful.py)...")
    dataset = prior.load_dataset("procthor-10k")

    
    # Use a specific house index as in userful.py, e.g., 11, or keep 15 for consistency in tests if it works.
    # Let's stick to one that userful.py uses for closer mimicry if prior behavior is very specific.
    try:
        house_data = dataset["train"][11] # Mimicking house = dataset["train"][11] from userful.py
    except Exception as e:
        print(f"Error accessing dataset['train'][11]: {e}")
        return None

    print(f"Initializing controller with house: {house_data}")
    
    try:
        try:
            import ai2thor_colab
            ai2thor_colab.start_xserver()
            print("X server started via ai2thor_colab.")
        except ImportError:
            print("ai2thor_colab not found, assuming X server is available or not needed for this environment.")
        except Exception as e_xserver:
            print(f"Error starting X server via ai2thor_colab: {e_xserver}. Continuing without it.")

        # Align controller parameters with userful.py, keep renderInstanceSegmentation for now
        controller = Controller(
            scene=house_data,
            quality="High WebGL", # from userful.py
            width=640,          # from userful.py
            height=640,         # from userful.py
            renderInstanceSegmentation=True # Keep for potential future use
        )
        print("Controller initialized successfully.")
        return controller
    except Exception as e:
        print(f"Error initializing AI2-THOR controller: {e}")
        return None

def test_visualize_get_visible_reachable_positions(controller, fov_angle=90.0, output_dir="test_outputs/plots", perform_occlusion_check: bool = True, test_suffix: str = ""):
    if not controller:
        print("Controller not available for test.")
        return

    scene_name = "unknown_scene"
    agent_info_str = ""
    try:
        scene_name = controller.last_event.metadata['sceneName']
        agent_meta = controller.last_event.metadata['agent']
        agent_pos = agent_meta['position']
        agent_rot_y = agent_meta['rotation']['y']
        agent_info_str = f"Agent @ ({agent_pos['x']:.2f},{agent_pos['z']:.2f}), RotY: {agent_rot_y:.1f}°"
    except Exception:
        print("Could not get full scene/agent name from controller, using default.")

    occlusion_status_str = "Occlusion ON" if perform_occlusion_check else "Occlusion OFF (FOV only)"
    print(f"Running test for get_visible_reachable_positions in scene '{scene_name}' with FOV: {fov_angle}°, {occlusion_status_str}")

    # Ensure agent is in a good state by taking an action
    # This step is critical to populate controller.last_event for the visualizer
    event = controller.step(action="Pass") # Using Pass is gentler than MoveAhead if we just need metadata
    # event = controller.step(action="MoveAhead") \
    # if not event.metadata['lastActionSuccess']:\
    #     print(f"MoveAhead failed. Error: {event.metadata.get('errorMessage')}. Trying RotateRight...")\
    #     event = controller.step(action="RotateRight")\
    #     if not event.metadata['lastActionSuccess']:\
    #         print(f"RotateRight also failed. Error: {event.metadata.get('errorMessage')}. Test may not be accurate.")\
    #         return \

    # --- Call the main plotting function from AgentVisualizer ---\
    # This function internally calls get_visible_reachable_positions with the provided flag
    AgentVisualizer.plot_agent_fov_and_reachable_points(
        controller,
        fov_angle=fov_angle,
        perform_occlusion_check=perform_occlusion_check
    )

    # The plotting function now handles its own figure creation, plotting, and saving.
    # We need to ensure it saves to the correct, distinguished path.
    # For this, the plotting function itself would need to accept output_dir and a filename suffix.
    # Let's assume plot_agent_fov_and_reachable_points is modified to do this.
    # For now, the test script's output_dir parameter is illustrative for how we *want* to organize files.
    # The actual saving path is determined inside plot_agent_fov_and_reachable_points.
    # We will modify plot_agent_fov_and_reachable_points to accept output_dir and a filename_details string.

    # Create a unique filename based on parameters for this test run
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    sanitized_scene_name = "".join(c if c.isalnum() else "_" for c in scene_name)
    oc_suffix = "occl_ON" if perform_occlusion_check else "occl_OFF"
    filename_detail = f"{sanitized_scene_name}_fov{int(fov_angle)}_{oc_suffix}"
    if agent_info_str: # Add agent position to filename if available
        try: 
            agent_pos_x_str = f"x{controller.last_event.metadata['agent']['position']['x']:.1f}"
            agent_pos_z_str = f"z{controller.last_event.metadata['agent']['position']['z']:.1f}"
            agent_rot_y_str = f"rot{controller.last_event.metadata['agent']['rotation']['y']:.0f}"
            filename_detail += f"_{agent_pos_x_str}_{agent_pos_z_str}_{agent_rot_y_str}"
        except Exception: pass # Ignore if agent details are not fully available for filename
    if test_suffix: filename_detail += f"_{test_suffix}"

    # This is a conceptual call; plot_agent_fov_and_reachable_points needs to be adapted
    # to use these parameters for saving the plot. The current version saves with a fixed name. 
    # For now, this function `test_visualize_get_visible_reachable_positions` will rely on the 
    # `plot_agent_fov_and_reachable_points` to show the plot, and manual saving if needed, 
    # or we update `plot_agent_fov_and_reachable_points` next.
    print(f"Visualization for '{filename_detail}' (using internal saving of plot_utils). Output dir: {output_dir}")

if __name__ == "__main__":
    print("Setting up controller for testing...")
    test_controller = None
    try:
        test_controller = setup_controller_for_test()
        if test_controller:
            print("Controller setup complete. Running visualization tests...")
            base_output_dir = "test_outputs/plots"

            # Test 1: With occlusion checking (default)
            print("\n--- Test Group 1: Visualizing with OCCLUSION CHECK ENABLED ---")
            output_dir_occl_on = os.path.join(base_output_dir, "with_occlusion")
            test_visualize_get_visible_reachable_positions(
                test_controller, 
                fov_angle=90.0, 
                output_dir=output_dir_occl_on, 
                perform_occlusion_check=True,
                test_suffix="initial_pose"
            )

            # Test 2: Without occlusion checking (FOV only)
            print("\n--- Test Group 2: Visualizing with OCCLUSION CHECK DISABLED (FOV only) ---")
            output_dir_occl_off = os.path.join(base_output_dir, "without_occlusion")
            test_visualize_get_visible_reachable_positions(
                test_controller, 
                fov_angle=90.0, 
                output_dir=output_dir_occl_off, 
                perform_occlusion_check=False,
                test_suffix="initial_pose"
            )

            print("\nRotating agent and testing both occlusion settings again...")
            test_controller.step(action="RotateRight", degrees=75)
            test_controller.step(action="MoveAhead", moveMagnitude=0.5) # Move a bit

            test_visualize_get_visible_reachable_positions(
                test_controller, 
                fov_angle=75.0, 
                output_dir=output_dir_occl_on, 
                perform_occlusion_check=True,
                test_suffix="rotated_moved_pose"
            )
            test_visualize_get_visible_reachable_positions(
                test_controller, 
                fov_angle=75.0, 
                output_dir=output_dir_occl_off, 
                perform_occlusion_check=False,
                test_suffix="rotated_moved_pose"
            )

            print("\nTeleporting to a new pose and testing both occlusion settings again...")
            event_rps = test_controller.step(action="GetReachablePositions")
            if event_rps.metadata['lastActionSuccess'] and event_rps.metadata.get('actionReturn'):
                reachable_positions = event_rps.metadata.get('actionReturn', [])
                if reachable_positions:
                    new_pos_idx = len(reachable_positions) // 3 # Pick a different one
                    new_pos = reachable_positions[new_pos_idx]
                    new_rot = (test_controller.last_event.metadata['agent']['rotation']['y'] + 135) % 360
                    print(f"Teleporting to {new_pos} (index {new_pos_idx}) with rotation {new_rot}")
                    test_controller.step(action="Teleport", position=new_pos, rotation={"y": new_rot}, forceAction=True)
                    
                    test_visualize_get_visible_reachable_positions(
                        test_controller, 
                        fov_angle=60.0, 
                        output_dir=output_dir_occl_on, 
                        perform_occlusion_check=True,
                        test_suffix="teleported_pose"
                    )
                    test_visualize_get_visible_reachable_positions(
                        test_controller, 
                        fov_angle=60.0, 
                        output_dir=output_dir_occl_off, 
                        perform_occlusion_check=False,
                        test_suffix="teleported_pose"
                    )
                else:
                    print("No reachable positions found to choose a teleport location.")
            else:
                print("Could not get reachable positions to choose a teleport location.")
        else:
            print("Skipping test as controller setup failed.")
    finally:
        if test_controller:
            print("Stopping controller...")
            test_controller.stop()
            print("Controller stopped.")
        print("Test script finished.") 