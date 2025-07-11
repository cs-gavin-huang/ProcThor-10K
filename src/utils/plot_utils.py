import matplotlib.pyplot as plt
import math
import ai2thor.controller
from ai2thor.controller import Controller
import os

class AgentVisualizer:
    @staticmethod
    def get_visible_reachable_positions(controller: Controller, 
                                        fov_degrees: float = 90.0, 
                                        perform_occlusion_check: bool = True,
                                        occlusion_check_y_offset: float = 0.5):
        """
        Gets all reachable positions within the agent's current field of view,
        optionally checking for line-of-sight occlusion.

        Args:
            controller: An instance of AI2-THOR's Controller.
            fov_degrees: The horizontal field of view angle in degrees. Defaults to 90.0.
            perform_occlusion_check: If True, performs a raycast from the agent's camera 
                                     to each point to check for obstructions. Defaults to True.
            occlusion_check_y_offset: The y-offset from the agent's camera position to the target point's y
                                      when performing the occlusion check. This is to simulate looking at a point
                                      on the floor from a typical agent camera height.

        Returns:
            A list of reachable position dictionaries ({'x': float, 'y': float, 'z': float})
            that are within the agent's FOV and, if checked, are not occluded. 
            Returns an empty list if an error occurs or no positions meet the criteria.
        """
        last_event = controller.last_event
        if not last_event or not last_event.metadata.get('agent'):
            print("Error: last_event or agent metadata not available in controller for GRP.")
            # Attempt a Pass action to populate last_event if it's missing critical data
            last_event = controller.step(action="Pass")
            if not last_event or not last_event.metadata.get('agent'):
                print("Error: last_event or agent metadata still not available after Pass. Cannot get visible positions.")
                return []

        agent_pos = last_event.metadata['agent']['position']
        agent_rot_y = last_event.metadata['agent']['rotation']['y']
        
        # Ensure cameraPosition is available, it might not be if last_event is stale
        if 'cameraPosition' not in last_event.metadata:
            print("Warning: cameraPosition not in last_event.metadata. Taking a Pass action to refresh.")
            last_event = controller.step(action="Pass")
            if 'cameraPosition' not in last_event.metadata:
                print("Error: cameraPosition still not available after Pass. Using estimated camera position.")
                # Estimate camera height if not available, though this is less accurate
                agent_camera_y = agent_pos['y'] + 1.0 # Common agent camera height offset
            else:
                agent_camera_y = agent_pos['y'] + last_event.metadata['cameraPosition']['y']
        else:
            agent_camera_y = agent_pos['y'] + last_event.metadata['cameraPosition']['y']

        agent_camera_pos = {
            'x': agent_pos['x'],
            'y': agent_camera_y,
            'z': agent_pos['z']
        }

        event_rps = controller.step(action="GetReachablePositions")
        if not event_rps.metadata['lastActionSuccess']:
            print(f"Error getting reachable positions in GRP: {event_rps.metadata.get('errorMessage')}")
            return []
        all_reachable_positions = event_rps.metadata.get('actionReturn', [])
        if not all_reachable_positions:
            return []

        visible_fov_positions = []
        agent_heading_rad = math.radians(agent_rot_y)
        half_fov_rad = math.radians(fov_degrees / 2.0)

        for rp in all_reachable_positions:
            if not isinstance(rp, dict) or 'x' not in rp or 'z' not in rp:
                continue 

            dx = rp['x'] - agent_pos['x']
            dz = rp['z'] - agent_pos['z']

            angle_to_point_rad = math.atan2(dx, dz) 
            angle_diff_rad = (angle_to_point_rad - agent_heading_rad)
            while angle_diff_rad > math.pi: angle_diff_rad -= 2 * math.pi
            while angle_diff_rad < -math.pi: angle_diff_rad += 2 * math.pi

            if abs(angle_diff_rad) <= half_fov_rad:
                if perform_occlusion_check:
                    target_occlusion_check_pos = {
                        'x': rp['x'],
                        'y': agent_camera_pos['y'] - occlusion_check_y_offset, 
                        'z': rp['z']
                    }
                    origin_pos = agent_camera_pos
                    
                    # Using Raycast instead of CheckVisibility
                    raycast_event = controller.step(
                        action="Raycast",
                        origin=origin_pos,
                        coordinate=target_occlusion_check_pos,
                        raise_for_failure=False # Handle potential failures gracefully
                    )

                    if raycast_event.metadata['lastActionSuccess']:
                        hit_data = raycast_event.metadata.get('actionReturn')
                        if hit_data and hit_data.get('hit'): # Check if ray hit an object
                            distance_to_hit = hit_data['distance']
                            distance_to_target = math.sqrt(
                                (target_occlusion_check_pos['x'] - origin_pos['x'])**2 +
                                (target_occlusion_check_pos['y'] - origin_pos['y'])**2 +
                                (target_occlusion_check_pos['z'] - origin_pos['z'])**2
                            )
                            # If hit distance is less than target distance (minus tolerance), it's occluded
                            # Tolerance (e.g., 0.1m) helps avoid issues where ray hits target surface itself
                            if distance_to_hit < distance_to_target - 0.1:
                                pass # Occluded
                            else:
                                visible_fov_positions.append(rp) # Visible (hit target or beyond)
                        elif hit_data and not hit_data.get('hit'):
                            # Raycast successful but no object hit (e.g., aimed at sky or clear path)
                            # This implies the path to the target_occlusion_check_pos is clear.
                            visible_fov_positions.append(rp)
                        # else: hit_data is None or malformed, treat as not visible for safety
                    # else: Raycast action failed, treat as not visible or log error
                    #    print(f"Raycast failed for RP {rp}: {raycast_event.metadata.get('errorMessage')}")
                else:
                    visible_fov_positions.append(rp) # Not checking occlusion, add if in FOV
        return visible_fov_positions

    @staticmethod
    def plot_agent_fov_and_reachable_points(controller: Controller, 
                                            fov_angle: float = 90.0, 
                                            perform_occlusion_check: bool = True,
                                            output_dir: str = "test_outputs/plots_from_plot_utils", 
                                            filename_detail: str = "plot"):
        """
        Visualizes the agent's FOV, all reachable positions, and visible reachable positions.
        Saves the plot to a file.

        Args:
            controller: AI2-THOR controller instance.
            fov_angle: Agent's Field of View in degrees.
            perform_occlusion_check: Passed to get_visible_reachable_positions.
            output_dir: Directory to save the plot.
            filename_detail: String to include in the filename for identification.
        """
        if not controller.last_event or not controller.last_event.metadata.get('agent') or not controller.last_event.metadata.get('cameraPosition'):
            print("Controller last_event not fully populated. Taking a Pass action to refresh.")
            controller.step(action="Pass")
            if not controller.last_event or not controller.last_event.metadata.get('agent') or not controller.last_event.metadata.get('cameraPosition'):
                print("ERROR: Controller last_event still not fully populated after Pass. Cannot plot.")
                return

        # Get all reachable positions for context
        event_all_rp = controller.step(action="GetReachablePositions")
        all_reachable_for_plot = []
        if event_all_rp.metadata['lastActionSuccess']:
            all_reachable_for_plot = event_all_rp.metadata.get('actionReturn', [])
        else:
            print(f"Failed to get all reachable positions for plotting. Error: {event_all_rp.metadata.get('errorMessage')}")
            # Continue to plot what we can

        # Get visible reachable positions using the class method, passing the flag
        visible_rps = AgentVisualizer.get_visible_reachable_positions(
            controller,
            fov_degrees=fov_angle,
            perform_occlusion_check=perform_occlusion_check
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot all reachable points
        if all_reachable_for_plot:
            xs_all = [rp["x"] for rp in all_reachable_for_plot if isinstance(rp, dict) and "x" in rp]
            zs_all = [rp["z"] for rp in all_reachable_for_plot if isinstance(rp, dict) and "z" in rp]
            ax.scatter(xs_all, zs_all, color='lightgray', label=f'All Reachable ({len(xs_all)})', s=20, zorder=1)

        # Plot visible reachable points
        if visible_rps:
            xs_visible = [rp["x"] for rp in visible_rps if isinstance(rp, dict) and "x" in rp]
            zs_visible = [rp["z"] for rp in visible_rps if isinstance(rp, dict) and "z" in rp]
            color = 'green' if perform_occlusion_check else 'deepskyblue'
            label_suffix = "(Occl. Checked)" if perform_occlusion_check else "(FOV Only)"
            ax.scatter(xs_visible, zs_visible, color=color, label=f'Visible by Method {label_suffix} ({len(xs_visible)})', s=45, zorder=3, edgecolors='darkgreen' if perform_occlusion_check else 'blue', linewidth=0.5)
        else:
            print("No reachable positions identified as 'visible' by the method in this state.")
        
        agent_x, agent_z, agent_rot_y = None, None, None
        scene_name = controller.last_event.metadata.get('sceneName', 'UnknownScene')
        agent_info_str = ""
        try:
            agent_meta = controller.last_event.metadata['agent']
            agent_pos = agent_meta['position']
            agent_x, agent_z = agent_pos['x'], agent_pos['z']
            agent_rot_y = agent_meta['rotation']['y']
            agent_info_str = f"Agent @ ({agent_x:.2f},{agent_z:.2f}), RotY: {agent_rot_y:.1f}°"
            
            ax.plot(agent_x, agent_z, 'o', color='red', markersize=12, label='Agent Position', zorder=4, markeredgecolor='black')
            
            arrow_len = 1.5 
            agent_heading_rad = math.radians(agent_rot_y)
            half_fov_rad = math.radians(fov_angle / 2.0)

            angle1_rad = agent_heading_rad - half_fov_rad
            angle2_rad = agent_heading_rad + half_fov_rad
            
            ax.plot([agent_x, agent_x + arrow_len * math.sin(angle1_rad)], 
                    [agent_z, agent_z + arrow_len * math.cos(angle1_rad)], 
                    color='orangered', linestyle='--', lw=1.5, zorder=2)
            ax.plot([agent_x, agent_x + arrow_len * math.sin(agent_heading_rad)], 
                    [agent_z, agent_z + arrow_len * math.cos(agent_heading_rad)], 
                     color='red', linestyle='-', lw=1.0, zorder=2, label='Agent Heading')
            ax.plot([agent_x, agent_x + arrow_len * math.sin(angle2_rad)], 
                    [agent_z, agent_z + arrow_len * math.cos(angle2_rad)], 
                    color='orangered', linestyle='--', lw=1.5, zorder=2)
        except KeyError:
            print("Can't get full agent position and orientation for plotting.")

        ax.set_xlabel("$x$ position (meters)")
        ax.set_ylabel("$z$ position (meters)")
        title = f"Reachable Points Visualization (FOV: {fov_angle}°)\n{scene_name} | {filename_detail}\n{agent_info_str}"
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.80, 1])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory for plots: {output_dir}")
        
        save_path = os.path.join(output_dir, f"{filename_detail}.png")
        try:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
        plt.close(fig) 