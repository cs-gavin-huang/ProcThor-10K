import networkx as nx
import math
import typing
import random

from src.utils.door_target_handler import get_door_target_xz # <<< IMPORT THE CORRECT HELPER

class PathPlanner:
    def __init__(self, reachable_positions: list, grid_size: float, verbose: bool = False):
        """
        Initializes the PathPlanner with reachable positions and grid parameters.

        Args:
            reachable_positions: A list of dictionaries, where each dict has 'x', 'y', 'z'.
            grid_size: The size of the grid for movement (e.g., 0.25m).
            verbose: If True, prints detailed logs.
        """
        self.reachable_positions = reachable_positions
        self.grid_size = grid_size
        self.verbose = verbose
        self.graph = nx.Graph()
        self._build_graph()

    def _round_to_grid(self, value):
        """Rounds a coordinate to the nearest grid multiple."""
        return round(round(value / self.grid_size) * self.grid_size, 4) # Added inner round for robustness

    def _build_graph(self):
        """
        Builds a NetworkX graph from the reachable positions.
        Nodes are (x, z) tuples. Edges represent possible 90-degree grid movements.
        """
        if not self.reachable_positions:
            if self.verbose: print("PathPlanner: No reachable positions, graph is empty.")
            return

        if self.verbose: 
            print(f"PathPlanner: Building graph for {len(self.reachable_positions)} reachable positions with grid_size {self.grid_size}...")
        
        nodes_xz = set()
        # Using a dictionary to quickly get the full reachable_position dict (including y) if needed later,
        # though for graph structure, only x, z are primary.
        self.rp_dict_map = {}

        for rp_dict in self.reachable_positions:
            # Round positions to be consistent with grid, helps avoid floating point issues with graph nodes.
            # Note: GetReachablePositions from AI2-THOR should already be grid-aligned if snapToGrid is on.
            # However, explicit rounding here can be a safeguard.
            # x_rounded = self._round_to_grid(rp_dict['x'])
            # z_rounded = self._round_to_grid(rp_dict['z'])
            # For now, assume GetReachablePositions is sufficiently precise from AI2-THOR
            x_coord = round(rp_dict['x'], 4) # Keep some precision
            z_coord = round(rp_dict['z'], 4)
            node_xz_tuple = (x_coord, z_coord)
            
            nodes_xz.add(node_xz_tuple)
            if node_xz_tuple not in self.rp_dict_map: # Store the first encountered y for this xz
                 self.rp_dict_map[node_xz_tuple] = rp_dict 
        
        for node_xz_tuple in nodes_xz:
            self.graph.add_node(node_xz_tuple)

        for p1_xz in nodes_xz:
            # Neighbors are grid_size away in cardinal directions
            # Check +X neighbor
            p2_x_plus = round(p1_xz[0] + self.grid_size, 4)
            p2_xz_plus_x = (p2_x_plus, p1_xz[1])
            if p2_xz_plus_x in nodes_xz:
                self.graph.add_edge(p1_xz, p2_xz_plus_x, weight=1) # Weight is 1 for unweighted grid steps

            # Check +Z neighbor
            p2_z_plus = round(p1_xz[1] + self.grid_size, 4)
            p2_xz_plus_z = (p1_xz[0], p2_z_plus)
            if p2_xz_plus_z in nodes_xz:
                self.graph.add_edge(p1_xz, p2_xz_plus_z, weight=1)
            
            # -X and -Z neighbors will be covered due to iterating all points and undirected graph

        if self.verbose: 
            print(f"PathPlanner: Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        if self.graph.number_of_nodes() == 0 and self.reachable_positions:
             if self.verbose: print("PathPlanner: WARNING - Reachable positions were provided, but graph has 0 nodes. Check rounding or grid alignment.")

    def find_closest_graph_node(self, point_xz: tuple) -> typing.Optional[tuple]:
        """
        Finds the closest node in the graph to a given (x,z) point.

        Args:
            point_xz: An (x,z) tuple for the target point.

        Returns:
            The (x,z) tuple of the closest graph node, or None if graph is empty.
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            if self.verbose: print("PathPlanner: Graph is empty, cannot find closest node.")
            return None

        # Rounded target point to match graph node format potential
        # target_actual_rounded_xz = (self._round_to_grid(point_xz[0]), self._round_to_grid(point_xz[1]))
        target_actual_rounded_xz = (round(point_xz[0],4), round(point_xz[1],4)) # Match precision used for nodes

        closest_node = None
        min_dist_sq = float('inf')

        # Optimization: If the rounded target_actual_xz is directly in graph, it's the closest (dist 0)
        if self.graph.has_node(target_actual_rounded_xz):
            if self.verbose: print(f"PathPlanner: Target point {target_actual_rounded_xz} is directly a graph node.")
            return target_actual_rounded_xz

        # Iterate through graph nodes if direct match not found (or as a general approach)
        for node_xz_tuple in self.graph.nodes():
            dist_sq = (node_xz_tuple[0] - target_actual_rounded_xz[0])**2 + \
                      (node_xz_tuple[1] - target_actual_rounded_xz[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_node = node_xz_tuple
        
        if closest_node and self.verbose:
            print(f"PathPlanner: Closest graph node to {target_actual_rounded_xz} is {closest_node} (distance: {math.sqrt(min_dist_sq):.3f}m).")
        elif not closest_node and self.verbose:
            print(f"PathPlanner: Could not find a closest graph node for {target_actual_rounded_xz} (graph nodes: {self.graph.number_of_nodes()}).")

        return closest_node

    def plan_path(self, start_xz: tuple, end_xz: tuple, graph_to_use: typing.Optional[nx.Graph] = None) -> typing.Optional[list]:
        """
        Plans a path from start_xz to end_xz using the graph.

        Args:
            start_xz: (x,z) tuple for the start point (should be a graph node).
            end_xz: (x,z) tuple for the end point (should be a graph node).
            graph_to_use: Optional. A specific NetworkX graph to use for planning. Defaults to self.graph.

        Returns:
            A list of (x,z) tuples representing the path, or None if no path found.
        """
        current_graph = graph_to_use if graph_to_use is not None else self.graph

        if not current_graph or current_graph.number_of_nodes() == 0:
            if self.verbose: print("PathPlanner: Graph is empty, cannot plan path.")
            return None
        
        # Ensure start and end nodes are actually in the graph chosen for planning
        if not current_graph.has_node(start_xz):
            if self.verbose: print(f"PathPlanner: Start node {start_xz} not in the planning graph. Attempting to find closest proxy.")
            # This behavior might be too implicit. For now, require exact node match or prior proxy finding.
            # start_xz = self.find_closest_graph_node(start_xz) # This could lead to unexpected start points
            # if not start_xz: return None
            print(f"PathPlanner: Critical - Start node {start_xz} must be an existing node in the graph for plan_path.")
            return None
            
        if not current_graph.has_node(end_xz):
            if self.verbose: print(f"PathPlanner: End node {end_xz} not in the planning graph. Attempting to find closest proxy.")
            # end_xz = self.find_closest_graph_node(end_xz)
            # if not end_xz: return None
            print(f"PathPlanner: Critical - End node {end_xz} must be an existing node in the graph for plan_path.")
            return None

        try:
            path_nodes = nx.shortest_path(current_graph, source=start_xz, target=end_xz, weight='weight')
            if self.verbose: print(f"PathPlanner: Path planned from {start_xz} to {end_xz} with {len(path_nodes)} points.")
            return path_nodes
        except nx.NetworkXNoPath:
            if self.verbose: print(f"PathPlanner: No path found between {start_xz} and {end_xz}.")
            return None
        except nx.NodeNotFound as e:
            # This should be caught by has_node checks above, but as a safeguard:
            if self.verbose: print(f"PathPlanner: Node not found during path planning: {e}. This indicates an issue with graph integrity or node validation.")
            return None
        except Exception as e_path:
            if self.verbose: print(f"PathPlanner: Unexpected error during path planning: {e_path}")
            return None

    def get_graph_nodes(self) -> list:
        """Returns a list of all nodes (x,z tuples) in the graph."""
        if self.graph:
            return list(self.graph.nodes())
        return []
    
    def get_reachable_point_details(self, node_xz: tuple) -> typing.Optional[dict]:
        """Returns the original reachable position dictionary (including y) for a given graph node."""
        return self.rp_dict_map.get(node_xz) 

def find_valid_path_to_target(
    path_planner: PathPlanner,
    valid_targets: list,
    all_reachable_nodes: list,
    min_path_len: int = 5,
    max_path_len: int = 50,
    verbose: bool = False
) -> typing.Optional[dict]:
    """
    Finds a valid, reachable path from a random start point to the closest valid target object.

    Args:
        path_planner: An initialized PathPlanner object.
        valid_targets: A list of valid targets (e.g., doors) from the house metadata.
        all_reachable_nodes: A list of all (x, z) nodes considered valid start points.
        min_path_len: The minimum desired number of steps in the path.
        max_path_len: The maximum desired number of steps in the path.
        verbose: If True, prints detailed logs.

    Returns:
        A dictionary containing the path, start/end nodes, and target info, or None if no valid path is found.
    """
    if not all_reachable_nodes:
        if verbose: print("PathFinder: No reachable nodes provided to find a path.")
        return None

    if not valid_targets:
        if verbose: print(f"PathFinder: No valid targets provided to search for.")
        return None

    # Shuffle starting points to get random paths
    random.shuffle(all_reachable_nodes)

    for start_node_xz in all_reachable_nodes:
        # Sort targets by distance to the current start_node_xz
        valid_targets.sort(key=lambda t: 
            (get_door_target_xz(t['struct'], t['scene'], None, False)[0][0] - start_node_xz[0])**2 + 
            (get_door_target_xz(t['struct'], t['scene'], None, False)[0][1] - start_node_xz[1])**2
            if get_door_target_xz(t['struct'], t['scene'], None, False)[0] is not None else float('inf')
        )

        for target_info in valid_targets:
            # The scene object is passed to get_door_target_xz for robust coordinate finding
            scene_obj = target_info.get('scene')
            target_actual_xz, _ = get_door_target_xz(target_info['struct'], scene_obj, scene_obj.get('objectType'), False)
            
            if not target_actual_xz:
                continue

            # Find the closest reachable point to the door's actual location
            target_proxy_node_xz = path_planner.find_closest_graph_node(target_actual_xz)
            if not target_proxy_node_xz:
                continue

            # Don't allow start and end points to be the same proxy node
            if start_node_xz == target_proxy_node_xz:
                continue

            path_nodes = path_planner.plan_path(start_node_xz, target_proxy_node_xz)
            
            if path_nodes and min_path_len <= len(path_nodes) <= max_path_len:
                if verbose:
                    print(f"  SUCCESS: Found path from {start_node_xz} to proxy {target_proxy_node_xz} for target '{target_info.get('id')}'")
                
                return {
                    "path_nodes_xz": path_nodes,
                    "start_node_xz": start_node_xz,
                    "target_proxy_node_xz": target_proxy_node_xz,
                    "target_actual_xz": target_actual_xz,
                    "target_info": target_info,
                }

    if verbose: print(f"PathFinder: Could not find any valid path to a target matching criteria after checking all start nodes.")
    return None 