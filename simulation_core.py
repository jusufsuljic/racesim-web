# simulation_core.py

import cv2
import numpy as np
from scipy.spatial import KDTree
import time

# --- 1. CONSTANTS & PARAMETERS ---
# We've added acceleration and braking capabilities to the kart model.
G = 9.81
KART_PARAMS = {
    "grip_level": 2.2,       # Lateral G's the tires can handle
    "max_speed": 45.0,       # Top speed in m/s (engine/drag limited)
    "max_accel": 3.5,        # Max forward acceleration in m/s^2
    "max_braking": -8.0      # Max braking deceleration in m/s^2 (negative)
}
PIXELS_PER_METER = 6.0
OPTIMIZER_PARAMS = {
    "num_points": 300,
    "iterations": 100,
    "initial_step_size": 0.5,
    "step_decay": 0.98,
    "smoothing_factor": 0.25
}

# --- 2. IMAGE & PATH PROCESSING ---
def process_track_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        return None, None, None
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2: return None, None, None
    try:
        outer_index = next(i for i, h in enumerate(hierarchy[0]) if h[3] == -1)
        inner_index = hierarchy[0][outer_index][2]
        outer_boundary = np.squeeze(contours[outer_index])
        inner_boundary = np.squeeze(contours[inner_index])
    except:
        return None, None, None
    kernel = np.ones((20, 20), np.uint8)
    safe_zone_image = cv2.erode(binary_image, kernel, iterations=1)
    return outer_boundary, inner_boundary, safe_zone_image

# In simulation_core.py

def calculate_centerline(outer, inner, num_points):
    """
    Calculates a smooth, evenly-spaced centerline for the track.
    This version includes a robust ordering algorithm to prevent path-breaking.
    """
    # 1. Calculate rough centerline points
    inner_tree = KDTree(inner)
    rough_centerline = np.array([((p + inner[inner_tree.query(p)[1]]) / 2) for p in outer])

    # 2. Order the points to form a continuous path
    # THIS IS THE CRUCIAL FIX
    # Instead of starting at an arbitrary point (index 0), we find the
    # leftmost point on the track. This gives us a reliable, consistent
    # starting position for ordering the path.
    start_index = np.argmin(rough_centerline[:, 0])
    
    # Use a nearest-neighbor approach to order the remaining points
    points_list = rough_centerline.tolist()
    start_point = points_list.pop(start_index)
    ordered_path = [start_point]

    while points_list:
        last_pt = ordered_path[-1]
        # Find the index and coordinates of the closest point in the remaining list
        closest_dist = float('inf')
        closest_idx = -1
        for i, p in enumerate(points_list):
            dist = np.linalg.norm(np.array(last_pt) - np.array(p))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        ordered_path.append(points_list.pop(closest_idx))

    ordered_path = np.array(ordered_path)

    # 3. Resample the path to have evenly spaced points
    # This ensures consistent segment lengths for the physics calculation
    path_diffs = np.diff(ordered_path, axis=0, prepend=ordered_path[-1:])
    cumulative_dist = np.cumsum(np.linalg.norm(path_diffs, axis=1))
    
    # Create new, evenly spaced distances
    even_dist = np.linspace(0, cumulative_dist[-1], num_points)
    
    # Interpolate the x and y coordinates
    interp_x = np.interp(even_dist, cumulative_dist, ordered_path[:, 0])
    interp_y = np.interp(even_dist, cumulative_dist, ordered_path[:, 1])
    
    return np.vstack((interp_x, interp_y)).T

def smooth_path(path, factor=0.25):
    smoothed = path.copy()
    for _ in range(5):
        prev_points, next_points = np.roll(smoothed, 1, axis=0), np.roll(smoothed, -1, axis=0)
        smoothed = smoothed * (1 - factor) + (prev_points + next_points) * (factor / 2)
    return smoothed


# --- 3. THE NEW PHYSICS ENGINE ---
def calculate_dynamic_lap(path, kart, return_speeds=False):
    """
    Calculates lap time based on a two-pass acceleration/braking model.
    """
    path_m = path / PIXELS_PER_METER
    num_points = len(path_m)
    dist_segments = np.linalg.norm(np.diff(path_m, axis=0, append=path_m[0:1]), axis=1)

    # --- PASS 0: Calculate maximum speed based on cornering grip ---
    p_prev, p_next = np.roll(path_m, 1, axis=0), np.roll(path_m, -1, axis=0)
    d12, d23, d31 = np.linalg.norm(p_prev - path_m, axis=1), np.linalg.norm(path_m - p_next, axis=1), np.linalg.norm(p_next - p_prev, axis=1)
    area = 0.5 * np.abs(p_prev[:,0]*(path_m[:,1]-p_next[:,1]) + path_m[:,0]*(p_next[:,1]-p_prev[:,1]) + p_next[:,0]*(p_prev[:,1]-path_m[:,1]))
    denominator = d12 * d23 * d31
    curvatures = np.divide(4 * area, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    radii = np.divide(1, curvatures, out=np.full_like(curvatures, 1e9), where=curvatures!=0)
    v_grip = np.sqrt(kart["grip_level"] * G * radii)
    v_max_corner = np.minimum(v_grip, kart["max_speed"])

    # --- PASS 1: Forwards pass (acceleration) ---
    v_forward = np.zeros(num_points)
    v_forward[0] = v_max_corner[0] # Assume flying lap start
    for i in range(1, num_points):
        v_prev = v_forward[i-1]
        dist = dist_segments[i-1]
        v_accel = np.sqrt(v_prev**2 + 2 * kart["max_accel"] * dist)
        v_forward[i] = min(v_max_corner[i], v_accel)

    # --- PASS 2: Backwards pass (braking) ---
    v_final = np.copy(v_forward)
    for i in range(num_points - 2, -1, -1):
        v_next = v_final[i+1]
        dist = dist_segments[i]
        v_brake = np.sqrt(v_next**2 - 2 * kart["max_braking"] * dist)
        v_final[i] = min(v_final[i], v_brake)
    
    if return_speeds:
        return v_final

    # --- Final Lap Time Calculation ---
    avg_speeds = (v_final + np.roll(v_final, -1)) / 2
    time_segments = dist_segments / avg_speeds
    return np.sum(time_segments)


# --- 4. OPTIMIZER ---
def optimize_path(initial_path, inner, outer, safe_zone_image, kart, params):
    print("\n--- Starting Path Optimization ---")
    start_time = time.time()
    best_path, best_time = initial_path.copy(), calculate_dynamic_lap(initial_path, kart)
    print(f"Initial Lap Time (Centerline): {best_time:.3f} s")

    inner_tree, outer_tree = KDTree(inner), KDTree(outer)
    step_size = params["initial_step_size"]
    h, w = safe_zone_image.shape

    for i in range(params["iterations"]):
        for point_idx in range(len(best_path)):
            current_path = best_path.copy()
            p_prev, p_next = current_path[point_idx - 1], current_path[(point_idx + 1) % len(current_path)]
            tangent = p_next - p_prev
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-6: continue
            normal = np.array([-tangent[1], tangent[0]]) / tangent_norm
            
            # Wiggle point and test time
            for move_factor in [0.5, -0.5]: # Outwards and inwards
                new_point = current_path[point_idx] + normal * step_size * move_factor
                px, py = int(new_point[0]), int(new_point[1])

                if 0 <= px < w and 0 <= py < h and safe_zone_image[py, px] > 0:
                    test_path = current_path.copy(); test_path[point_idx] = new_point
                    time_test = calculate_dynamic_lap(test_path, kart)
                    if time_test < best_time:
                        best_time, best_path = time_test, test_path
                        break # Found improvement, move to next point

        proposed_smoothed_path = smooth_path(best_path, params["smoothing_factor"])
        is_valid = all(0 <= int(p[0]) < w and 0 <= int(p[1]) < h and safe_zone_image[int(p[1]), int(p[0])] > 0 for p in proposed_smoothed_path)
        if is_valid:
            best_path = proposed_smoothed_path
        
        step_size *= params["step_decay"]
        if (i + 1) % 10 == 0: print(f"Iter {i+1}/{params['iterations']} | Time: {best_time:.3f} s")
    
    print(f"--- Optimization Finished in {time.time() - start_time:.2f} seconds ---")
    return best_path


def calculate_time_delta(path1, kart1, path2, kart2):
    """
    Calculates a point-by-point time delta between two laps.
    Returns an array of time differences for coloring.
    """
    # Helper to get cumulative time for a lap
    def get_cumulative_time(path, kart):
        path_m = path / PIXELS_PER_METER
        speeds = calculate_dynamic_lap(path, kart, return_speeds=True)
        dist_segments = np.linalg.norm(np.diff(path_m, axis=0, append=path_m[0:1]), axis=1)
        avg_speeds = (speeds + np.roll(speeds, -1)) / 2
        
        # Avoid division by zero if a speed is zero
        time_segments = np.divide(dist_segments, avg_speeds, out=np.full_like(avg_speeds, 0), where=avg_speeds!=0)

        return np.cumsum(time_segments)

    # Calculate cumulative time for both the new lap (path1) and the baseline (path2)
    time_lap1 = get_cumulative_time(path1, kart1)
    time_lap2 = get_cumulative_time(path2, kart2)
    
    # The delta is the difference in time taken to reach each corresponding point
    return time_lap2 - time_lap1