# app.py

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # THE FIX IS HERE
import io

# Import our backend simulation logic
from simulation_core import *

# --- 1. PAGE CONFIGURATION & INITIALIZATION ---

# Set the page layout to wide mode for better plot visibility
st.set_page_config(layout="wide", page_title="Racesim-Web")

# Initialize Session State: This is Streamlit's way of remembering variables
# across user interactions. It's perfect for storing simulation history.
if 'history' not in st.session_state:
    st.session_state.history = []
if 'track_file_bytes' not in st.session_state:
    st.session_state.track_file_bytes = None


# --- 2. HELPER PLOTTING FUNCTIONS ---

def create_speed_plot(track_img, result):
    """Generates a Matplotlib figure for the speed profile."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(track_img)
    
    speeds = result["speeds"]
    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
    
    path_points = result["racing_line"]
    p1, p2 = path_points, np.roll(path_points, -1, axis=0)
    segments = np.array([p1, p2]).transpose((1, 0, 2))
    
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=6)
    lc.set_array(speeds)
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label('Speed (m/s)')
    ax.set_title("Speed Profile")
    return fig

def create_delta_plot(track_img, current_result, baseline_result):
    """Generates a Matplotlib figure for the time delta comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(track_img)

    time_deltas = calculate_time_delta(
        current_result["racing_line"], current_result["kart_params"],
        baseline_result["racing_line"], baseline_result["kart_params"]
    )
    
    max_delta = np.max(np.abs(time_deltas))
    if max_delta < 1e-4: max_delta = 0.1 # Avoid zero range
    norm = plt.Normalize(vmin=-max_delta, vmax=max_delta)
    
    path_points = current_result["racing_line"]
    p1, p2 = path_points, np.roll(path_points, -1, axis=0)
    segments = np.array([p1, p2]).transpose((1, 0, 2))
    
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=6)
    lc.set_array(time_deltas)
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label('Time Delta (s) [Green=Faster]')
    st.session_state.baseline_run_id = baseline_result.get('run_id', 'Previous')
    ax.set_title(f"Comparison to Run {st.session_state.baseline_run_id}")
    return fig


# --- 3. SIDEBAR - USER CONTROLS ---

st.sidebar.title("Racesim-Web")
st.sidebar.header("1. Upload Track")

# We store the uploaded file in the session state to avoid reprocessing
uploaded_file = st.sidebar.file_uploader(
    "Choose a track image (white track on black background)",
    type=["png", "jpg"]
)
if uploaded_file is not None:
    st.session_state.track_file_bytes = uploaded_file.getvalue()

if st.session_state.track_file_bytes:
    st.sidebar.header("2. Kart Parameters")
    # Sliders for kart setup
    grip = st.sidebar.slider("Grip Level (g)", 0.5, 4.0, KART_PARAMS["grip_level"], 0.1)
    max_speed = st.sidebar.slider("Max Speed (m/s)", 10.0, 80.0, KART_PARAMS["max_speed"], 1.0)
    max_accel = st.sidebar.slider("Max Acceleration (m/s²)", 1.0, 8.0, KART_PARAMS["max_accel"], 0.1)
    max_braking = st.sidebar.slider("Max Braking (m/s²)", -12.0, -2.0, KART_PARAMS["max_braking"], 0.1)
    
    st.sidebar.header("3. Run Simulation")
    # The main "Run" button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner('Optimizing path... This can take up to a minute.'):
            # Convert the stored bytes back to an image array for processing
            img_array = np.frombuffer(st.session_state.track_file_bytes, np.uint8)
            track_img_cv = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # Use a temporary file path for the core logic
            temp_track_path = "temp_track.png"
            cv2.imwrite(temp_track_path, track_img_cv)
            
            kart_params = {"grip_level": grip, "max_speed": max_speed, "max_accel": max_accel, "max_braking": max_braking}

            # --- Call the backend simulation ---
            outer, inner, safe_zone = process_track_image(temp_track_path)
            if outer is None:
                st.error("Could not process track image. Please ensure it's a valid closed loop.")
            else:
                centerline = calculate_centerline(outer, inner, OPTIMIZER_PARAMS["num_points"])
                racing_line = optimize_path(centerline, inner, outer, safe_zone, kart_params, OPTIMIZER_PARAMS)
                
                time_racing = calculate_dynamic_lap(racing_line, kart_params)
                speeds_racing = calculate_dynamic_lap(racing_line, kart_params, return_speeds=True)
                
                # --- Store results in history ---
                run_id = len(st.session_state.history) + 1
                st.session_state.history.append({
                    "run_id": run_id,
                    "time_racing": time_racing,
                    "kart_params": kart_params,
                    "racing_line": racing_line,
                    "speeds": speeds_racing
                })
        st.balloons()


# --- 4. MAIN PAGE - DISPLAY RESULTS ---

st.title("Go-Kart Lap Time Simulator")

if not st.session_state.track_file_bytes:
    st.info("Please upload a track image using the sidebar to begin.")
else:
    # Display the track image
    pil_image = Image.open(io.BytesIO(st.session_state.track_file_bytes))
    track_img_rgb = pil_image.convert('RGB')
    
    # Check if any simulations have been run
    if not st.session_state.history:
        st.subheader("Track Loaded. Adjust parameters and run a simulation.")
        st.image(track_img_rgb, caption="Current Track")
    # In app.py

# --- 4. MAIN PAGE - DISPLAY RESULTS ---
# ... (code before this is unchanged) ...
    else:
        # --- LAYOUT RESTRUCTURE START ---
        
        # Get the latest result once
        latest_result = st.session_state.history[-1]
        
        # --- ROW 1: Headers and Controls ---
        hcol1, hcol2 = st.columns(2)
        
        with hcol1:
            st.header(f"Run {latest_result['run_id']}: Speed Profile")

        with hcol2:
            st.header("Comparison")
            baseline_result = None # Initialize baseline_result
            if len(st.session_state.history) > 1:
                # Let user choose which past run to compare against
                history_options = {
                    f"Run {run['run_id']} ({run['time_racing']:.3f}s)": run
                    for run in st.session_state.history[:-1]
                }
                selected_run_label = st.selectbox(
                    "Compare against:",
                    options=history_options.keys(),
                    index=len(history_options) - 1 # Default to the most recent previous run
                )
                baseline_result = history_options[selected_run_label]
        
        # --- ROW 2: Plots ---
        pcol1, pcol2 = st.columns(2)
        
        # Get the track image once
        pil_image = Image.open(io.BytesIO(st.session_state.track_file_bytes))
        track_img_rgb = pil_image.convert('RGB')
        
        with pcol1:
            fig_speed = create_speed_plot(track_img_rgb, latest_result)
            st.pyplot(fig_speed)

        with pcol2:
            if baseline_result:
                fig_delta = create_delta_plot(track_img_rgb, latest_result, baseline_result)
                st.pyplot(fig_delta)
            else:
                # This message shows if it's the very first run
                st.info("Run another simulation with different parameters to see a comparison plot here.")
                
        # --- LAYOUT RESTRUCTURE END ---
        
        # --- Display the full simulation history table at the bottom ---
        st.divider()
        st.header("Simulation History")
        
        # Format the history for display in a clean table
        display_data = []
        for run in st.session_state.history:
            display_data.append({
                "Run ID": run['run_id'],
                "Lap Time (s)": f"{run['time_racing']:.3f}",
                "Grip (g)": run['kart_params']['grip_level'],
                "Max Speed (m/s)": run['kart_params']['max_speed'],
                "Accel (m/s²)": run['kart_params']['max_accel'],
                "Braking (m/s²)": run['kart_params']['max_braking']
            })
        
        df = pd.DataFrame(display_data).set_index("Run ID")
        st.dataframe(df, use_container_width=True)