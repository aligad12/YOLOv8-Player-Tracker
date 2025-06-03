import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize YOLOv8 model for detection
model = YOLO('yolov8m.pt')

# Initialize DeepSORT for tracking
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=None)

# Open the football match video
video_path = r'C:\Users\Alaa\Downloads\Video\DFL Bundesliga 460 MP4 Videos in 30Sec. + CSV_3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer to save the output video
output_video_path = 'tracked_players_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create a dictionary to store heatmaps for each player
player_heatmaps = {}
player_ids = set()

# Initialize tkinter window
root = tk.Tk()
root.title("Player Tracking Viewer")
root.geometry("900x700")

# Video display panel
video_panel = tk.Label(root)
video_panel.pack()

# Combobox for selecting player ID
selected_player_id = tk.StringVar(value="Show All")
combobox = ttk.Combobox(root, textvariable=selected_player_id, state="readonly", font=("Arial", 10))
combobox.pack(pady=10)

# Function to update GUI with each frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame")
        cap.release()
        out.release()
        root.quit()
        return

    # Run detection with YOLOv8
    results = model(frame)

    # Extract bounding boxes for players
    raw_detections = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        conf = float(detection.conf[0].item())
        label = int(detection.cls[0].item())

        if label == 0:  # 'person' class
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([float(x1), float(y1), float(w), float(h)], conf, 'person'))

    # Update DeepSORT tracker
    if raw_detections:
        try:
            tracks = deepsort.update_tracks(raw_detections, frame=frame)
        except Exception as e:
            print("Error in DeepSORT update_tracks:", e)
            return

        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue

            player_id = track.track_id
            x1, y1, w, h = map(int, track.to_tlwh())
            center_x = int(x1 + w / 2)
            center_y = int(y1 + h / 2)

            # Only display the selected player ID or all players if "Show All" is selected
            if selected_player_id.get() != "Show All" and int(selected_player_id.get()) != player_id:
                continue

            # Add player ID to set if not already added
            if player_id not in player_ids:
                player_ids.add(player_id)
                combobox['values'] = sorted(["Show All"] + list(player_ids))

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {player_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Initialize heatmap for the player if not already created
            if player_id not in player_heatmaps:
                player_heatmaps[player_id] = np.zeros((frame_height, frame_width), dtype=np.float32)

            # Update player's heatmap
            cv2.circle(player_heatmaps[player_id], (center_x, center_y), radius=10, color=1, thickness=-1)

    # Write the frame to the output video
    out.write(frame)

    # Update GUI with the frame
    # (Add code here to convert the frame to a format suitable for display)

    # Schedule the next frame update
    root.after(10, update_frame)

# Start updating frames
update_frame()

# Run the tkinter main loop
root.mainloop()
