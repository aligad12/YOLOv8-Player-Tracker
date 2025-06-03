
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle

# Initialize YOLOv8 model for detection
model = YOLO('yolov8m.pt')

# Initialize DeepSORT for tracking
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=None)

# Open the football match video
video_path =  r'C:\Users\Alaa\Downloads\Video\DFL Bundesliga 460 MP4 Videos in 30Sec. + CSV_3.mp4' # Path to your video file
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer to save the output video
output_video_path = 'tracked_players_outputmmm.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create a dictionary to store heatmaps for each player
player_heatmaps = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with YOLOv8
    results = model(frame)  # just results not connected

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
    if len(raw_detections) > 0:
        try:
            tracks = deepsort.update_tracks(raw_detections, frame=frame)  # the track for each player
        except Exception as e:
            print("Error in DeepSORT update_tracks:", e)
            break

        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue

            player_id = track.track_id  # Unique ID for the player
            x1, y1, w, h = map(int, track.to_tlwh())
            center_x = int(x1 + w / 2)
            center_y = int(y1 + h / 2)

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

# Save heatmaps and player IDs to a file
with open('player_heatmaps.pkl', 'wb') as f:
    pickle.dump(player_heatmaps, f)

print("Tracking results and video saved successfully!")

# Release resources
cap.release()
out.release()



