import tkinter as tk
from tkinter import ttk, messagebox
import pickle  # to read and save from file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

# Load heatmaps and track IDs from the saved file (rb => read binary)
with open('player_heatmaps.pkl', 'rb') as f:
    player_heatmaps = pickle.load(f)

# Convert player IDs to strings for consistency
player_heatmaps = {str(k): v for k, v in player_heatmaps.items()}  # k => player id, v => his heat map

# Initialize the main application window
root = tk.Tk()
root.title("Player Heatmap Viewer")
root.geometry("400x300")

# Label to show instructions
label = tk.Label(root, text="Select a Player ID to view the Heatmap:", font=("Arial", 12))
label.pack(pady=10)

# Create a combobox to list all available player IDs
player_ids = list(player_heatmaps.keys())
selected_player_id = tk.StringVar()
combobox = ttk.Combobox(root, textvariable=selected_player_id, values=player_ids, state="readonly", font=("Arial", 10))
combobox.pack(pady=10)

# Function to display the heatmap of the selected player with a background image
def show_heatmap():
    player_id = selected_player_id.get()
    if player_id:
        # Check if the player ID exists in the dictionary
        if player_id in player_heatmaps:
            heatmap = player_heatmaps[player_id]
            heatmap_normalized = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap  # Normalize heatmap

            # Load the playground background image
            background = Image.open(r'C:\Users\Alaa\Downloads\football.jpg')

            # Plotting the heatmap with the background
            plt.figure(figsize=(10, 8))
            plt.imshow(background, extent=[0, heatmap.shape[1], heatmap.shape[0], 0], aspect='auto')  # Background image
            plt.imshow(heatmap_normalized, cmap='jet', alpha=0.6, aspect='auto', extent=[0, heatmap.shape[1], heatmap.shape[0], 0])  # Overlay heatmap

            plt.colorbar(label='Density of Player Movement')
            plt.xlabel('Field Width')
            plt.ylabel('Field Height')
            plt.title(f'Heatmap of Player ID: {player_id}')
            plt.show()
        else:
            messagebox.showerror("Error", f"Player ID {player_id} not found in the data.")
    else:
        messagebox.showwarning("Warning", "Please select a Player ID.")

# Button to display the selected player's heatmap
show_button = tk.Button(root, text="Show Heatmap", command=show_heatmap, font=("Arial", 12))
show_button.pack(pady=20)

# Run the application
root.mainloop()