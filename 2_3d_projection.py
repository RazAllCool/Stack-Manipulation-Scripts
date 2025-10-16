'''Reminder to myself: Don't use this without fixing the temp file cleanup- if it crashes all
the temporary files will remain. Original intent:
1) Select a video file
2) Load it in batches into memory-mapped numpy arrays, downscaling if necessary
    3) Generate a 3D projection (X,Y,Time) as a memory-mapped numpy array
    4) Preview the 3D projection using plotly
    5) Optionally save the 3D projection to a .npy file
    
    
    I had hopes to use either a watershed solving algorithm, a 3d CNN, or a maze solving algorithm to
    turn 3d binary tracks into tracks, but I haven't found a good way to do that yet.'''


import cv2
import numpy as np
import plotly.graph_objs as go
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import shutil

def select_directory(title):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory

def create_temp_file(temp_dir, suffix):
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)

def load_video_memmap(file_path, temp_dir, downscale_factor=1.0, progress_label=None):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale dimensions
    width = int(width * downscale_factor)
    height = int(height * downscale_factor)

    # Create a temporary file for memory-mapped video data
    temp_video_file = create_temp_file(temp_dir, '.dat')
    video_data = np.memmap(temp_video_file.name, dtype='uint8', mode='w+', shape=(frame_count, height, width))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if downscale_factor != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_data[i] = gray_frame

        if progress_label and i % max(1, (frame_count // 100)) == 0:
            progress_label.config(text=f"Loading video: {int((i/frame_count)*100)}%")
            progress_label.update()

    cap.release()
    return video_data, temp_video_file.name

def generate_3d_projection_memmap(video_data, temp_dir, progress_label=None):
    frame_count, height, width = video_data.shape

    # Create a temporary file for memory-mapped 3D projection data
    temp_projection_file = create_temp_file(temp_dir, '.dat')
    projection_data = np.memmap(temp_projection_file.name, dtype='int32', mode='w+', shape=(frame_count, height * width, 3))

    for t in range(frame_count):
        y, x = np.where(video_data[t] > 0)
        z = np.full_like(x, t)

        projection_data[t, :len(x), 0] = x
        projection_data[t, :len(x), 1] = y
        projection_data[t, :len(x), 2] = z

        if progress_label and t % max(1, (frame_count // 100)) == 0:
            progress_label.config(text=f"Generating 3D projection: {int((t/frame_count)*100)}%")
            progress_label.update()

    return projection_data, temp_projection_file.name

def preview_3d_projection_memmap(projection_file, frame_count, progress_label=None):
    projection_data = np.memmap(projection_file, dtype='int32', mode='r', shape=(frame_count, -1, 3))

    x_all, y_all, z_all = [], [], []

    chunk_size = 1000
    for t in range(0, frame_count, chunk_size):
        chunk = projection_data[t:t+chunk_size]
        valid_points = chunk[:, :, 0] != 0
        x_all.extend(chunk[valid_points, 0])
        y_all.extend(chunk[valid_points, 1])
        z_all.extend(chunk[valid_points, 2])

        if progress_label and t % max(1, (frame_count // 10)) == 0:
            progress_label.config(text=f"Preparing preview: {int((t/frame_count)*100)}%")
            progress_label.update()

    trace = go.Scatter3d(
        x=x_all, y=y_all, z=z_all,
        mode='markers',
        marker=dict(size=2, color=z_all, colorscale='Viridis', opacity=0.8)
    )

    layout = go.Layout(
        title='3D Projection of Object Movement',
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Time (Frames)')
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def save_3d_projection(projection_file, output_path="3d_projection.npy"):
    projection_data = np.memmap(projection_file, dtype='int32', mode='r')
    np.save(output_path, projection_data)
    messagebox.showinfo("Save", f"3D projection data saved to {output_path}")

def main():
    root = tk.Tk()
    root.title("3D Projection Generator")
    
    progress_label = tk.Label(root, text="Select a video file to start.", pady=10)
    progress_label.pack()

    try:
        # Select video file
        file_path = select_video_file()
        if not file_path:
            messagebox.showwarning("Warning", "No file selected. Exiting.")
            root.destroy()
            return
        
        # Select temporary directory
        temp_dir = select_directory("Select Directory for Temporary Files")
        if not temp_dir:
            messagebox.showwarning("Warning", "No directory selected for temporary files. Exiting.")
            root.destroy()
            return

        # Check available space
        _, _, free = shutil.disk_usage(temp_dir)
        required_space = os.path.getsize(file_path) * 3  # Estimate required space
        if free < required_space:
            messagebox.showerror("Error", f"Not enough space in selected directory. Need approximately {required_space // (1024*1024*1024)} GB, but only {free // (1024*1024*1024)} GB available.")
            root.destroy()
            return

        downscale_factor = 0.5
        progress_label.config(text="Loading video...")
        video_data, temp_video_file = load_video_memmap(file_path, temp_dir, downscale_factor, progress_label)

        if video_data is None:
            root.destroy()
            return

        progress_label.config(text="Generating 3D projection...")
        projection_data, temp_projection_file = generate_3d_projection_memmap(video_data, temp_dir, progress_label)

        progress_label.config(text="Preparing 3D preview...")
        preview_3d_projection_memmap(temp_projection_file, video_data.shape[0], progress_label)

        save_choice = messagebox.askyesno("Save", "Do you want to save the 3D projection data?")
        if save_choice:
            output_dir = select_directory("Select Directory to Save 3D Projection")
            if output_dir:
                save_3d_projection(temp_projection_file, os.path.join(output_dir, "3d_projection.npy"))
            else:
                messagebox.showinfo("Info", "3D projection data was not saved.")
        else:
            messagebox.showinfo("Info", "3D projection data was not saved.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")
    finally:
        if 'temp_video_file' in locals():
            os.remove(temp_video_file)
        if 'temp_projection_file' in locals():
            os.remove(temp_projection_file)
        root.destroy()

if __name__ == "__main__":
    main()