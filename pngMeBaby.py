import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import h5py
import os
script_definition = "CURRENTLY WORKING 6/3: This script tracks worms in a video file and saves the tracking data and metrics to CSV files."
# Define a set of templates for worm shapes
template_contours = [
    np.array([[0, 0], [10, 0], [10, 2], [8, 2], [8, 4], [6, 4], [6, 6], [4, 6], [4, 8], [2, 8], [2, 6], [0, 6], [0, 4], [2, 4], [2, 2], [0, 2]]),
    np.array([[0, 0], [12, 0], [12, 1], [10, 1], [10, 3], [8, 3], [8, 5], [6, 5], [6, 7], [4, 7], [4, 9], [2, 9], [2, 7], [0, 7], [0, 5], [2, 5], [2, 3], [0, 3], [0, 1]]),
    # Add more templates as needed
]
template_moments = [cv2.HuMoments(cv2.moments(template)) for template in template_contours]

next_worm_id = 0
tracked_worms = {}

# Initialize DataFrame to store detection data
columns = ['frame', 'id', 'centroid_x', 'centroid_y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'merged', 'split']
df = pd.DataFrame(columns=columns)

def detect_worms(frame, frame_idx):
    global next_worm_id
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    worms = []
    for contour in contours:
        # Filter contours based on their properties
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the threshold as needed
            # Calculate the centroid, bounding box, and skeleton
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(contour)
            skeleton = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(skeleton, [contour], 0, 255, 1)
            
            # Calculate the Hu Moments of the contour
            contour_moments = cv2.HuMoments(M)
            
            # Compare the contour with the templates using Hu Moments
            match_scores = [cv2.matchShapes(contour_moments, template_moment, cv2.CONTOURS_MATCH_I3, 0) for template_moment in template_moments]
            best_match_idx = np.argmin(match_scores)
            best_match_score = match_scores[best_match_idx]
            
            worm_info = {
                'id': next_worm_id,
                'contour': contour,
                'centroid': (cx, cy),
                'bbox': (x, y, w, h),
                'skeleton': skeleton,
                'match_score': best_match_score,
                'match_template': template_contours[best_match_idx],
                'trail': [(cx, cy)],
                'merged': False,
                'frame': frame_idx
            }
            next_worm_id += 1
            worms.append(worm_info)
    
    return worms

def track_worms(previous_worms, current_worms):
    merged_worms = []
    split_worms = []

    # Match current worms with previous worms based on centroid proximity
    for cworm in current_worms:
        min_distance = float('inf')
        matched_worm = None
        for pworm in previous_worms:
            distance = np.linalg.norm(np.array(cworm['centroid']) - np.array(pworm['centroid']))
            if distance < min_distance and distance < 50:  # Threshold distance
                min_distance = distance
                matched_worm = pworm

        if matched_worm:
            cworm['id'] = matched_worm['id']
            cworm['trail'] = matched_worm['trail'] + [cworm['centroid']]
            cworm['merged'] = matched_worm['merged']
        else:
            tracked_worms[cworm['id']] = cworm

    # Detect merging events
    previous_centroids = np.array([pw['centroid'] for pw in previous_worms])
    current_centroids = np.array([cw['centroid'] for cw in current_worms])
    if len(previous_centroids) > 1 and len(current_centroids) < len(previous_centroids):
        merged_worms = previous_centroids
        print("Merging event detected")
    
    # Detect splitting events
    if len(previous_centroids) < len(current_centroids):
        split_worms = current_centroids
        print("Splitting event detected")
    
    return merged_worms, split_worms

##def show_detections(frame, worms):
    for worm in worms:
        # Draw the contour, centroid, bounding box, and match template
        cv2.drawContours(frame, [worm['contour']], 0, (0, 255, 0), 2)
        cv2.circle(frame, worm['centroid'], 3, (0, 0, 255), -1)
        x, y, w, h = worm['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(frame, [worm['match_template']], 0, (0, 128, 255), 2)
        cv2.putText(frame, f"ID: {worm['id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw the trail
        if len(worm['trail']) > 1:
            for i in range(1, len(worm['trail'])):
                cv2.line(frame, worm['trail'][i - 1], worm['trail'][i], (255, 255, 0), 2)
    
    return frame

def append_data_to_df(df, worms):
    rows = []
    for worm in worms:
        row = {
            'frame': worm['frame'],
            'id': worm['id'],
            'centroid_x': worm['centroid'][0],
            'centroid_y': worm['centroid'][1],
            'bbox_x': worm['bbox'][0],
            'bbox_y': worm['bbox'][1],
            'bbox_w': worm['bbox'][2],
            'bbox_h': worm['bbox'][3],
            'merged': worm['merged'],
            'split': False  # Placeholder for split detection
        }
        rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

def calculate_metrics(df):
    df['velocity'] = df.groupby('id').apply(lambda group: group[['centroid_x', 'centroid_y']].diff().apply(np.linalg.norm, axis=1)).reset_index(level=0, drop=True)
    df['velocity'].fillna(0, inplace=True)
    
    avg_velocity = df.groupby('id')['velocity'].mean().reset_index(name='avg_velocity')
    time_detected = df.groupby('id').size().reset_index(name='time_detected')
    
    metrics_df = pd.merge(avg_velocity, time_detected, on='id')
    return metrics_df

def remove_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z_scores) < threshold]
def save_worm_images(worms, output_dir, original_frames, max_images=1):
    os.makedirs(output_dir, exist_ok=True)
    
    images_saved = 0
    
    for worm in worms:
        x, y, w, h = worm['bbox']
        worm_image = original_frames[worm['frame']][y:y+h, x:x+w]
        worm_id = worm['id']
        image_path = os.path.join(output_dir, f"worm_{worm_id}_frame_{worm['frame']}.png")
        cv2.imwrite(image_path, worm_image)
        images_saved += 1
        
        if images_saved >= max_images:
            break

# Open the video or HDF5 file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title='Select Video or HDF5 File')

previous_worms = []

if file_path.endswith('.avi'):
    video = cv2.VideoCapture(file_path)
    frame_idx = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = filedialog.askdirectory(title='Select Output Directory for Worm Images')
    
    original_frames = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        original_frames.append(frame)
        
        print(f"Processing frame {frame_idx + 1}/{total_frames}")
        
        current_worms = detect_worms(frame, frame_idx)
        merged_worms, split_worms = track_worms(previous_worms, current_worms)
        previous_worms = current_worms
        
        # Append detection data to DataFrame
        df = append_data_to_df(df, current_worms)
        
        # Show detections on the frame
        #frame_with_detections = show_detections(frame, current_worms)
        #cv2.imshow('Frame', frame_with_detections)
        
        # Save worm images
        avg_bbox_w = df['bbox_w'].mean()
        std_bbox_w = df['bbox_w'].std()
        avg_bbox_h = df['bbox_h'].mean()
        std_bbox_h = df['bbox_h'].std()
        
       # Update the function call for save_worm_images
        save_worm_images(current_worms, output_dir, original_frames, max_images=1)



        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    video.release()
    cv2.destroyAllWindows()

# Extract directory path from the video file path
directory = os.path.dirname(file_path)

# Remove outliers based on bounding box width and height
print("Removing outliers from the detection data")
df = remove_outliers(df, 'bbox_w')
df = remove_outliers(df, 'bbox_h')

# Save the tracking data to a CSV file
tracking_data_path = os.path.join(directory, 'tracking_data.csv')
df.to_csv(tracking_data_path, index=False)
print(f"Tracking data saved to {tracking_data_path}")

# Calculate and save metrics
print("Calculating metrics")
metrics_df = calculate_metrics(df)
metrics_path = os.path.join(directory, 'metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved to {metrics_path}")