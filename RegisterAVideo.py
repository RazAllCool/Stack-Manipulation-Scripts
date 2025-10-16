import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def register_frames_sift(frame1_gray, frame2_gray, prev_homography=None):
    """
    Registers frame2_gray to frame1_gray using SIFT and returns the homography.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(frame1_gray, None)
    kp2, des2 = sift.detectAndCompute(frame2_gray, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("Warning: Not enough descriptors found for SIFT matching.")
        return prev_homography if prev_homography is not None else np.eye(3)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Ensure descriptors are of type CV_32F for FLANN
    if des1.dtype != np.float32:
        des1 = np.float32(des1)
    if des2.dtype != np.float32:
        des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    # Need to handle cases where matches might be empty or not have 2 neighbors
    for i in range(len(matches)):
        if len(matches[i]) == 2:
            m, n = matches[i]
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0) # Transform from frame2 to frame1
        if H is None:
            print("Warning: Homography estimation failed. Using previous or identity matrix.")
            return prev_homography if prev_homography is not None else np.eye(3)
        return H
    else:
        print(f"Warning: Not enough good matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return prev_homography if prev_homography is not None else np.eye(3)

def process_video_batchwise(input_video_path, output_video_path, batch_size=50):
    """
    Loads a video, corrects drift in batches using SIFT, and saves the output.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # For .avi
    # If output_video_path ends with .mp4, you might prefer 'MP4V' or 'avc1'
    # For example: if output_video_path.lower().endswith(".mp4"):
    #                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open output video for writing: {output_video_path}")
        cap.release()
        return

    print(f"Processing video: {input_video_path}")
    print(f"Outputting to: {output_video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    reference_frame_gray = None
    accumulated_homography = np.eye(3)
    frames_processed_count = 0
    first_frame_written = False

    while True:
        frames_in_batch = []
        original_frames_in_batch = []

        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames_in_batch.append(frame)
            original_frames_in_batch.append(frame.copy())
        
        if not frames_in_batch:
            break

        print(f"\nProcessing batch starting around frame {frames_processed_count + 1}...")

        if reference_frame_gray is None and frames_in_batch:
            reference_frame_gray = cv2.cvtColor(frames_in_batch[0], cv2.COLOR_BGR2GRAY)
            print("Set initial reference frame.")
            out.write(frames_in_batch[0])
            first_frame_written = True
            frames_processed_count += 1
            start_index_in_batch = 1
        elif not first_frame_written and frames_in_batch: # Should only happen if first batch was empty, then not.
            reference_frame_gray = cv2.cvtColor(frames_in_batch[0], cv2.COLOR_BGR2GRAY)
            print("Set initial reference frame (delayed).")
            out.write(frames_in_batch[0])
            first_frame_written = True
            frames_processed_count += 1
            start_index_in_batch = 1
        else:
            start_index_in_batch = 0
        
        if reference_frame_gray is None:
            print("Error: No reference frame could be set. Writing original frames for this batch.")
            for original_frame in original_frames_in_batch[start_index_in_batch:]:
                 out.write(original_frame)
            frames_processed_count += len(original_frames_in_batch[start_index_in_batch:])
            continue

        for i in range(start_index_in_batch, len(frames_in_batch)):
            current_frame = frames_in_batch[i]
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            print(f"  Registering frame {frames_processed_count + 1}...")
            
            H_current_to_ref = register_frames_sift(reference_frame_gray, current_frame_gray, accumulated_homography)

            if H_current_to_ref is not None and not np.array_equal(H_current_to_ref, np.eye(3)):
                corrected_frame = cv2.warpPerspective(current_frame, H_current_to_ref, (frame_width, frame_height))
                out.write(corrected_frame)
                # Update accumulated_homography if you were chaining frame-to-frame.
                # For fixed reference, this just stores the last good one for potential fallback.
                accumulated_homography = H_current_to_ref 
                print(f"    Frame {frames_processed_count + 1} registered and written.")
            else:
                print(f"    Warning: Registration failed for frame {frames_processed_count + 1}. Writing original frame.")
                out.write(current_frame)

            frames_processed_count += 1
            if frames_processed_count % 10 == 0:
                print(f"  Processed {frames_processed_count}/{total_frames} frames...")

    print(f"\nFinished processing. Total frames processed/written: {frames_processed_count}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Corrected video saved to: {output_video_path}")

# --- Main execution ---
if __name__ == '__main__':
    # --- Parameters ---
    batch_processing_size = 30 # Number of frames to load into memory at a time

    # --- Set up Tkinter for file dialog ---
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    # --- Ask user to select an input video file ---
    input_video = filedialog.askopenfilename(
        title="Select Input Video File",
        filetypes=(("AVI files", "*.avi"), 
                   ("MP4 files", "*.mp4"),
                   ("MOV files", "*.mov"),
                   ("All files", "*.*"))
    )

    if not input_video:
        print("No input file selected. Exiting.")
    else:
        print(f"Input video selected: {input_video}")

        # --- Determine output video path ---
        # Suggest an output name based on the input name
        input_dir, input_filename_ext = os.path.split(input_video)
        input_filename, input_ext = os.path.splitext(input_filename_ext)
        output_video = os.path.join(input_dir, f"{input_filename}_registered{input_ext}")
        
        # Optionally, ask the user where to save the output file
        output_video = filedialog.asksaveasfilename(
             title="Save Registered Video As",
             defaultextension=".avi",
             initialfile=f"{input_filename}_registered{input_ext}",
             filetypes=(("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*"))
         )

        if not output_video: # If using asksaveasfilename and user cancels
             print("No output file specified. Exiting.")
        else:
            if os.path.exists(input_video):
                process_video_batchwise(input_video, output_video, batch_size=batch_processing_size)
            else:
                # This case should ideally not be reached if filedialog returns a valid path
                print(f"Error: Selected input video '{input_video}' does not exist. This should not happen.")
    
    # Clean up Tkinter root window if it was used
    # No explicit root.destroy() needed if root.withdraw() was the only interaction for a simple dialog.
    # However, if more complex GUI elements were planned, proper destruction is good.
    # For a simple filedialog, it often manages its lifecycle well.