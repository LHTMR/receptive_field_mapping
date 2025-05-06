import cv2
import numpy as np
import tempfile
import os

def process_video_with_roi(input_path, output_path):
    """
    Process a video by allowing the user to select a Region of Interest (ROI)
    on the middle frame, then crop all frames based on the ROI and save the processed video.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the processed video.
    """
    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties: FPS={original_fps}, Width={original_width}, Height={original_height}, Total Frames={total_frames}")

    # Seek to the middle frame
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # Read the middle frame
    ret, middle_frame = cap.read()
    if not ret:
        print("Error: Cannot read the middle frame.")
        cap.release()
        return

    # Let the user select the ROI
    roi = cv2.selectROI("Select ROI", middle_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("Error: No ROI selected.")
        cap.release()
        return

    x, y, w, h = roi
    print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

    # Set the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (w, h))

    # Process each frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the selected ROI
        cropped_frame = frame[y:y + h, x:x + w]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Processed video saved to {output_path}")

def convert_all_videos_in_directory(input_dir, output_dir):
    """
    Convert all videos in a directory by allowing the user to select an ROI
    for each video and saving the processed videos to the output directory.

    Args:
        input_dir (str): Path to the directory containing input videos.
        output_dir (str): Path to the directory where processed videos will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mov') or filename.endswith('.mp4'):
            input_video = os.path.join(input_dir, filename)
            output_video = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_converted.mp4")

            print(f"Processing {filename}...")
            process_video_with_roi(input_video, output_video)

    print("All videos converted.")


#input_dir = r'C:\Python Programming\LIU\Data\Videos\2025-04-29_train_white_tip'
#output_dir = r'C:\Python Programming\LIU\Data\Videos\2025-04-29_train_white_tip\Converted'
#convert_all_videos_in_directory(input_dir, output_dir)