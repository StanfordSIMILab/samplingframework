import os
import cv2

def process_video(video_path, video_output_dir):
    """
    Extract frames from a video at a specified interval and save them to a directory.

    Parameters:
    video_path (str): Path to the video file.
    video_output_dir (str): Directory where the extracted frames will be saved.
    frame_interval (int): Interval at which frames will be saved.
    """

    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames to capture at 10 fps
    frame_interval = int(fps / 10)

    frame_count = 0
    saved_count = 1

    while True:
        ret, frame = cap.read()
        
        # Break the loop if there are no more frames
        if not ret:
            break
        
        # Only save frames at the specified interval
        if frame_count % frame_interval == 0:
            filename = os.path.join(video_output_dir, f"{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)  # Save the frame as a JPEG file
            saved_count += 1
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_count - 1} frames from {video_path} at {frame_interval} fps.")

def process_all_videos(input_dir, output_dir):
    """
    Process all video files in the input directory and its subdirectories to extract frames.

    Parameters:
    input_dir (str): Directory containing video files.
    output_dir (str): Directory where the extracted frames will be saved.
    """
    # Convert input and output directories to absolute paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting frames from videos in {input_dir} to {output_dir}...")

    # Walk through the input directory and process each video file
    for root, dirs, files in os.walk(input_dir):
        print(f"Processing folder: {root}")
        for file in files:
            # Check if the file is a video file based on its extension
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
                video_path = os.path.join(root, file)
                print(f"Found video: {video_path}")

                # Create the corresponding output directory structure
                relative_path = os.path.relpath(root, input_dir)
                video_name = os.path.splitext(file)[0]
                video_output_dir = os.path.join(output_dir, relative_path, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                print (video_output_dir)
                process_video(video_path, video_output_dir)