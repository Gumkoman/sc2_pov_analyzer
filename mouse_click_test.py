import cv2
import os
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor


def find_best_template_in_frame(
    frame, 
    template_image_paths: Dict[str, Tuple[str, Tuple[float, float, float, float]]]
) -> Tuple[str, float, Tuple[int, int], Tuple[int, int]]:
    """
    Find the best matching template in a video frame, within specified percentage-based areas if provided.

    Args:
        frame (np.ndarray): A single video frame (image).
        template_image_paths (Dict[str, Tuple[str, Tuple[float, float, float, float]]]): 
            Dictionary mapping template paths to optional search areas as percentages (x%, y%, w%, h%).

    Returns:
        Tuple[str, float, Tuple[int, int], Tuple[int, int]]: 
            Best template path, match score, top-left corner, and bottom-right corner of the match.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame_gray.shape
    best_template = None
    best_score = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    for template_image_path, search_area_percentage in template_image_paths.items():
        # Load the template image
        template_image = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
        if template_image is None:
            print(f"Template image at {template_image_path} not found. Skipping.")
            continue

        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Calculate the search area based on percentages
        if search_area_percentage:
            x_percent, y_percent, w_percent, h_percent = search_area_percentage
            x = int(width * x_percent)
            y = int(height * y_percent)
            w = int(width * w_percent)
            h = int(height * h_percent)
            search_frame = frame_gray[y:y + h, x:x + w]
        else:
            search_frame = frame_gray

        # Perform template matching
        result = cv2.matchTemplate(search_frame, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Adjust coordinates for search area
        if search_area_percentage:
            max_loc = (max_loc[0] + x, max_loc[1] + y)

        # Update the best match if the current one is better
        if max_val > best_score:
            best_template = template_image_path
            best_score = max_val
            best_top_left = max_loc
            best_bottom_right = (max_loc[0] + template_image.shape[1], max_loc[1] + template_image.shape[0])

    return best_template, best_score, best_top_left, best_bottom_right


def process_video(
    video_path, 
    output_video_folder, 
    template_image_paths, 
    frame_skip=1
):
    """
    Process a single video, tracking mouse cursor and generating an output video.

    Args:
        video_path (str): Path to the input video.
        output_video_folder (str): Path to the folder for saving output videos.
        template_image_paths (Dict[str, Tuple[str, Tuple[float, float, float, float]]]): 
            Template paths with optional percentage-based search areas.
        frame_skip (int): Number of frames to skip for faster processing.

    Returns:
        None
    """
    input_video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_video_folder, f"{input_video_name}_tracked.avi")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for faster processing
        if frame_count % frame_skip != 0:
            continue

        # Find the best matching template
        best_template, best_score, best_top_left, best_bottom_right = find_best_template_in_frame(
            frame, template_image_paths
        )
        mouse_x1, mouse_y1 = best_top_left
        mouse_x2, mouse_y2 = best_bottom_right

        # Draw the bounding box and template name on the frame
        cv2.rectangle(frame, best_top_left, best_bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, f"{best_template.split('/')[-1]} ({best_score:.2f})", 
                    (mouse_x1, mouse_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")


def process_videos_in_folder_parallel(
    input_folder, 
    output_video_folder, 
    template_image_paths, 
    frame_skip=1, 
    max_workers=4
):
    """
    Process all videos in the input folder in parallel, generating output videos.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_video_folder (str): Path to the folder for saving output videos.
        template_image_paths (Dict[str, Tuple[str, Tuple[float, float, float, float]]]): 
            Template paths with optional percentage-based search areas.
        frame_skip (int): Number of frames to skip for faster processing.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        None
    """
    os.makedirs(output_video_folder, exist_ok=True)
    video_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_file in video_files:
            executor.submit(process_video, video_file, output_video_folder, template_image_paths, frame_skip)


# Example Usage
if __name__ == "__main__":
    input_video_folder = './input_videos'  # Folder containing input videos
    output_video_folder = './output_videos'  # Folder to save output videos
    template_image_paths = {
        'templates/mouse.png': None,  # Search full frame
        'templates/left_bracket.png': (0, 0, 1, 0.8),  # Search in bottom-right 20% of the frame
        'templates/left_bracket_gold.png': (0, 0, 1, 0.8),  # Search in bottom-right 20% of the frame
        'templates/left_camera.png': (0, 0, 0.2, 1),  # Search in bottom-right 20% of the frame
        'templates/target_center.png': None,  # Centered search area
        'templates/target_center_2.png': None,  # Centered search area
    }

    process_videos_in_folder_parallel(input_video_folder, output_video_folder, template_image_paths, frame_skip=1)
