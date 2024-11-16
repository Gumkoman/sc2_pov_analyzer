import cv2
import numpy as np
import os

def detect_click_based_on_panel(frame, prev_frame, panel_coords, threshold=10):
    """
    Detects changes in the panel to infer mouse clicks.

    Args:
        frame (np.ndarray): The current frame of the video.
        prev_frame (np.ndarray): The previous frame of the video.
        panel_coords (Tuple[int, int, int, int]): Coordinates of the panel (x, y, width, height).
        threshold (float): Threshold for detecting significant change.

    Returns:
        bool: True if a significant change is detected in the panel, else False.
    """
    x, y, w, h = panel_coords
    panel_current = frame[y:y+h, x:x+w]
    panel_previous = prev_frame[y:y+h, x:x+w]

    # Convert to grayscale for comparison
    panel_current_gray = cv2.cvtColor(panel_current, cv2.COLOR_BGR2GRAY)
    panel_previous_gray = cv2.cvtColor(panel_previous, cv2.COLOR_BGR2GRAY)

    # Preprocessing: Blur to reduce noise
    panel_current_gray = cv2.GaussianBlur(panel_current_gray, (5, 5), 0)
    panel_previous_gray = cv2.GaussianBlur(panel_previous_gray, (5, 5), 0)

    # Compute absolute difference
    diff = cv2.absdiff(panel_current_gray, panel_previous_gray)
    mean_diff = np.mean(diff)

    # Debugging output
    print(f"Mean difference in panel: {mean_diff:.2f} (Threshold: {threshold})")

    return mean_diff > threshold


def find_best_template_in_frame(frame, template_image_paths):
    """
    Find the best matching template in a video frame.

    Args:
        frame (np.ndarray): A single video frame (image).
        template_image_paths (List[str]): List of paths to template images.

    Returns:
        Tuple[str, float, Tuple[int, int], Tuple[int, int]]: 
            Best template path, match score, top-left and bottom-right corners.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_template = None
    best_score = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    for template_image_path in template_image_paths:
        # Load the template image
        template_image = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
        if template_image is None:
            print(f"Template image at {template_image_path} not found. Skipping.")
            continue

        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update the best match if the current one is better
        if max_val > best_score:
            best_template = template_image_path
            best_score = max_val
            best_top_left = max_loc
            best_bottom_right = (max_loc[0] + template_image.shape[1], max_loc[1] + template_image.shape[0])

    return best_template, best_score, best_top_left, best_bottom_right


def process_video_with_click_detection(video_path, template_image_paths, output_frames_dir, output_video_path, panel_coords, threshold=10):
    """
    Process a video, detect template matches and click events based on panel changes, and create an output video.

    Args:
        video_path (str): Path to the video file.
        template_image_paths (List[str]): List of paths to template images.
        output_frames_dir (str): Directory to save processed frames.
        output_video_path (str): Path to the output video file.
        panel_coords (Tuple[int, int, int, int]): Coordinates of the panel for detecting mouse clicks.
        threshold (float): Threshold for detecting significant change.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(output_frames_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    prev_frame = None
    frame_count = 0
    click_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Detect click based on panel changes
        change_detected = False
        if prev_frame is not None:
            change_detected = detect_click_based_on_panel(frame, prev_frame, panel_coords, threshold)
            if change_detected:
                click_frames.append(frame_count)

        # Find the best matching template
        best_template, best_score, top_left, bottom_right = find_best_template_in_frame(frame, template_image_paths)

        # Annotate frame with template matching results
        if best_template:
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, f"{os.path.basename(best_template)}: {best_score:.2f}",
                        (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Always draw the panel box
        x, y, w, h = panel_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text if change is detected
        if change_detected:
            cv2.putText(frame, "Change Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        prev_frame = frame.copy()

        # Save the processed frame
        output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_frame_path, frame)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Mouse clicks detected at frames: {click_frames}")
    print("Video processing and output generation complete.")


# Example Usage
if __name__ == "__main__":
    video_path = 'input_video.mp4'  # Replace with your video file path
    template_image_paths = [
        'templates\\mouse.png',  
        'templates\\left_bracket_gold.png',  
        'templates\\left_bracket.png',  
        'templates\\left_camera.png',  
        'templates\\target_center.png',  
        'templates\\target_center_2.png',  
        'templates\\right_bracket.png',  
    ]
    output_frames_dir = 'output_frames'  # Directory to save frames with changes drawn
    output_video_path = 'output_video.mp4'  # Path for the generated video
    panel_coords = (1564, 838, 297, 198)  # Example: x=100, y=200, width=150, height=50

    process_video_with_click_detection(video_path, template_image_paths, output_frames_dir, output_video_path, panel_coords, threshold=10)
