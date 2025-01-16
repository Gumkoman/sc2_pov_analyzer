import cv2
import os


def extract_frame(video_path, output_path, frame_number):
    """
    Extract a specific frame from a video and save it as a PNG file.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the extracted frame as a PNG file.
        frame_number (int): Frame number to extract (1-based index).

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 1 or frame_number > total_frames:
        print(f"Invalid frame number. The video has {total_frames} frames.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Set to the desired frame
    ret, frame = cap.read()
    if ret:
        # Save the frame as a PNG file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} extracted and saved to {output_path}")
    else:
        print(f"Failed to extract frame {frame_number}")

    cap.release()


# Example usage
if __name__ == "__main__":
    video_path = "C:\\Pasje\\AnalizaVideo\\test_input_maxpax_annotated.mp4"
    output_path = "C:\\Pasje\\AnalizaVideo\\output_frame.png"
    frame_number = 165  # Change this to the desired frame number

    extract_frame(video_path, output_path, frame_number)
