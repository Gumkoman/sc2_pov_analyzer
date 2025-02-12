from Constants import TEMPLATES
from GenerateThreshold import generate_thresholds
from ProcessVideo import process_video
from StateMachines import execute_state_machine
import cv2
def help_function(video_path,start,end,name):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start < 0 or start >= total_frames:
        print("Error: 'start' frame is out of bounds.")
        cap.release()
        return
    if end < start or end >= total_frames:
        print("Error: 'end' frame is out of bounds or less than 'start' frame.")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object.
    # Here, 'mp4v' is used for mp4 files; you can change it if needed.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    
    # Set the capture to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    current_frame = start
    while current_frame <= end:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame {current_frame}. Stopping.")
            break
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Clip saved to {name}")
if __name__ == "__main__":
    
    ### Init Data
    # video_path = "./input_cut.mp4"
    video_path = "./input_videos/2025-01-13 16-13-20.mp4"
    output_video_path = "test.mp4"
    templates_paths = TEMPLATES
    
    ### Process Video
    matching_list = process_video(video_path,output_video_path,templates_paths)
    ### Generate Thresholds
    thresholds = generate_thresholds(matching_list,video_path)
    procedure_list = execute_state_machine(matching_list,thresholds)
    avg = 0
    for i,element in enumerate(procedure_list):
        help_function(video_path,element["start"],element["end"],f"res\\procedure_{i}.mp4")
        avg += element['procedure_time']

    print(f"Average time in frames is {(avg/len(procedure_list)):.2f}")