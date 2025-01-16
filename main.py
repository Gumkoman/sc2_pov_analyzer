from Constants import TEMPLATES
# from GenerateThreshold import
from ProcessVideo import process_video
# from StateMachines import


if __name__ == "__main__":
    
    ### Init Data
    video_path = "./input_videos/2025-01-13 16-13-20.mp4"
    output_video_path = "test.mp4"
    templates_paths = TEMPLATES
    
    ### Process Video
    process_video(video_path,output_video_path,templates_paths)
    ### Generate Thresholds

    ### Apply State machine
