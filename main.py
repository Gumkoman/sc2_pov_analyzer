from Constants import TEMPLATES
from GenerateThreshold import generate_thresholds
from ProcessVideo import process_video
# from StateMachines import


if __name__ == "__main__":
    
    ### Init Data
    video_path = "./input_cut.mp4"
    # video_path = "./input_videos/2025-01-13 16-13-20.mp4"
    output_video_path = "test.mp4"
    templates_paths = TEMPLATES
    
    ### Process Video
    matching_list = process_video(video_path,output_video_path,templates_paths)
    ### Generate Thresholds
    for element in matching_list:
        print(f"maching type:{type(element)}, element: {element}")
    thresholds = generate_thresholds(matching_list,video_path)
    for element in thresholds:
        print(f"Threshold for {element} is {thresholds[element]}")
    #TODO Apply State machine
