import cv2
from typing import Dict, Tuple

from Constants import BuildingState, Template

#TODO make it better, problem that it should resolve is that video 
# resolution might be diffrent to orginal resolution of screenshoot
# that is base of template
def load_templates(
    templates: Dict[str, Tuple[str, Tuple[float, float, float, float]]],
    frame_width: int,
    frame_height:int
):
    loaded_templates = []
    for name, (path, region) in templates.items():

        template_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template_grey is None:
            print(f"Cannot load template: {path}")
            continue

        # Calculate search region size in pixels
        if region:
            x_percent, y_percent, w_percent, h_percent = region
            search_w = int(w_percent * frame_width)
            search_h = int(h_percent * frame_height)
        else:
            search_w, search_h = frame_width, frame_height

        template_h, template_w = template_grey.shape

        # Check if resizing is needed
        if template_w > search_w or template_h > search_h:
            scaling_factor = min(search_w / template_w, search_h / template_h)
            new_width = max(1, int(template_w * scaling_factor))
            new_height = max(1, int(template_h * scaling_factor))
            template_grey = cv2.resize(template_grey, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized template '{name}' to {new_width}x{new_height} to fit within search region.")
        else:
            print(f"Template '{name}' fits within the search region. No resizing needed.")
        temp = Template(name,region,template_grey)
        loaded_templates.append(temp)
    return loaded_templates

def generate_templates_probability(
    templates,
    frame_width,
    frame_height,
    frame_gray
):
    result = {}
    for template in templates:
        if template.region:
            x_percent, y_percent, w_percent, h_percent = template.region
            x = int(x_percent * frame_width)
            y = int(y_percent * frame_height)
            w = int(w_percent * frame_width)
            h = int(h_percent * frame_height)
            search_area = frame_gray[y:y + h, x:x + w]
        else:
            search_area = frame_gray
            x, y = 0, 0
            w, h = frame_width, frame_height

        template_h, template_w = template.template_grey.shape
        if search_area.shape[0] < template_h or search_area.shape[1] < template_w:
            print(f"Warning: After resizing, template '{template.name}' is still larger than the search area. Skipping this template.")
            continue
        match_template = cv2.matchTemplate(search_area, template.template_grey, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_template)
        if template.region:
            max_loc = (max_loc[0] + x, max_loc[1] + y)         
        result[template.name] = max_val
    return result

def test_states(frame, templates_probability,current_state):

    thresholds = {
        "worker_selected": 0.55,#
        "test_to_spray_3": 0.9,#
        "basic_building_menu": 0.85,#
        "advanced_building_menu": 0.7,#
        "build_selection": 0.38,#
    }
    # print(f"templates_probability {templates_probability} ")

    is_worker_selected = (
        (templates_probability["worker_selected"] > thresholds["worker_selected"]) or
        (templates_probability["test_to_spray_3"] > thresholds["test_to_spray_3"])
    )
    is_basic_building_menu = (
        (templates_probability["basic_building_menu"] > thresholds["basic_building_menu"]) or
        (templates_probability["advanced_building_menu"] > thresholds["advanced_building_menu"])
    )
    is_advanced_building_menu = templates_probability["build_selection"] > thresholds["build_selection"]
    procedure_counter=0
    match current_state:
        case BuildingState.IDLE:
            if is_worker_selected:
                current_state = BuildingState.WORKER_SELECTED
        case BuildingState.WORKER_SELECTED:
            if is_basic_building_menu:
                current_state = BuildingState.BUILDING_MENU
            elif not is_worker_selected:
                current_state = BuildingState.IDLE
        case BuildingState.BUILDING_MENU:
            if is_advanced_building_menu:
                current_state = BuildingState.PLACE_BUILDING
        case BuildingState.PLACE_BUILDING:
            if is_worker_selected and not is_advanced_building_menu:
                current_state = BuildingState.FINISHED
        case BuildingState.FINISHED:
            current_state = BuildingState.IDLE
            print("FINISHED")

    start_x,start_y = 10 ,40
    cv2.putText(frame,f"State:{current_state}",(start_x,start_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    for name in thresholds:
        start_y += 40
        if templates_probability[name] > thresholds[name]:
            cv2.putText(frame,f"{name}:\t{templates_probability[name]:.2f}",(start_x,start_y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,f"{name}:\t{templates_probability[name]:.2f}",(start_x,start_y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)


    return frame,current_state



def process_video(
    video_path:str,
    output_video_path:str,
    templates:dict
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #
    loaded_templates = load_templates(templates,frame_width,frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    current_state = BuildingState.IDLE

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        templates_probability = generate_templates_probability(loaded_templates,frame_width,frame_height,frame_gray)

        
        new_frame,current_state = test_states(frame,templates_probability,current_state)

        out.write(new_frame)
    
    cap.release()
    out.release()