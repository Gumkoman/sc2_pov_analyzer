import cv2
import os
from typing import Dict, Tuple

def detect_templates_in_video(
    video_path: str,
    templates: Dict[str, Tuple[str, Tuple[float, float, float, float]]],
    output_video_path: str,
    thresholds: Dict[str, float] = None
):
    """
    Detect templates in a video with optional region-based detection and annotate with probabilities.

    Args:
        video_path (str): Path to the input video.
        templates (Dict[str, Tuple[str, Tuple[float, float, float, float]]]):
            A dictionary mapping template names to their paths and search regions (x%, y%, w%, h%).
        output_video_path (str): Path to save the annotated video.
        thresholds (Dict[str, float]): Detection thresholds for each template.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Load and resize templates
    loaded_templates = {}
    for name, (path, region) in templates.items():
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Cannot load template: {path}")
            continue

        # Calculate search region size in pixels
        if region:
            x_percent, y_percent, w_percent, h_percent = region
            search_w = int(w_percent * frame_width)
            search_h = int(h_percent * frame_height)
        else:
            search_w, search_h = frame_width, frame_height

        template_h, template_w = template.shape

        # Check if resizing is needed
        if template_w > search_w or template_h > search_h:
            scaling_factor = min(search_w / template_w, search_h / template_h)
            new_width = max(1, int(template_w * scaling_factor))
            new_height = max(1, int(template_h * scaling_factor))
            template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized template '{name}' to {new_width}x{new_height} to fit within search region.")
        else:
            print(f"Template '{name}' fits within the search region. No resizing needed.")

        loaded_templates[name] = template

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 100 == 0:
            print(f"###{frame_count}/{total_frames}")

        # Prepare a list to hold annotation texts
        annotations = []
        # Define starting position for annotations (e.g., top-left corner)
        start_x = 10
        start_y = 20
        line_height = 20  # Space between lines

        for name, (path, region) in templates.items():
            if name not in loaded_templates:
                continue

            template = loaded_templates[name]

            # Determine the search region
            if region:
                x_percent, y_percent, w_percent, h_percent = region
                x = int(x_percent * frame_width)
                y = int(y_percent * frame_height)
                w = int(w_percent * frame_width)
                h = int(h_percent * frame_height)
                search_area = frame_gray[y:y + h, x:x + w]
            else:
                search_area = frame_gray
                x, y = 0, 0
                w, h = frame_width, frame_height

            # Check if the search area is larger than the template
            template_h, template_w = template.shape
            if search_area.shape[0] < template_h or search_area.shape[1] < template_w:
                print(f"Warning: After resizing, template '{name}' is still larger than the search area. Skipping this template.")
                continue

            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            threshold = thresholds.get(name, 0.5) if thresholds else 0.5

            # Determine text color based on threshold
            if max_val >= threshold:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            # Prepare annotation text
            annotation_text = f"{name}: {max_val:.2f}"
            annotations.append((annotation_text, color))

            # Optional: Draw rectangle and text only if above threshold
            # Uncomment the following block if you still want to draw rectangles for detections above threshold
            """
            if max_val >= threshold:
                top_left = max_loc if not region else (max_loc[0] + x, max_loc[1] + y)
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

                # Annotate the frame with rectangle
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({max_val:.2f})", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            """

        # Draw all annotations on the frame
        for idx, (text, color) in enumerate(annotations):
            position = (start_x, start_y + idx * line_height)
            cv2.putText(frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_video_path}")

# Example usage
if __name__ == "__main__":
    # video_path = "input_videos/input_maxpax.mp4"
    video_path = "2025-01-13 16-13-20.mp4"
    output_video_path = "./my_res_1.mp4"
    # video_path = "./gumek_test_input.mp4"
    # output_video_path = "./gumek_result.mp4"
    region1 = (1027/1280, 563/720, 233/1280, 140/720)
    region2 = (250/1280, 590/720, 650/1280, 110/720)
    templates = {
        "advanced_building_menu": ("./new_assets/advanced_building_menu.png", region1),
        "asymilator": ("./new_assets/asymilator.png", region1),
        "basic_building_menu": ("./new_assets/basic_building_menu.png", region1),
        "battery": ("./new_assets/battery.png", region1),
        "build_selection": ("./new_assets/build_selection.png", region1),
        "cyber_core": ("./new_assets/cyber_core.png", region1),
        "fleat_beacon": ("./new_assets/fleat_beacon.png", region1),
        "forge": ("./new_assets/forge.png", region1),
        "gateway": ("./new_assets/gateway.png", region1),
        "nexus": ("./new_assets/nexus.png", region1),
        "robo": ("./new_assets/robo.png", region1),
        "robobay": ("./new_assets/robobay.png", region1),
        "stargate": ("./new_assets/stargate.png", region1),
        "templar_archives": ("./new_assets/templar_archives.png", region1),
        "test_to_spray": ("./new_assets/test_to_spray.png", region1),
        "twighlight": ("./new_assets/twighlight.png", region1),
        "worker_selected": ("./new_assets/worker_selected.png", region2),
        "selected_workers": ("./new_assets/selected_workers.png", region2),
    }
    thresholds = {
        "advanced_building_menu": 0.5,
        "asymilator": 0.8,
        "basic_building_menu": 0.5,
        "battery": 0.8,
        "build_selection": 0.5,
        "cyber_core": 0.8,
        "fleat_beacon": 0.8,
        "forge": 0.8,
        "gateway": 0.8,
        "nexus": 0.8,
        "robo": 0.8,
        "robobay": 0.8,
        "stargate": 0.8,
        "templar_archives": 0.8,
        "test_to_spray": 0.65,
        "twighlight": 0.8,
        "worker_selected": 0.5,
        "selected_workers": 0.7,
    }

    detect_templates_in_video(video_path, templates, output_video_path, thresholds)
