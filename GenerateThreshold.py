import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
from Constants import TEMPLATES
import cv2

def compare_images(template_path, test_path):
    """
    Displays two images side by side with two buttons ("OK" and "Not OK") below.
    The images are resized to fit within a portion of the window.
    
    Parameters:
        template_path (str): Path to the template image (left side).
        test_path (str): Path to the test image (right side).
        
    Returns:
        str: "ok" if the OK button was clicked, "not ok" if the Not OK button was clicked.
    """
    # Dictionary to store the user's choice.
    result = {"value": None}

    # Create the main window.
    root = tk.Tk()
    root.title("Image Comparison")

    # Set a fixed window size.
    window_width, window_height = 800, 600
    root.geometry(f"{window_width}x{window_height}")

    # Define maximum size for each image.
    # For two side-by-side images, allocate roughly half the width (with padding)
    # and leave enough height for the images and the buttons.
    max_img_width = (window_width - 60) // 2  # 60 accounts for padding/margins.
    max_img_height = window_height - 150       # 150 pixels reserved for title, padding, and buttons.

    # Open the images.
    template_img = Image.open(template_path)
    test_img = Image.open(test_path)

    # Choose a resampling filter based on your Pillow version.
    try:
        # For Pillow >= 10.0.0
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # For older versions of Pillow
        resample_filter = Image.LANCZOS

    # Resize images while preserving aspect ratio.
    template_img.thumbnail((max_img_width, max_img_height), resample_filter)
    test_img.thumbnail((max_img_width, max_img_height), resample_filter)

    # Convert the PIL images to a format Tkinter can display.
    tk_template_img = ImageTk.PhotoImage(template_img)
    tk_test_img = ImageTk.PhotoImage(test_img)

    # Create a frame to hold the images.
    image_frame = tk.Frame(root)
    image_frame.pack(padx=10, pady=10)

    # Place the images side by side.
    template_label = tk.Label(image_frame, image=tk_template_img)
    template_label.grid(row=0, column=0, padx=10)
    test_label = tk.Label(image_frame, image=tk_test_img)
    test_label.grid(row=0, column=1, padx=10)

    # Create a frame for the buttons.
    button_frame = tk.Frame(root)
    button_frame.pack(padx=10, pady=10)

    # Define button callback functions.
    def on_ok():
        result["value"] = True
        root.destroy()

    def on_not_ok():
        result["value"] = False
        root.destroy()

    # Create the OK and Not OK buttons.
    ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side="left", padx=10)
    
    not_ok_button = tk.Button(button_frame, text="Not OK", command=on_not_ok, width=10)
    not_ok_button.pack(side="left", padx=10)

    # Start the Tkinter event loop.
    root.mainloop()

    return result["value"]

def save_frame_from_video(cap, frame_number, output_image_path):
    # Set the video position to the desired frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame.
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read frame {frame_number}")
        cap.release()
        return False

    # Save the frame as an image.
    success = cv2.imwrite(output_image_path, frame)
    if success:
        print(f"Frame {frame_number} saved successfully to {output_image_path}")
    else:
        print(f"Error: Unable to save the image to {output_image_path}")

    
    return success

def find_closest(current_val,list_of_elements):
    for element in list_of_elements:
        if element['result'] <= current_val :
            print(f"Found closest with { element['result']} to {current_val}")
            return element
    return None

def find_threshold(template_name, template_results, video_path):
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return False
    
    
    # get max
    max_value = template_results[0]["result"]
    template_path = TEMPLATES[template_name][0]
    test_path =  save_frame_from_video(cap,template_results[0]["frame"],"./assets/temp.png")
    res = compare_images(template_path,"./assets/temp.png")
    print(f"Res:{res}")
    if res == False :
        cap.release()
        return 1.0
    
    current_val = 0.9 * max_value

    while True:
        print(f"Current value is {current_val}")
        frame_data = find_closest(current_val,template_results)
        if frame_data == None:
            cap.release()
            return current_val * 0.9
        save_frame_from_video(cap,frame_data["frame"],"./assets/temp.png")
        res = compare_images(template_path,"./assets/temp.png")
        if res == False:
            cap.release()
            return current_val * 1.1
        else:
            current_val = current_val*0.9
   
    


def generate_thresholds(matching_results,video_path):
    thresholds = {}
    if len(matching_results) < 1:
        return
    keys = matching_results[0]
    lists_of_results = {key: [] for key in keys}
    for i,frame_result in enumerate(matching_results):
        for template_name in frame_result:
            lists_of_results[template_name].append({
                "frame":i,
                "result":frame_result[template_name]
            })

    
    for template_results in lists_of_results:
        print(f"template_res:{template_results}")
        sorted_data = sorted(lists_of_results[template_results], key=lambda x: x['result'], reverse=True)
        print(sorted_data[0])
        threshold = find_threshold(template_results,sorted_data,video_path)
        thresholds[template_results] = threshold
        print(f"Threshold for {template_results} is {threshold}")
        
    return thresholds