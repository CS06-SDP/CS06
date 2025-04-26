from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage
import threading
import zipfile
import pandas as pd

# Model Paths
MODEL_PATH_1 = r"D:\SDP2 NEW\trial 2_sdp2_only\runs\detect\train\weights\best.pt"  # Cheating & Phone
MODEL_PATH_2 = r"D:\SDP2 NEW\trial 3_sdp2_with_EEDPT\runs\detect\train\weights\best.pt"  # PhoneCall & Texting
MODEL_PATH_3 = r"D:\CS06 SDP I\runs\detect\train15\weights\best.pt"  # Cup, Mug, Bottle

# Video Path
#VIDEO_PATH = r"D:\SDP2 NEW\cheating_videos_rodha\Xray0017 C3-00-133352-133359.mp4"
#VIDEO_PATH = r"D:\SDP2 NEW\cheating_videos_rodha\Xray0017 C1-00-122122-122132.mp4"
#VIDEO_PATH = r"D:\SDP2 NEW\cheating_videos_rodha\Xray0017 C1-00-122707-122723.mp4"
#VIDEO_PATH= r"C:\Users\Zayed\Downloads\04172025 2\04172025\Xray0017 C3-00-132422-132432.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C1-00-130730-130741.mp4"

#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C1-00-130755-130807.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C3-00-132118-132135.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\untouched vids\untouched vids\Xray0017 C3-00-132118-132135.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\untouched vids\untouched vids\Xray0017 C3-00-132158-132212.mp4"
VIDEO_PATH= r"C:\Users\Zayed\Downloads\sdp2videos\sdp2videos\malecheatingvid2_Cropinend.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C3-00-132344-132353.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C3-00-132422-132432.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C3-00-132522-132536.mp4"
#VIDEO_PATH= r"D:\CS06 SDP I\Testing videos\Video43.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C1-00-130505-130517.mp4"

# Output directory for saved recordings
OUTPUT_DIR = r"D:\SDP2 NEW\INTERFACES_SR\Interface_Cheating"

# Load YOLOv8 models
print("Loading models...")
try:
    model1 = YOLO(MODEL_PATH_1)
    print(f"Model 1 loaded with {len(model1.names) if hasattr(model1, 'names') else '?'} classes")
except Exception as e:
    print(f"Error loading Model 1: {e}")
    model1 = None

try:
    model2 = YOLO(MODEL_PATH_2)
    print(f"Model 2 loaded with {len(model2.names) if hasattr(model2, 'names') else '?'} classes")
except Exception as e:
    print(f"Error loading Model 2: {e}")
    model2 = None

try:
    model3 = YOLO(MODEL_PATH_3)
    print(f"Model 3 loaded with {len(model3.names) if hasattr(model3, 'names') else '?'} classes")
except Exception as e:
    print(f"Error loading Model 3: {e}")
    model3 = None

# Set confidence threshold lower for better detection
if model1: model1.conf = 0.2  # Lower from 0.25
if model2: model2.conf = 0.2  # Lower from 0.25
if model3: model3.conf = 0.1  # Keep as is

# Print model class mappings
print("\nModel 1 class names:", model1.names if hasattr(model1, 'names') else "Not accessible")
print("Model 2 class names:", model2.names if hasattr(model2, 'names') else "Not accessible")
print("Model 3 class names:", model3.names if hasattr(model3, 'names') else "Not accessible")

# Updated Class Mapping based on data.yaml
CLASS_NAMES = {
    0: "Cup", 1: "MainFrame", 2: "Bottle", 3: "Human", 4: "Snack", 5: "Mug", 
    6: "Biscuit", 7: "SafetyJacket", 8: "Helmet", 9: "Eating", 10: "Drinking", 
    11: "Classroom", 12: "Phone", 13: "PhoneCall", 14: "Texting", 15: "Cheating"
}

# List of class IDs that should be treated as cheating behaviors
CHEATING_CLASS_IDS = [13, 14, 15]  # PhoneCall, Texting, Cheating

# Color scheme (BGR format)
COLORS = {
    "Cheating": (0, 0, 255),     # Red for cheating behaviors
    "Phone": (0, 165, 255),      # Orange for phone
    "PhoneCall": (0, 0, 255),    # Red (treated as cheating)
    "Texting": (0, 0, 255),      # Red (treated as cheating)
    "Cup": (150, 100, 0),        # Light blue for cup
    "Bottle": (179, 132, 25),    # Medium blue for bottle
    "Mug": (204, 153, 51),       # Dark blue for mug
    "header_bg": (77, 43, 30),   # Dark navy blue for header
    "panel_bg": (30, 30, 30),    # Dark gray for panels
    "panel_title_bg": (65, 40, 25),  # Darker blue for panel titles
    "text_normal": (255, 255, 255),  # White for normal text
    "text_subtitle": (200, 200, 200), # Light gray for subtitles
    "live_indicator": (0, 100, 255)    # Red for live indicator
}

# Event text colors
EVENT_COLORS = {
    "Phone": (0, 165, 255),
    "PhoneCall": (0, 0, 255),     # bright red
    "Texting": (0, 0, 255),       # bright red
    "CHEATING": (0, 0, 255),      # bright red
    "Cheating": (0, 0, 255),      # bright red
    "Cup": (150, 100, 0),
    "Bottle": (179, 132, 25),
    "Mug": (204, 153, 51)
}

# Helper function to get class name consistently
def get_class_name(model_result, cls_id):
    """Get class name consistently across all models"""
    # First try using model's internal names if available
    if hasattr(model_result, 'names') and cls_id in model_result.names:
        return model_result.names[cls_id]
    # Then try our custom mapping
    elif cls_id in CLASS_NAMES:
        return CLASS_NAMES[cls_id]
    # Fallback to generic name
    else:
        return f"Class{cls_id}"

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo properties: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define fourcc for video writing
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize variables for cheating detection recording
is_cheating_detected = False
cheating_out = None
cheating_record_start_time = None
cheating_recording_duration = 5  # Duration to record in seconds after cheating is detected

# Create main display window with title
window_name = "EXAM MONITORING - HALL 1"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Detection log for display
detection_log = []
max_log_entries = 10  # We can store more entries but will display what fits

# Create a simple Khalifa University logo placeholder
def create_ku_logo():
    """Create a better-looking Khalifa University logo placeholder"""
    logo = np.zeros((60, 250, 3), dtype=np.uint8)
    # Create a gradient background (dark blue to light blue)
    for i in range(logo.shape[1]):
        blue_gradient = int(40 + (i / logo.shape[1]) * (100 - 40))
        logo[:, i] = [blue_gradient, int(blue_gradient/2), 0]  # Dark blue gradient
    
    # Draw a border
    cv2.rectangle(logo, (0, 0), (logo.shape[1]-1, logo.shape[0]-1), (120, 80, 0), 2)
    
    # Add text "KHALIFA UNIVERSITY" with improved styling
    cv2.putText(logo, "KHALIFA", (15, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(logo, "UNIVERSITY", (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
    
    return logo

# Try to load the university logo from multiple possible paths
def load_university_logo():
    possible_paths = [ r"D:\SDP2 NEW\university_cheating_sys_rodha\logo.png" ]
    for path in possible_paths:
        try:
            print(f"Trying to load logo from: {path}")
            logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if logo is not None:
                print(f"Successfully loaded logo from: {path}")
                # Resize logo to appropriate size
                logo_height = 60
                aspect_ratio = logo.shape[1] / logo.shape[0]
                logo_width = int(logo_height * aspect_ratio)
                logo = cv2.resize(logo, (logo_width, logo_height))
                return logo, True
        except Exception as e:
            print(f"Failed to load logo from {path}: {e}")
    # If all paths fail, create a placeholder
    print("All logo paths failed, using placeholder")
    return create_ku_logo(), True

# Modified logo placement code to ensure it's visible
def place_logo_in_corner(canvas, logo, header_height):
    """Places the logo in the top left corner ensuring it's properly visible"""
    # Make sure the logo fits within the header height
    logo_y_offset = max(0, (header_height - logo.shape[0]) // 2)
    logo_height = min(header_height, logo.shape[0])
    logo_width = logo.shape[1]
    
    # Create a clear background for the logo to ensure visibility
    cv2.rectangle(canvas, (10, logo_y_offset), 
                 (10 + logo_width + 10, logo_y_offset + logo_height), 
                 COLORS["header_bg"], -1)
    
    try:
        # Get the region to place the logo
        logo_region = canvas[logo_y_offset:logo_y_offset+logo_height, 10:10+logo_width]
        
        # Check if regions have compatible shapes
        if logo_region.shape[0] > 0 and logo_region.shape[1] > 0 and logo.shape[0] > 0 and logo.shape[1] > 0:
            # Handle different image types
            if len(logo.shape) == 3 and logo.shape[2] == 4:  # With alpha channel
                # Use alpha blending for transparent logos
                alpha_logo = logo[:logo_height, :, 3] / 255.0
                for c in range(0, 3):
                    logo_region[:, :, c] = (1 - alpha_logo) * logo_region[:, :, c] + alpha_logo * logo[:logo_height, :, c]
            elif len(logo.shape) == 3 and logo.shape[2] == 3:  # RGB without alpha
                # Direct copy for RGB images
                logo_region[:] = logo[:logo_height, :, :]
            else:
                # For grayscale images, convert to BGR
                logo_region[:] = cv2.cvtColor(logo[:logo_height, :], cv2.COLOR_GRAY2BGR)
                
            # Add a subtle border around the logo
            cv2.rectangle(canvas, (9, logo_y_offset-1), 
                         (11 + logo_width, logo_y_offset + logo_height+1), 
                         (200, 200, 200), 1)
            
            return True
    except Exception as e:
        print(f"Error placing logo: {e}")
    
    return False

# Try to load the university logo
logo_path = "khalifa_university_logo.png"  # Replace with actual path if available
try:
    # Try to load from multiple possible paths
    logo, has_logo = load_university_logo()
    print(f"Logo loaded with shape: {logo.shape}")
except Exception as e:
    print(f"Logo loading error: {e}")
    logo = create_ku_logo()
    has_logo = True

# Create pulsing live indicator animation
def create_live_indicator(frame, pulse_cycle, video_width):
    live_x = video_width - 120   # adjust as needed
    live_y = 100                  # vertical position
    pulse_alpha = abs(np.sin(pulse_cycle * 0.2)) * 0.5 + 0.5

    # Larger background rectangle
    rect_width = 110
    rect_height = 40
    cv2.rectangle(frame, (live_x, live_y - rect_height // 2), 
                  (live_x + rect_width, live_y + rect_height // 2), 
                  (40, 40, 40), -1)

    # Animated border
    cv2.rectangle(frame, (live_x, live_y - rect_height // 2), 
                  (live_x + rect_width, live_y + rect_height // 2), 
                  (0, 0, int(255 * pulse_alpha)), 3)

    # Bigger red dot
    cv2.circle(frame, (live_x + 20, live_y), 12, (0, 0, int(255 * pulse_alpha)), -1)

    # Bigger text
    cv2.putText(frame, "LIVE", (live_x + 40, live_y + 8),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

# Add debug logging
def log_debug(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Add a function to draw cheating alert banner at the bottom
def draw_cheating_alert_banner(frame, confidence=None):
    # Draw a red banner at the bottom of the frame
    banner_height = 60
    # Calculate bottom position
    bottom_y = frame.shape[0] - banner_height
    
    # Draw the rectangle at the bottom
    cv2.rectangle(frame, (0, bottom_y), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    
    if confidence:
        cv2.putText(frame, f"ALERT: CHEATING BEHAVIOR DETECTED (Conf: {confidence:.2f})", 
                    (10, bottom_y + 40),  # slightly lower if font is bigger
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,  # 🔥 make text larger
                    (255, 255, 255), 3)             # 🔥 make it bolder
    else:
        cv2.putText(frame, "ALERT: CHEATING BEHAVIOR DETECTED", 
                    (10, bottom_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3)

# ======== ORGANIZED VIDEO OUTPUT ========
session_date = datetime.now().strftime("%Y-%m-%d")
session_time = datetime.now().strftime("%H-%M-%S")
room_name = "Room_C02021"

session_folder = os.path.join(OUTPUT_DIR, room_name, session_date)
os.makedirs(session_folder, exist_ok=True)

output_path = os.path.join(session_folder, f"{room_name}_{session_date}_{session_time}.mp4")
main_out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
# ========================================

last_cheating_label = "Cheating"
frame_count = 0
total_cheating_detections = 0
cheating_confidence = 0.0

# Process video
pulse_counter = 0  # For live indicator animation
slow_log_update_counter = 0  # To slow down log updates for better visibility
slow_update_interval = 10  # Increased interval for even slower updates (better visibility)

# Determine layout dimensions
# We'll create a canvas that's wider than the original video to accommodate the side panel
video_panel_width = int(original_width * 0.7)  # Video takes 70% of width
log_panel_width = int(original_width * 0.3)    # Log takes 30% of width
canvas_width = video_panel_width + log_panel_width

# Use the original video height
canvas_height = original_height

# Define header height
header_height = 75

print("Starting video processing...")

while cap.isOpened():
    success, frame = cap.read()
    human_boxes = []
    if not success:
        break  # End of video

    frame_count += 1
    pulse_counter += 1  # Increment for animation

    # Progress reporting
    if frame_count % 100 == 0:
        percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        log_debug(f"Processing frame {frame_count}/{total_frames} ({percent_complete:.1f}%)")

    # Create a new canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas.fill(50)  # Dark gray background

    # Create header across the entire canvas
    cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), COLORS["header_bg"], -1)

    # Make sure the logo fits within the header height
    logo_y_offset = max(0, (header_height - logo.shape[0]) // 2)
    logo_height = min(header_height, logo.shape[0])
    logo_width = logo.shape[1]

    # Check if logo region is valid before assignment
    if logo_y_offset + logo_height <= header_height and logo_width > 0:
        try:
            # Get the region to place the logo
            logo_region = canvas[logo_y_offset:logo_y_offset+logo_height, 20:20+logo_width]

            # Check if regions have compatible shapes
            if logo_region.shape[0] > 0 and logo_region.shape[1] > 0:
                # Use appropriate part of the logo
                if has_logo and logo.shape[2] == 4:  # With alpha channel
                    alpha_logo = logo[:logo_height, :, 3] / 255.0
                    for c in range(0, 3):
                        logo_region[:, :, c] = (1 - alpha_logo) * logo_region[:, :, c] + alpha_logo * logo[:logo_height, :, c]
                else:  # No alpha channel
                    logo_region[:] = logo[:logo_height, :, :]
        except Exception as e:
            log_debug(f"Logo placement error: {e}")

    # Add title
    title = "EXAM MONITORING - HALL 1"
    cv2.putText(canvas, title, (canvas_width//2 - 400, header_height//2 + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, COLORS["text_normal"], 2)
    # ======== LIVE FEEDBACK WINDOW ========
    # Show latest feedback at top-right corner of video panel
    if 'feedback_text' not in globals():
        feedback_text = "No cheating detected in last 10 minutes"

    cv2.putText(canvas, feedback_text, 
                (video_panel_width + 10, 45),  # adjust position if needed
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    # ======================================

    # Add timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(canvas, current_time, (video_panel_width - 350, header_height//2 + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    # Define video area
    video_area_y = header_height
    video_area_height = canvas_height - header_height

    # Resize frame to fit in the video panel area
    display_frame = cv2.resize(frame, (video_panel_width, video_area_height))


    # Place video frame in the canvas
    canvas[video_area_y:canvas_height, 0:video_panel_width] = display_frame

    # Add live indicator in the video panel area
    create_live_indicator(canvas, pulse_counter, video_panel_width)

    # Calculate scale factors for bounding box display
    scale_x = video_panel_width / original_width
    scale_y = video_area_height / original_height

    # Current detections for this frame
    current_detections = []

    # Flag to check if any cheating behavior is detected in this frame
    cheating_detected_in_frame = False
    highest_cheating_confidence = 0.0

    # Make a copy of the original frame for recording
    frame_with_detections = frame.copy()

    # Run detection with all three models (on original frame for accuracy)
    # Check if models are available before using them
    results1 = model1(frame, conf=0.2) if model1 else []
    results2 = model2(frame, conf=0.2) if model2 else []
    results3 = model3.predict(source=frame) if model3 else []

    # Process Model 1 results (Cheating & Phone) - Process ALL detections
    for result in results1:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get the class name consistently
            cls_name = get_class_name(result, cls_id)
            
            # Debug output for all detections with reasonable confidence
            if conf > 0.3:
                log_debug(f"Model 1 detected: {cls_name} (ID: {cls_id}) with confidence {conf:.2f}")

            # Get original coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Scale coordinates to fit displayed video
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y) + video_area_y
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y) + video_area_y

            # Check for cheating behaviors
            is_cheating = cls_id in CHEATING_CLASS_IDS or cls_name in ["Cheating", "PhoneCall", "Texting"]
            
            # Special check for class 15 which is Cheating
            if cls_id == 15 or cls_name == "Cheating":
                log_debug(f"DIRECT CHEATING DETECTED! ID: {cls_id}, Name: {cls_name}, Conf: {conf:.2f}")
                total_cheating_detections += 1
                is_cheating = True

            # Set color based on detection type
            if is_cheating:
                color = COLORS["Cheating"]  # Red for cheating
                cheating_detected_in_frame = True
                highest_cheating_confidence = max(highest_cheating_confidence, conf)
                last_cheating_label = cls_name
            elif cls_name == "Phone" or cls_id == 12:
                color = COLORS["Phone"]  # Orange for phone
            else:
                if cls_name in COLORS:
                    color = COLORS[cls_name]
                elif cls_id == 12 or cls_name.lower() == "phone":
                    color = COLORS["Phone"]
                elif cls_id in CHEATING_CLASS_IDS or cls_name in ["Cheating", "PhoneCall", "Texting"]:
                    color = COLORS["Cheating"]
                else:
                    continue  # 🚫 skip unknown classes — don't draw anything


            # Set line thickness based on importance
            thickness = 3 if is_cheating else 2

            # Draw bounding box on the display canvas
            # Skip drawing bounding box for 'Classroom' class
            if cls_id == 11 or cls_name.lower() == "classroom":
                continue

            cv2.rectangle(canvas, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), color, thickness)
            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, thickness)

            # Prepare label - mark cheating behaviors clearly
            if is_cheating and cls_name != "Cheating":
                label = f"CHEATING ({cls_name}): {conf:.2f}"
            else:
                label = f"{cls_name}: {conf:.2f}"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background for text on canvas
            cv2.rectangle(canvas, (scaled_x1, scaled_y1 - text_size[1] - 10),
                         (scaled_x1 + text_size[0] + 10, scaled_y1), color, -1)

            # Add text on canvas
            cv2.putText(canvas, label, (scaled_x1 + 5, scaled_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Background for text on original frame
            cv2.rectangle(frame_with_detections, (x1, y1 - text_size[1] - 10),
                         (x1 + text_size[0] + 10, y1), color, -1)

            # Add text on original frame
            cv2.putText(frame_with_detections, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add to detection log with coordinates (X,Y)
            if is_cheating:
                current_detections.append({
                    "text": f"CHEATING ({cls_name}) at ({x1},{y1})" if cls_name != "Cheating" else f"CHEATING at ({x1},{y1})",
                    "color": EVENT_COLORS.get("CHEATING", (247, 99, 99)),
                    "is_alert": True,
                    "timestamp": current_time
                })
            elif cls_name == "Phone" or cls_id == 12:
                current_detections.append({
                    "text": f"Phone at ({x1},{y1})",
                    "color": EVENT_COLORS.get("Phone", (0, 165, 255)),
                    "is_alert": True,
                    "timestamp": current_time
                })
            else:
                current_detections.append({
                    "text": f"{cls_name} at ({x1},{y1})",
                    "color": EVENT_COLORS.get(cls_name, (200, 200, 200)),
                    "is_alert": False,
                    "timestamp": current_time
                })

    # Process Model 2 results (PhoneCall & Texting) - treat as cheating
    for result in results2:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name consistently
            cls_name = get_class_name(result, cls_id)
            
            # Debug output for PhoneCall and Texting
            if cls_id in [13, 14] or cls_name in ["PhoneCall", "Texting"]:
                log_debug(f"Model 2 detected phone activity: {cls_name} (ID: {cls_id}) with confidence {conf:.2f}")

            # Only process PhoneCall and Texting
            if cls_id in [13, 14] or cls_name in ["PhoneCall", "Texting"]:
                # Get original coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale coordinates to fit displayed video
                scaled_x1 = int(x1 * scale_x)
                scaled_y1 = int(y1 * scale_y) + video_area_y
                scaled_x2 = int(x2 * scale_x)
                scaled_y2 = int(y2 * scale_y) + video_area_y

                # Red color for PhoneCall and Texting (cheating behaviors)
                color = COLORS["Cheating"]  # Always use red for cheating
                thickness = 3

                # Draw bounding box on the display frame
                cv2.rectangle(canvas, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), color, thickness)
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, thickness)

                # Override the label to show "CHEATING" for PhoneCall and Texting as requested
                label = f"CHEATING ({cls_name}): {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Background for text on canvas
                cv2.rectangle(canvas, (scaled_x1, scaled_y1 - text_size[1] - 10),
                             (scaled_x1 + text_size[0] + 10, scaled_y1), color, -1)

                # Add text on canvas
                cv2.putText(canvas, label, (scaled_x1 + 5, scaled_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Background for text on original frame
                cv2.rectangle(frame_with_detections, (x1, y1 - text_size[1] - 10),
                             (x1 + text_size[0] + 10, y1), color, -1)

                # Add text on original frame
                cv2.putText(frame_with_detections, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Mark as cheating detected in this frame
                cheating_detected_in_frame = True
                highest_cheating_confidence = max(highest_cheating_confidence, conf)
                last_cheating_label = cls_name
                # Log Phone activity as cheating
                total_cheating_detections += 1

                # Add to detection log with alert
                # Add to detection log with coordinates (X,Y)
                current_detections.append({
                    "text": f"CHEATING ({cls_name}) at ({x1},{y1})",
                    "color": EVENT_COLORS.get(cls_name, (247, 99, 99)),
                    "is_alert": True,
                    "timestamp": current_time
                })


    # Process Model 3 results (Cup, Mug, Bottle)
    for result in results3:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # === OCCUPANCY TRACKING ===
            if cls_id == 3:  # Human class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                human_boxes.append((cx, cy))
                continue  # Skip further processing for humans (no drawing or logging)
          
            # Get class name consistently
            if hasattr(result, 'names'):
                cls_name = result.names.get(cls_id, CLASS_NAMES.get(cls_id, f"Class{cls_id}"))
            else:
                cls_name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")

            # Only process specific objects
            if cls_name in ["cup", "Cup", "bottle", "Bottle", "mug", "Mug"]:
                # Get original coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale coordinates to fit displayed video
                scaled_x1 = int(x1 * scale_x)
                scaled_y1 = int(y1 * scale_y) + video_area_y
                scaled_x2 = int(x2 * scale_x)
                scaled_y2 = int(y2 * scale_y) + video_area_y

                # Check for phone (potential cheating device)
                is_phone = cls_name.lower() == "phone"
                color = COLORS["Phone"] if is_phone else COLORS.get(cls_name, (0, 255, 0))
                thickness = 3 if is_phone else 2

                # Draw bounding box
                cv2.rectangle(canvas, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), color, thickness)
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, thickness)

                # Display label
                label = f"{cls_name}: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Background for text on canvas
                cv2.rectangle(canvas, (scaled_x1, scaled_y1 - text_size[1] - 10),
                             (scaled_x1 + text_size[0] + 10, scaled_y1), color, -1)

                # Add text on canvas
                cv2.putText(canvas, label, (scaled_x1 + 5, scaled_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Background for text on original frame
                cv2.rectangle(frame_with_detections, (x1, y1 - text_size[1] - 10),
                             (x1 + text_size[0] + 10, y1), color, -1)

                # Add text on original frame
                cv2.putText(frame_with_detections, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add to detection log with coordinates (X,Y)
                current_detections.append({
                    "text": f"{cls_name} at ({x1},{y1})",
                    "color": EVENT_COLORS.get(cls_name, color),
                    "is_alert": is_phone,
                    "timestamp": current_time
                })

    occupancy_count = len(human_boxes)
    cv2.putText(canvas, f"Occupancy: {occupancy_count}",
            (20, header_height + 50),
            cv2.FONT_HERSHEY_DUPLEX, 1.3, (139, 0, 0), 2)

    # Add alert banner if cheating detected
    if cheating_detected_in_frame:
        draw_cheating_alert_banner(canvas, highest_cheating_confidence)
        draw_cheating_alert_banner(frame_with_detections, highest_cheating_confidence)
    # Update live feedback message
    if cheating_detected_in_frame:
        feedback_text = f"Cheating detected at {time.strftime('%I:%M:%S')} ({last_cheating_label})"
        last_cheating_frame = frame_count  # 👈 This is critical!
    else:
        if 'last_cheating_frame' not in globals():
            last_cheating_frame = frame_count
        if frame_count - last_cheating_frame > fps * 60:
            feedback_text = "No cheating detected in last 1 minute"

    # Only update detection log at intervals for better visibility
    slow_log_update_counter += 1
    if slow_log_update_counter >= slow_update_interval and current_detections:
        # Reset counter
        slow_log_update_counter = 0

        # Add timestamp to each detection for better tracking
        timestamp_detailed = time.strftime("%H:%M:%S")
        for detection in current_detections:
            detection["timestamp"] = timestamp_detailed
 
        # Update detection log with current detections - prioritize alerts
        alerts = [d for d in current_detections if d.get("is_alert", False)]
        non_alerts = [d for d in current_detections if not d.get("is_alert", False)]

        # Put alerts first — don't trim it here!
        detection_log = alerts + non_alerts + detection_log

        # Optional (only if you want to avoid growing forever, e.g. limit to last 100 entries)
        MAX_DETECTION_HISTORY = 100
        if len(detection_log) > MAX_DETECTION_HISTORY:
            detection_log = detection_log[:MAX_DETECTION_HISTORY]

    # Add vertical separator between video and detection log
    cv2.line(canvas, (video_panel_width, 0), (video_panel_width, canvas_height), (150, 150, 150), 2)

    # Define detection log panel area
    log_area_x = video_panel_width + 1
    log_area_y = header_height
    log_area_width = log_panel_width - 1
    log_area_height = canvas_height - header_height

    # Fill detection log panel with darker background for better contrast
    cv2.rectangle(canvas, (log_area_x, log_area_y),
                 (log_area_x + log_area_width, canvas_height),
                 COLORS["panel_bg"], -1)

    # Add title for detection log
    log_title_height = 40

    # Create a title bar with text in the center
    cv2.rectangle(canvas, (log_area_x, log_area_y),
                 (log_area_x + log_area_width, log_area_y + log_title_height),
                 COLORS["panel_title_bg"], -1)

    # Add centered "DETECTION LOG" title
    FONT = cv2.FONT_HERSHEY_COMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2

    title_width = cv2.getTextSize("DETECTION LOG", cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0][0]
    title_x = log_area_x + (log_area_width - title_width) // 2
    cv2.putText(canvas, "DETECTION LOG",
           (title_x, log_area_y + log_title_height//2 + 10),
           FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, lineType=cv2.LINE_AA)

    # Define table header area
    table_header_y = log_area_y + log_title_height
    table_header_height = 30

    # Draw table header background
    cv2.rectangle(canvas, (log_area_x, table_header_y),
                 (log_area_x + log_area_width, table_header_y + table_header_height),
                 (45, 45, 45), -1)  # Darker gray for header

    # Define column positions for the table
    time_col_x = log_area_x + 20
    event_col_x = log_area_x + 180
    status_col_x = log_area_x + log_area_width - 90

    # Draw column headers with very clear text
    header_text_y = table_header_y + table_header_height - 8

    # Add column headers with bold font
    cv2.putText(canvas, "TIME", (time_col_x, header_text_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "EVENT", (event_col_x, header_text_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "STATUS", (status_col_x, header_text_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Draw separator line below header
    cv2.line(canvas, (log_area_x, table_header_y + table_header_height),
            (log_area_x + log_area_width, table_header_y + table_header_height),
            (100, 100, 100), 1)
            
    # Calculate available height for log entries
    log_entries_start_y = table_header_y + table_header_height + 5
    log_entry_height = 45  # Increased height for better spacing

    # Calculate how many entries will fit
    available_height = log_area_height - (log_title_height + table_header_height + 5)
    max_visible = max(1, available_height // log_entry_height)
    visible_entries = detection_log[:max_visible]  # NEWEST first already

    for i, entry in enumerate(visible_entries):
        entry_y = log_entries_start_y + (i * log_entry_height) + 20
        entry = detection_log[i]
        entry_y = log_entries_start_y + (i * log_entry_height) + 20

        # Skip if we're out of bounds
        if entry_y >= canvas_height - 5:
            break

        # Alternating row backgrounds with better contrast
        if i % 2 == 0:
            cv2.rectangle(canvas, (log_area_x, entry_y - 22),
                         (log_area_x + log_area_width, entry_y + 13),
                         (40, 40, 40), -1)  # Slightly lighter
        else:
            cv2.rectangle(canvas, (log_area_x, entry_y - 22),
                         (log_area_x + log_area_width, entry_y + 13),
                         (20, 20, 20), -1)  # Darker

        # Get entry data
        text = entry["text"]
        color = entry["color"]
        is_alert = entry["is_alert"]
        timestamp = entry.get("timestamp", "")

        # Format time as HH:MM:SS for TIME column
        time_only = timestamp

        # Format the timestamp exactly like in your screenshot: HH:MM:SS
        if ":" in timestamp:
            # Already in HH:MM:SS format
            time_only = timestamp
        elif " " in timestamp:
            # Contains date and time, extract only time
            time_parts = timestamp.split(" ")
            if len(time_parts) > 1:
                time_only = time_parts[1].split(".")[0]  # Remove seconds fraction if present

        # Display TIME column
        cv2.putText(canvas, time_only, (time_col_x, entry_y),
            cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # Display EVENT column
        cv2.putText(canvas, text, (event_col_x, entry_y),
            cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)

        # Display STATUS column
        if is_alert:
            label = "ALERT"
            (label_width, label_height), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

            # Add dynamic padding
            padding_x = 10
            padding_y = 6

            # Draw dynamic background box
            cv2.rectangle(
                canvas,
                (status_col_x - padding_x, entry_y - label_height - padding_y),
                (status_col_x + label_width + padding_x, entry_y + padding_y // 2),
                (0, 0, 255),
                -1
            )
            # Add the ALERT text
            cv2.putText(
                canvas,
                label,
                (status_col_x, entry_y),
                FONT,
                FONT_SCALE,
                (255, 255, 255),
                FONT_THICKNESS,
                lineType=cv2.LINE_AA
            )

        else:
            # Normal status
            cv2.putText(canvas, "Normal", (status_col_x, entry_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Show the complete canvas
    cv2.imshow(window_name, canvas)

    # Write to main output recording
    main_out.write(frame_with_detections)

    # Handle recording of cheating incidents
    if cheating_detected_in_frame:
        # If it's a new cheating detection (not already being recorded)
        if not is_cheating_detected:
            is_cheating_detected = True
            cheating_record_start_time = time.time()

            # Generate a filename with timestamp for this cheating incident
            cheating_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            cheating_output_filename = f"cheating_incident_{cheating_timestamp}_conf{int(highest_cheating_confidence*100)}.mp4"
            cheating_output_path = os.path.join(OUTPUT_DIR, cheating_output_filename)

            # Create a new video writer for this cheating incident
            cheating_out = cv2.VideoWriter(cheating_output_path, fourcc, fps, (canvas_width, canvas_height))
            log_debug(f"⚠️ Cheating detected! Recording started: {cheating_output_filename}")

        # If we're already recording, continue adding frames
        if cheating_out is not None:
            # Write the current canvas (with detection boxes) to the output
            cheating_out.write(canvas)

    # If we're recording but no longer detecting cheating, check if we should stop
    elif is_cheating_detected:
        # Check if we've reached the recording duration limit
        if time.time() - cheating_record_start_time >= cheating_recording_duration:
            # Stop recording
            if cheating_out is not None:
                cheating_out.release()
                cheating_out = None
                log_debug(f"✅ Cheating incident recording completed")
            is_cheating_detected = False
        # Continue recording for the duration even if cheating is no longer detected
        elif cheating_out is not None:
            cheating_out.write(canvas)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======== SESSION SUMMARY FUNCTION ========
def generate_session_summary(detection_log):
    if not detection_log:
        return "No cheating detected during this session."

    total = len([entry for entry in detection_log if entry['is_alert']])
    phones = len([e for e in detection_log if "Phone" in e['text']])
    direct_cheating = len([e for e in detection_log if "Cheating" in e['text']])

    first_time = detection_log[-1].get("timestamp", "N/A")
    last_time = detection_log[0].get("timestamp", "N/A")
    session_date = datetime.now().strftime("%Y-%m-%d")

    summary = (
        f"ALERT Type: Cheating\n"
        f"Date: {session_date}\n"
        f"Start Time: {first_time}\n"
        f"End Time: {last_time}\n"
    )
    return summary

class EmailSender(threading.Thread):
    def __init__(self, subject, body, recipients, attachments, sender_email, sender_password):
        threading.Thread.__init__(self)
        self.subject = subject
        self.body = body
        self.recipients = recipients
        self.attachments = attachments
        self.sender_email = sender_email
        self.sender_password = sender_password

    def run(self):
        msg = EmailMessage()
        msg['Subject'] = self.subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.recipients)
        msg.set_content(self.body)

        for filepath in self.attachments:
            try:
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(filepath)
                    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
            except Exception as e:
                print(f"❌ Error attaching file: {filepath}: {e}")

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.starttls()
                smtp.login(self.sender_email, self.sender_password)
                smtp.send_message(msg)
                print("📨 Email sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

# ==========================================

# Cleanup
cap.release()
if cheating_out is not None:
    cheating_out.release()
main_out.release()
cv2.destroyAllWindows()

# ======== DISPLAY SESSION SUMMARY ========
print("\n📊 SESSION SUMMARY 📊")
print(generate_session_summary(detection_log))

# Save detection log to Excel
log_df = pd.DataFrame(detection_log)
excel_path = os.path.join(session_folder, f"detection_log_{session_time}.xlsx")
log_df.to_excel(excel_path, index=False)
print(f"📁 Log saved to Excel: {excel_path}")

# Create subfolder for cheating recordings
cheating_folder_name = f"CHEATING_DETECTIONS_{session_time}"
cheating_folder_path = os.path.join(session_folder, cheating_folder_name)
os.makedirs(cheating_folder_path, exist_ok=True)

# Move all cheating clips to that folder
cheating_files = []
for file in os.listdir(OUTPUT_DIR):
    if file.startswith("cheating_incident") and file.endswith(".mp4"):
        src_path = os.path.join(OUTPUT_DIR, file)
        dest_path = os.path.join(cheating_folder_path, file)
        os.rename(src_path, dest_path)
        cheating_files.append(file)
        print(f"📁 Moved to cheating folder: {file}")

# Zip the cheating folder
zip_path = os.path.join(session_folder, f"{cheating_folder_name}.zip")
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for file in cheating_files:
        zipf.write(os.path.join(cheating_folder_path, file), arcname=file)
        print(f"🗜️ Zipped: {file}")



# Email setup
EMAIL_SENDER = "yasrsdp2@gmail.com"
EMAIL_PASSWORD = "wbryxopiisutdllb"
EMAIL_RECIPIENTS = ["ruwyas01@gmail.com"]

email_subject = f"Exam Session Report - {session_date} {session_time}"
email_body = generate_session_summary(detection_log)
MAX_ATTACHMENT_SIZE = 25 * 1024 * 1024  # 25 MB

attachments = [excel_path]
zip_size = os.path.getsize(zip_path)

if zip_size <= MAX_ATTACHMENT_SIZE:
    attachments.append(zip_path)
else:
    print(f"⚠️ ZIP file too large to attach ({zip_size / (1024 * 1024):.2f} MB), skipping it.")
    email_body += "\n\n⚠️ Note: The video evidence ZIP file was too large to attach and has been omitted."


email_thread = EmailSender(email_subject, email_body, EMAIL_RECIPIENTS, attachments, EMAIL_SENDER, EMAIL_PASSWORD)
email_thread.start()

# =========================================
print(f"✅ Processing completed. Full recording saved to {output_path}")
print(f"Total cheating detections: {total_cheating_detections}")
