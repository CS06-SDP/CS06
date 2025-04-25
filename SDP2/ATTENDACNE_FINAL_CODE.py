
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

# Model and Video paths
MODEL_PATH = r"D:\SDP2 NEW\occupancy\yolov8s.pt"
#VIDEO_PATH = r"C:\Users\Zayed\Downloads\CAMERA_VIDEOS_UNEDITED\CAMERA_VIDEOS_UNEDITED\MVI_1364.MOV"
#VIDEO_PATH= r"C:\Users\Zayed\Downloads\1minute.mp4"
#VIDEO_PATH= r"D:\SDP2 NEW\untouched vids\untouched vids\Xray0017 C1-00-131131-131141.mp4"
#VIDEO_PATH =r"D:\SDP2 NEW\untouched vids\untouched vids\Xray0017 C3-00-131811-131817.mp4"
VIDEO_PATH=r"D:\SDP2 NEW\untouched vids\untouched vids\Xray0017 C1-00-140259-140308.mp4"

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Colors and layout
HEADER_BG = (50, 30, 20)  # Navy blue  # Dark blue
TEXT_COLOR = (255, 255, 255)  # White
LIVE_COLOR = (0, 0, 255)

# Logo utilities
def create_ku_logo():
    logo = np.zeros((60, 250, 3), dtype=np.uint8)
    for i in range(logo.shape[1]):
        blue_gradient = int(20 + (i / logo.shape[1]) * (80 - 20))
        logo[:, i] = [blue_gradient, int(blue_gradient / 2), 0]
    cv2.rectangle(logo, (0, 0), (logo.shape[1]-1, logo.shape[0]-1), (120, 80, 0), 2)
    cv2.putText(logo, "KHALIFA", (15, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(logo, "UNIVERSITY", (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
    return logo

def load_university_logo():
    possible_paths = [r"D:\SDP2 NEW\university_cheating_sys_rodha\logo.png"]
    for path in possible_paths:
        try:
            logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if logo is not None:
                logo_height = 55
                aspect_ratio = logo.shape[1] / logo.shape[0]
                logo_width = int(logo_height * aspect_ratio)
                logo = cv2.resize(logo, (logo_width, logo_height))
                return logo, True
        except Exception as e:
            print(f"Error loading logo: {e}")
    return create_ku_logo(), True


def place_logo_in_corner(canvas, logo, header_height):
    logo_y_offset = max(0, (header_height - logo.shape[0]) // 2)
    logo_height = min(header_height, logo.shape[0])
    logo_width = logo.shape[1]

    # Prepare the region of interest on the canvas
    roi = canvas[logo_y_offset:logo_y_offset + logo_height, 10:10 + logo_width]

    # Check if logo has an alpha channel
    if logo.shape[2] == 4:
        alpha_channel = logo[:, :, 3] / 255.0
        for c in range(3):  # BGR
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + logo[:, :, c] * alpha_channel
    else:
        roi[:] = logo[:logo_height]

    return canvas


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_PATH}")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Automatically downscale large videos to fit a max canvas width (e.g., 1280px)
MAX_CANVAS_WIDTH = 1280
scale_factor = 1.0
if original_frame_width > MAX_CANVAS_WIDTH:
    scale_factor = MAX_CANVAS_WIDTH / original_frame_width
    frame_width = int(original_frame_width * scale_factor)
    frame_height = int(original_frame_height * scale_factor)
else:
    frame_width = original_frame_width
    frame_height = original_frame_height

# Define layout after scaling
video_panel_width = int(frame_width * 0.7)
log_panel_width = frame_width - video_panel_width
header_height = int(frame_height * 0.12)
canvas_width = video_panel_width + log_panel_width
canvas_height = frame_height + header_height




header_height = int(frame_height * 0.12)  # Responsive header height
canvas_width = frame_width
canvas_height = frame_height + header_height

frame_count = 0
pulse_counter = 0
max_occupancy = 0

def draw_live_indicator(canvas, pulse_counter):
    alpha = abs(np.sin(pulse_counter * 0.1)) * 0.5 + 0.5
    x, y = canvas_width - 150, 40
    cv2.circle(canvas, (x, y), 10, (0, 0, int(255 * alpha)), -1)
    
    cv2.putText(canvas, "LIVE", (x + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_COLOR, 2)

print("🚀 Starting Occupancy Monitoring...")

logo, has_logo = load_university_logo()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    pulse_counter += 1

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    person_count = sum(1 for det in detections if model.names[int(det[5])] == 'person')
    max_occupancy = max(max_occupancy, person_count)
    timestamp = time.strftime("%H:%M:%S")

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (50, 30, 20)

    cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), HEADER_BG, -1)
    text = "CLASS OCCUPANCY MONITORING"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = frame_width / 1280
    font_scale = max(0.6, min(1.4, font_scale))  # Scaled for high-res video
    thickness = 2
    (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (canvas_width - text_width) // 2
    cv2.putText(canvas, text, (text_x, 40), font, font_scale,(255,255,255) , thickness)
    cv2.putText(canvas, timestamp, (canvas_width - 300, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_COLOR, 2)

    if has_logo:
        canvas = place_logo_in_corner(canvas, logo, header_height)

    video_frame = cv2.resize(frame, (canvas_width, frame_height))
    canvas[header_height:header_height + frame_height, 0:canvas_width] = video_frame

    for det in detections:
        if model.names[int(det[5])] == 'person':
            x1, y1, x2, y2 = det[:4]
            x1 = int(x1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y1 = int(y1 * scale_factor)
            y2 = int(y2 * scale_factor)

            cv2.rectangle(canvas, (int(x1), int(y1) + header_height),
                          (int(x2), int(y2) + header_height),
                          (0, 255, 0), 2)

    cv2.putText(canvas, f"Current Occupancy: {person_count}", (30, header_height + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    draw_live_indicator(canvas, pulse_counter)

    cv2.imshow("CLASS MONITORING", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

slip = np.ones((400, 600, 3), dtype=np.uint8) * 255
cv2.putText(slip, "KHALIFA UNIVERSITY", (50, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 2)
cv2.putText(slip, "Class Attendance Report", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.line(slip, (50, 160), (550, 160), (0, 0, 0), 2)

cv2.putText(slip, f"Session Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
cv2.putText(slip, f"Maximum Occupancy: {max_occupancy}", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 102, 0), 2)
cv2.putText(slip, "Note: This reflects the highest number of people detected", (50, 310), cv2.FONT_HERSHEY_PLAIN, 1, (80, 80, 80), 1)

cv2.imshow("Attendance Slip", slip)
cv2.waitKey(0)
cv2.destroyAllWindows()
