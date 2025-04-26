import cv2
import os
import tkinter as tk
from tkinter import Label, Text
from PIL import Image, ImageTk
from ultralytics import YOLO
 
 
# Load YOLO model
model = YOLO(r"D:\FINALGCSDP\runs\detect\train15\weights\best.pt")  # Adjust path
model.conf = 0.1  # Confidence threshold
 
 
# Directory containing videos
video_directory = r"D:\Dataset Final\Testing videos"
video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi'))]
 
 
# Alert directory
alert_dir =  r"D:\FINALGCSDP\Alerts"
os.makedirs(alert_dir, exist_ok=True)
 
 
# Initialize Tkinter interface
window = tk.Tk()
window.title("Classroom Occupancy Monitoring & Behavior Analysis System")
window.geometry("900x700")
window.configure(bg="lightgrey")
 
 
# Title Label
title_label = Label(
   window,
   text="Classroom Occupancy Monitoring & Behavior Analysis System",
   font=("Helvetica", 16, "bold"),
   bg="lightgrey",
   fg="darkblue"
)
title_label.pack(pady=10)
 
 
# Video panel
video_label = Label(window, bg="black")
video_label.pack(pady=10)
 
 
# Text area for alerts
alert_text = Text(window, height=15, width=100, font=("Courier", 10), wrap="word", bg="white", fg="black")
alert_text.pack(pady=10)
 
 
# Function to process a single video
def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    alert_text.insert(tk.END, f"Processing video: {video_path}\n")
    window.update()
 
    # Create subdirectories for each video
    runs_dir = os.path.join("runs", "detect", video_name)
    os.makedirs(runs_dir, exist_ok=True)
    alerts_dir = os.path.join(alert_dir, video_name)
    os.makedirs(alerts_dir, exist_ok=True)
 
    # Initialize CSV file
    csv_path = os.path.join(runs_dir, f"{video_name}_detections.csv")
    with open(csv_path, 'w') as csv_file:
        csv_file.write("frame,class_name,confidence,x1,y1,x2,y2\n")
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        alert_text.insert(tk.END, f"Error: Could not open video {video_path}\n")
        window.update()
        return
 
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            alert_text.insert(tk.END, f"Finished processing video: {video_name}\n")
            window.update()
            next_video()
            return
 
        # Resize frame for display
        frame = cv2.resize(frame, (800, 450))
 
        # Perform object detection
        results = model.predict(
            source=frame,
            save=True,
            project=runs_dir,
            name="predict",
            exist_ok=True
        )
 
        alerts = []
        human_boxes = []
        sensitive_area_boxes = []
 
        # Save detections to CSV and process results
        with open(csv_path, 'a') as csv_file:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
 
                # Write to CSV
                csv_file.write(f"{frame_number},{label},{confidence:.2f},{x1},{y1},{x2},{y2}\n")
 
                # Handle MainFrame (restricted area)
                if label == "MainFrame":
                    # Increase size of MainFrame bounding box by 50%
                    width = x2 - x1
                    height = y2 - y1
                    new_width = int(width * 1.5)
                    new_height = int(height * 1.5)
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
 
                    x1 = max(0, center_x - new_width // 2)
                    y1 = max(0, center_y - new_height // 2)
                    x2 = min(frame.shape[1], center_x + new_width // 2)
                    y2 = min(frame.shape[0], center_y + new_height // 2)
                    sensitive_area_boxes.append((x1, y1, x2, y2))
                    continue
 
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
                # Handle Humans
                if label == "Human":
                    human_boxes.append((x1, y1, x2, y2))
                alerts.append(f"{label} detected at ({x1}, {y1})")
 
        # Check for unauthorized access
        alert_triggered = False
        for human_box in human_boxes:
            for sensitive_area_box in sensitive_area_boxes:
                if calculate_iou(human_box, sensitive_area_box) > 0:  # Any overlap triggers alert
                    alert_triggered = True
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    alert_path = os.path.join(alerts_dir, f"alert_frame_{frame_number}.jpg")
                    cv2.imwrite(alert_path, frame)
 
                    # Draw red bounding box and alert text
                    cv2.rectangle(frame, (human_box[0], human_box[1]), (human_box[2], human_box[3]), (0, 0, 255), 2)
                    cv2.putText(frame, "Human", (human_box[0], human_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    alerts.append(f"ALERT: Unauthorized access in frame {frame_number}.")
 
        if alert_triggered:
            alert_text.insert(tk.END, "ALERT: Unauthorized Personnel Detected!\n")
            window.update()
            # Display alert text on the frame in red
            cv2.putText(frame, "ALERT: Unauthorized Personnel Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
        # Update alert box in the interface
        alert_text.delete(1.0, tk.END)
        for alert in alerts:
            alert_text.insert(tk.END, alert + "\n")
 
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
 
        # Schedule next frame update
        window.after(10, update_frame)
 
    update_frame()
 
# Global video index
video_index = 0
 
# Function to start the next video
def next_video():
    global video_index
    if video_index < len(video_files):
        process_video(video_files[video_index])
        video_index += 1
    else:
        alert_text.insert(tk.END, "All videos processed.\n")
        window.update()
 
# Function to calculate IoU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)
 
# Start the first video
next_video()
 
# Run the Tkinter event loop
window.mainloop()
