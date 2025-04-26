import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tkinter as tk
from tkinter import Label, Text, Frame, scrolledtext, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import time
from collections import deque, Counter
import math
import traceback
from statistics import median  
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import threading
import queue
import pandas as pd 

# Configuration Constants 

MODEL1_PATH = r"D:\SDP2 NEW\trial 2_sdp2_only\runs\detect\train\weights\best.pt"  # Behavior analysis model
MODEL2_PATH = r"D:\CS06 SDP I\runs\detect\train15\weights\best.pt"    # Object detection model
MODEL3_PATH=r"D:\SDP2 NEW\trial 3_sdp2_with_EEDPT\runs\detect\train\weights\best.pt"
#VIDEO_PATH= r"D:\CS06 SDP I\Testing videos\Video43.mp4"
#VIDEO_PATH = r"D:\SDP2 NEW\cheating_videos_rodha\Xray0017 C3-00-133352-133359.mp4"
#VIDEO_PATH = r"D:\SDP2 NEW\cheating_videos_rodha\Xray0017 C1-00-122122-122132.mp4"
#VIDEO_PATH = r"D:\CS06 SDP I\Testing videos\Xray0017 C3-00-014231-014241.mp4"
#VIDEO_PATH= r"C:\Users\Zayed\Downloads\04172025 2\04172025\Xray0017 C3-00-132422-132432.mp4"
VIDEO_PATH= r"D:\SDP2 NEW\LAB_NEW_TESTING_VIDEOS\04172025\Xray0017 C1-00-130730-130741.mp4"


ALERT_DIR = r"D:\SDP2 NEW\INTERFACES_ALERTS\Interface_Behavior"

EMAIL_ENABLED = True  # Set to False to disable email alerts
EMAIL_SENDER = "yasrsdp2@gmail.com"  # Your email address DONT CHANGE THIS 
EMAIL_PASSWORD = "wbryxopiisutdllb"  # Your email app password (for Gmail, use App Password) DONT CHANGE
EMAIL_RECIPIENTS = ["ruwyas01@gmail.com"]  # HERE YOU CAN CHANGE put your own email or List of recipients
EMAIL_SMTP_SERVER = "smtp.gmail.com"  # SMTP server
EMAIL_SMTP_PORT = 587  # SMTP port
EMAIL_COOLDOWN = 60  # Minimum seconds between emails for the same type of alert

# Detection Parameters - Enhanced confidence thresholds for more reliable detections
CLASS_MODEL_ASSIGNMENT = {
    "MainFrame": (2, 0.35),  # Reduced threshold for better detection
    "Classroom": (1, 0.35),  # Reduced threshold for better detection
    "Human": (2, 0.4),       # Balanced for reliable detection without missing people
    "Bottle": (2, 0.3),
    "Mug": (2, 0.3),
    "Cup": (2, 0.3),
    "Cheating": (1, 0.4),
    "Eating": (3, 0.4),
    "Drinking": (3, 0.4),
    "Phone": (1, 0.45),
    "PhoneCall": (3, 0.45),
    "Texting": (3, 0.45),
    "SafetyJacket": (1, 0.4),
    "Helmet": (1, 0.4)
}

# Make all classes visible by default
VISIBLE_CLASSES = {
    "MainFrame": True,      # make it false if you dont wanna see mainframe 
    "Classroom": False,      
    "Human": True,
    "Bottle": True,
    "Mug": True,
    "Cup": True,
    "Cheating": True,
    "Eating": True,
    "Drinking": True,
    "Phone": True,
    "PhoneCall": True,
    "Texting": True,
    "SafetyJacket": True,
    "Helmet": True
}

# Enhanced color scheme for better visibility DONT CHANGE THIS PLEASEE :) 
COLOR_MAP = {
    "Human": (0, 0, 255),       # Red
    "Cup": (0, 255, 0),         # Green
    "Bottle": (255, 0, 0),      # Blue
    "Mug": (0, 255, 255),       # Yellow
    "MainFrame": (255, 0, 255), # Magenta
    "Classroom": (200, 200, 200), # Light Gray for better visibility
    "SafetyJacket": (34, 139, 34), # Forest Green
    "Helmet": (0, 0, 128),      # Dark Blue
    "Cheating": (0, 69, 255),   # Orange
    "Eating": (180, 105, 255),  # Pink
    "Drinking": (128, 255, 0),  # Light Green
    "Phone": (0, 165, 255),     # Orange
    "PhoneCall": (226, 43, 138), # Purple
    "Texting": (255, 191, 0)    # Cyan
}

# Increased thickness for better visibility
BBOX_THICKNESS = {
    "Human": 2,
    "SafetyJacket": 2,
    "Helmet": 2,
    "MainFrame": 3,
    "Classroom": 2,
    "default": 2
}
EMAIL_ALERT_CLASSES = {
    "Cheating": True,
    "Phone": False, #turn these 3 to true in exam monitor mode :) 
    "PhoneCall": False,
    "Texting":False ,
    "Unauthorized": True,  # For unauthorized mainframe access
}
# Improved tracking parameters
MAX_AGE = 20               # Frames before deleting a tracker that's missing
MIN_HITS = 5               # Frames before considering a tracker stable
IOU_THRESHOLD = 0.3        # Increased for better matching
MIN_BOX_AREA = 800         # Reduced to avoid missing smaller humans
EQUIPMENT_DISTANCE_THRESHOLD = 150
CLASSROOM_ENTRY_FRAMES = 8 # Reduced for faster response
CLASSROOM_EXIT_FRAMES = 8  # Reduced for faster response
USE_CLASSROOM_LOGIC=False #Change to true if we want the classroom occupancy logic 


class HumanTracker:
    """Tracks a human and their associated equipment/status"""
    def __init__(self, id, initial_box):
        self.id = id
        self.box = initial_box
        self.hits = 1
        self.missed_frames = 0
        self.history = deque(maxlen=30)
        self.history.append(initial_box)
        self.in_classroom = False
        self.authorized = False
        self.near_mainframe = False
        self.uncertain = False
        
        # Equipment tracking with improved counter approach
        self.safety_equipment = {"SafetyJacket": False, "Helmet": False}
        self.equipment_count = {"SafetyJacket": 0, "Helmet": 0}
        self.classroom_transition_time = 0
        
        # Motion prediction
        self.prev_center = self.get_center()
        self.velocity = [0, 0]
        self.aspect_ratio = self.calculate_aspect_ratio(initial_box)
        self.size = self.calculate_size(initial_box)
        
        # For smoothing
        self.smoothed_box = initial_box
        self.box_history = deque(maxlen=5)
        self.box_history.append(initial_box)
        self.classroom_history=deque(maxlen=5)
        self.missed_classroom_frames=0
        
    def calculate_aspect_ratio(self, box):
        """Calculate width/height ratio of bounding box"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width / height if height > 0 else 1.0
        
    def calculate_size(self, box):
        """Calculate area of bounding box"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height
        
    def update(self, box):
        """Update tracker with new detection"""
        # Store current box in history
        self.box = box
        self.box_history.append(box)
        self.history.append(box)
        self.hits += 1
        self.missed_frames = 0
        
        # Apply smoothing to reduce jitter
        if len(self.box_history) >= 3:
            # Simple moving average for smoother tracking
            x1 = sum(b[0] for b in self.box_history) // len(self.box_history)
            y1 = sum(b[1] for b in self.box_history) // len(self.box_history)
            x2 = sum(b[2] for b in self.box_history) // len(self.box_history)
            y2 = sum(b[3] for b in self.box_history) // len(self.box_history)
            self.smoothed_box = (x1, y1, x2, y2)
        else:
            self.smoothed_box = box
        
        # Update appearance model
        self.aspect_ratio = self.calculate_aspect_ratio(box)
        self.size = self.calculate_size(box)
        
        # Update motion prediction
        current_center = self.get_center()
        self.velocity = [
            current_center[0] - self.prev_center[0],
            current_center[1] - self.prev_center[1]
        ]
        self.prev_center = current_center
        
        return self.smoothed_box
        
    def predict(self):
        """Predict next position using simple motion model"""
        self.missed_frames += 1
        
        if len(self.history) == 0:
            return self.box
            
        # Use last known box as base
        last_box = self.history[-1]
        width = last_box[2] - last_box[0]
        height = last_box[3] - last_box[1]
        
        # Predict new center using velocity with decay
        decay = min(1.0, self.missed_frames / 10.0)  # Slow down prediction over time
        center_x = self.prev_center[0] + self.velocity[0] * (1 - decay)
        center_y = self.prev_center[1] + self.velocity[1] * (1 - decay)
        
        # Create new box
        new_box = (
            int(center_x - width/2),
            int(center_y - height/2),
            int(center_x + width/2),
            int(center_y + height/2)
        )
        
        self.box = new_box
        self.prev_center = (center_x, center_y)
        
        return new_box
        
    def get_center(self, box=None):
        """Get center coordinates of a box"""
        if box is None:
            box = self.smoothed_box if hasattr(self, 'smoothed_box') and self.smoothed_box else self.box
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        
    def update_safety_equipment(self, equipment_detections):
        """Update safety equipment status with improved association"""
        equipment_detected = {"SafetyJacket": False, "Helmet": False}
        
        # Check each equipment type
        for eq_type, detections in equipment_detections.items():
            for det, conf in detections:
                # Check distance and overlap
                person_center = self.get_center()
                det_center = self.get_center(det)
                dist = math.sqrt((person_center[0] - det_center[0])**2 + 
                               (person_center[1] - det_center[1])**2)
                iou = self.calculate_iou(self.box, det)
                
                # If equipment is associated with this person
                if iou > 0.05 or dist < EQUIPMENT_DISTANCE_THRESHOLD:
                    equipment_detected[eq_type] = True
                    # Higher quality detections increase counter more
                    increment = 3 if conf > 0.6 else 2
                    self.equipment_count[eq_type] = min(30, self.equipment_count[eq_type] + increment)
                    break
        
        # Update counters for equipment not detected
        for eq_type in ["SafetyJacket", "Helmet"]:
            if not equipment_detected[eq_type]:
                # When not detected, slowly decrease counter
                self.equipment_count[eq_type] = max(0, self.equipment_count[eq_type] - 1)
                
            # Equipment is present if counter is above threshold - lower threshold for higher sensitivity
            self.safety_equipment[eq_type] = self.equipment_count[eq_type] >= 8
        
        # BOTH are required for authorization
        self.authorized = self.safety_equipment["SafetyJacket"] and self.safety_equipment["Helmet"]
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def is_near_mainframe(self, mainframe_box):
        """Check if person is near mainframe with improved logic"""
        if not mainframe_box or not self.box:
            return False
            
        # Check distance to mainframe
        person_center = self.get_center()
        mainframe_center = ((mainframe_box[0] + mainframe_box[2]) // 2, 
                          (mainframe_box[1] + mainframe_box[3]) // 2)
        
        # Calculate distance
        distance = math.sqrt((person_center[0] - mainframe_center[0])**2 + 
                           (person_center[1] - mainframe_center[1])**2)
                           
        # Expanded area around mainframe - increased for better detection
        x1, y1, x2, y2 = mainframe_box
        width = x2 - x1
        height = y2 - y1
        expanded = (x1-width//2, y1-height//2, x2+width//2, y2+height//2)
        
        # Check overlap with expanded area
        overlap = not (self.box[2] < expanded[0] or self.box[0] > expanded[2] or 
                     self.box[3] < expanded[1] or self.box[1] > expanded[3])
        
        # Also check direct IoU with mainframe - improved for better detection
        iou = self.calculate_iou(self.box, mainframe_box)
        
        # Return true if any condition is met
        return distance < 200 or overlap or iou > 0.02

    def appearance_similarity(self, box):
        """Calculate appearance similarity score between current tracker and a new detection"""
        new_aspect_ratio = max(0.1, (box[2] - box[0]) / max(1, box[3] - box[1]))
        new_size = (box[2] - box[0]) * (box[3] - box[1])
        
        ar_diff = abs(self.aspect_ratio - new_aspect_ratio) / max(self.aspect_ratio, new_aspect_ratio)
        size_diff = abs(self.size - new_size) / max(self.size, new_size)
        
        # Lower score is better (0 = perfect match)
        return 0.5 * ar_diff + 0.5 * size_diff

class EmailSender:
    """Handles email notifications in a background thread to avoid blocking the main application"""
    def __init__(self):
        self.email_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.last_sent = {}  # Track when we last sent each alert type
        self.running = True
        self.thread.start()
        
    def _process_queue(self):
        """Process email queue in background"""
        while self.running:
            try:
                # Wait for a message with a short timeout to allow for clean shutdown
                try:
                    email_data = self.email_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process the email
                try:
                    self._send_email(email_data)
                except Exception as e:
                    print(f"Error sending email: {str(e)}")
                finally:
                    self.email_queue.task_done()
            except Exception as e:
                print(f"Error in email thread: {str(e)}")
                
    def _send_email(self, email_data):
        """Send an email with screenshot attachment"""
        alert_type = email_data.get('alert_type', 'Unknown')
        screenshot = email_data.get('screenshot')
        details = email_data.get('details', {})
        
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = ', '.join(EMAIL_RECIPIENTS)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg['Subject'] = f"ALERT: {alert_type} detected at {timestamp}"
        
        # Create email body
        body = f"Alert Type: {alert_type}\n"
        body += f"Time: {timestamp}\n"
        
        # Add details to the body
        for key, value in details.items():
            body += f"{key}: {value}\n"
            
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach the screenshot if available
        if screenshot is not None:
            # Convert NumPy array to bytes
            _, img_encoded = cv2.imencode('.jpg', screenshot)
            img_bytes = img_encoded.tobytes()
            
            # Create image attachment
            image = MIMEImage(img_bytes)
            image.add_header('Content-Disposition', 'attachment', filename=f"{alert_type}_{timestamp}.jpg")
            msg.attach(image)
        
        # Connect to server and send email
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        # Update last sent time for this alert type
        self.last_sent[alert_type] = time.time()
        
    def queue_email(self, alert_type, screenshot, details=None):
        """Queue an email to be sent"""
        if not EMAIL_ENABLED:
            return False
            
        # Check cooldown period to avoid email flooding
        current_time = time.time()
        last_sent_time = self.last_sent.get(alert_type, 0)
        if current_time - last_sent_time < EMAIL_COOLDOWN:
            return False  # Still in cooldown period
            
        # Queue the email
        self.last_sent[alert_type]=current_time
        self.email_queue.put({
            'alert_type': alert_type,
            'screenshot': screenshot.copy() if screenshot is not None else None,
            'details': details or {}
        })
        return True
        
    def shutdown(self):
        """Shutdown the email sender thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

class ClassroomMonitor:
    """Main application class for classroom monitoring"""
    def __init__(self):
        # Video playback control variables
        self.video_paused = False
        self.video_ended = False
        self.current_frame = None
        self.current_image = None
        
        # Initialize UI first
        self.setup_ui()
        
        # Then initialize the detection system
        self.initialize_system()
        self.detections_log=[]
        
    def setup_ui(self):
        """Initialize the user interface with enhanced controls"""
        self.window = tk.Tk()
        self.window.title("CS06 Monitoring System")
        self.window.geometry("900x700")
        self.window.minsize(900, 700)
        self.window.configure(bg="#d3d3d3")  # Dark blue background
        
        # Configure grid weights for resizing
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Title with better styling
        title_frame = Frame(self.window, bg="#d3d3d3", pady=10)
        title_frame.pack(fill="x")
        # Load and display logo at top-left
        logo_path = r"D:\SDP2 NEW\university_cheating_sys_rodha\logo.png"
        try:
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((185, 65))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = Label(title_frame, image=self.logo_photo, bg="#d3d3d3")
            logo_label.pack(side="left", padx=(10, 20))
        except Exception as e:
            print(f"Failed to load logo: {e}")

        
        Label(title_frame, text="Occupancy Monitoring & Behavior Analysis",
              font=("Segoe UI", 35, "bold"), bg="#d3d3d3", fg="#1e1e1e").pack()
        
        # Info Panel with better contrast
        info_frame = Frame(self.window, bg="#d3d3d3", padx=10, pady=5)  # Darker blue
        info_frame.pack(fill="x")
        
        # LIVE indicator with pulsing effect
        self.realtime_indicator = Label(info_frame, text="● LIVE", 
                                      font=("Helvetica", 14, "bold"), bg="#d3d3d3", fg="#e74c3c")
        self.realtime_indicator.pack(side="left", padx=10)
        self._blink_state = True
        
        # File info label
        self.file_info_label = Label(info_frame, text="No video loaded", 
                                   font=("Helvetica", 12), bg="#d3d3d3", fg="#1e1e1e")
        self.file_info_label.pack(side="left", padx=20)
        
        # Occupancy count with better visibility
        self.occupancy_label = Label(info_frame, text="Occupancy Count: 0", 
                                   font=("Helvetica", 16, "bold"), bg="#d3d3d3", fg="darkblue")
        self.occupancy_label.pack(side="left", padx=20)
        
        # Alert label
        self.alert_label = Label(info_frame, text="", font=("Helvetica", 16, "bold"), 
                               bg="#d3d3d3", fg="#e74c3c")
        self.alert_label.pack(side="right", padx=10)
        
        # Video Display - darker border for contrast
        self.video_frame = Frame(self.window, bg="#d3d3d3", bd=2, relief="solid")
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.window.update_idletasks() 
        self.display_width  = self.video_frame.winfo_width()
        self.display_height = self.video_frame.winfo_height()
        self.video_label = Label(self.video_frame, bg="#000000")
        self.video_label.pack(fill="both", expand=True)
        
        # Control Panel with better styling
        control_frame = Frame(self.window, bg="#d3d3d3", padx=10, pady=5)
        control_frame.pack(fill="x")
        self.alert_label = Label(info_frame, text="", font=("Helvetica", 16, "bold"), bg="#d3d3d3", fg="white")
        # Improved button styling
        button_style = {"font": ("Helvetica", 11), "padx": 12, "pady": 6, 
                      "bd": 2, "relief": "groove", "padx": 10}
        
        self.play_button = Button(control_frame, text="Pause", command=self.toggle_pause,
                                bg="#001f3f", fg="white", activebackground="#2980b9", 
                                activeforeground="white", **button_style)
        self.play_button.pack(side="left", padx=5)
        
        self.restart_button = Button(control_frame, text="Restart", command=self.restart_video,
                                  bg="#001f3f", fg="white", activebackground="#27ae60",
                                  activeforeground="white", **button_style)
        self.restart_button.pack(side="left", padx=5)
        
        # Toggle buttons for visibility control
        self.toggle_mainframe_btn = Button(control_frame, text="Toggle MainFrame", 
                                       command=lambda: self.toggle_visibility("MainFrame"),
                                       bg="#001f3f", fg="white", activebackground="#8e44ad",
                                       activeforeground="white", **button_style)
        self.toggle_mainframe_btn.pack(side="left", padx=5)
        
        self.toggle_classroom_btn = Button(control_frame, text="Toggle Classroom", 
                                       command=lambda: self.toggle_visibility("Classroom"),
                                       bg="#001f3f", fg="white", activebackground="#8e44ad",
                                       activeforeground="white", **button_style)
        self.toggle_classroom_btn.pack(side="left", padx=5)
        
        # Detection display with better styling
        detection_frame = Frame(self.window, bg="#d3d3d3", padx=10, pady=5)
        detection_frame.pack(fill="x")
        
        Label(detection_frame, text="Real-time Detections:", 
              font=("Helvetica", 12, "bold"), bg="#d3d3d3", fg="black").pack(anchor="w")
        
        self.detection_text = scrolledtext.ScrolledText(detection_frame, height=8, 
                                            font=("Courier", 10), wrap="word",
                                            bg="#d3d3d3", fg="#1e1e1e", padx=5, pady=5)
        self.detection_text.pack(fill="both", expand=True)
        
        # Improved info panel layout
        info_panel = Frame(self.window, bg="#d3d3d3", padx=10, pady=5)
        info_panel.pack(fill="both", expand=True)
        
        # Configure grid weights
        info_panel.columnconfigure(0, weight=1)
        info_panel.columnconfigure(1, weight=1)
        info_panel.rowconfigure(0, weight=1)
        
        # Authorization status frame
        auth_frame = Frame(info_panel, bg="#d3d3d3", padx=5, pady=5)
        auth_frame.grid(row=0, column=0, sticky="nsew")
        
        Label(auth_frame, text="MainFrame Authorization Status:", 
              font=("Helvetica", 12, "bold"), bg="#d3d3d3", fg="black").pack(anchor="w")
        
        self.auth_text = scrolledtext.ScrolledText(auth_frame, height=10,
                                               font=("Courier", 11), wrap="word", 
                                               bg="#d3d3d3", fg="#f1c40f", padx=5, pady=5)
        self.auth_text.pack(fill="both", expand=True)
        
        # Behavior alerts frame
        alert_frame = Frame(info_panel, bg="#d3d3d3", padx=5, pady=5)
        alert_frame.grid(row=0, column=1, sticky="nsew")
        
        alert_frame = Frame(info_panel, bg="#d3d3d3", padx=5, pady=5)
        alert_frame.grid(row=0, column=1, sticky="nsew")
        Label(alert_frame, text="Behavior Alerts:", 
              font=("Helvetica", 12, "bold"), bg="#d3d3d3", fg="black").pack(anchor="w")
        
        self.alert_text = scrolledtext.ScrolledText(alert_frame, height=10,
                                                font=("Courier", 11), wrap="word", 
                                                bg="#d3d3d3", fg="#2ecc71", padx=5, pady=5)
        self.alert_text.pack(fill="both", expand=True)
        
        # Status bar at bottom
        self.status_bar = Label(self.window, text="System initializing...", 
                              bd=1, relief="sunken", anchor="w", 
                              bg="#d3d3d3", fg="darkgreen", font=("Helvetica", 10))
        self.status_bar.pack(side="bottom", fill="x")
        
        # Start blinking indicator
        self._blink_realtime_indicator()
        
    def toggle_visibility(self, class_name):
        """Toggle visibility of detection classes"""
        if class_name in VISIBLE_CLASSES:
            VISIBLE_CLASSES[class_name] = not VISIBLE_CLASSES[class_name]
            status = "visible" if VISIBLE_CLASSES[class_name] else "hidden"
            self.show_alert(f"{class_name} is now {status}", "#3498db")
            
    def initialize_system(self):
        """Initialize the detection models and tracking system"""
        try:
            self.update_status("Loading models...")
            
            # Load YOLO models with error handling
            try:
                self.model1 = YOLO(MODEL1_PATH)
                self.model1.conf = 0.4  # Base confidence threshold
            except Exception as e:
                self.alert_text.insert(tk.END, f"Error loading model 1: {str(e)}\n")
                self.model1 = None
                
            try:
                self.model2 = YOLO(MODEL2_PATH)
                self.model2.conf = 0.4  # Base confidence threshold
            except Exception as e:
                self.alert_text.insert(tk.END, f"Error loading model 2: {str(e)}\n")
                self.model2 = None
            try:
                self.model3 = YOLO(MODEL3_PATH)
                self.model3.conf = 0.4  # Base confidence threshold
            except Exception as e:
                self.alert_text.insert(tk.END, f"Error loading model 3: {str(e)}\n")
                self.model3 = None
            
            # Initialize trackers and detections
            self.trackers = {}
            self.next_id = 1
            self.current_occupancy = 0
            self.classroom_box = None
            self.mainframe_box = None
            self.equipment_detections = {"SafetyJacket": [], "Helmet": []}
            self.behavior_detections = []
            self.all_detections = []
            self.human_detections_raw = []  # Store raw detections for direct occupancy check
            self.frame_count = 0
            self.classroom_history=deque(maxlen=15)
            self.missed_classroom_frames=0
            
            # Tracking for stable occupancy count
            self.person_count_history = deque(maxlen=15)
            self.tracked_entries = set()
            self.tracked_exits = set()

            self.email_sender=EmailSender()
            self.alert_timestamps={}
            
            
            # Create output directory
            os.makedirs(ALERT_DIR, exist_ok=True)
            
            # Configure tags for text coloring
            self.configure_text_tags()
            
            self.update_status("System initialized successfully")
            self.alert_text.insert(tk.END, "System initialized successfully.\n", "error")
            self.alert_text.see(tk.END)
            
        except Exception as e:
            self.update_status(f"Initialization error: {str(e)}")
            self.alert_text.insert(tk.END, f"Initialization error: {str(e)}\n", "error")
            traceback.print_exc()
    
    def configure_text_tags(self):
        """Configure text tags for colored output"""
        # General tags
        self.alert_text.tag_configure("info", foreground="#444444")
        self.alert_text.tag_configure("error", foreground="red")
        self.alert_text.tag_configure("error", foreground="darkgreen")
        self.alert_text.tag_configure("warning", foreground="#FF0000")
        
        # Alert specific tags
        self.alert_text.tag_configure("alert_cheating", foreground="#FF0000")
        self.alert_text.tag_configure("alert_phone", foreground="#FF0000")
        self.alert_text.tag_configure("alert_phonecall", foreground="#FF0000")
        self.alert_text.tag_configure("alert_texting", foreground="#FF0000")
        self.alert_text.tag_configure("alert_eating", foreground="#FF0000")
        self.alert_text.tag_configure("alert_drinking", foreground="#FF0000")
        
        # Auth specific tags
        self.auth_text.tag_configure("authorized", foreground="#2ecc71")
        self.auth_text.tag_configure("unauthorized", foreground="red")
        self.auth_text.tag_configure("info", foreground="black")
        
        # Detection text tags
        self.detection_text.tag_configure("human", foreground="darkblue")
        self.detection_text.tag_configure("equipment", foreground="red")
        self.detection_text.tag_configure("behavior", foreground="#e67e22")
        self.detection_text.tag_configure("environment", foreground="#95a5a6")

    def update_status(self, message):
        """Update status bar with message"""
        self.status_bar.config(text=message)
        self.window.update_idletasks()

    def fit_display_image(self, frame):
        """Resize frame to fit display area while maintaining aspect ratio"""
        if frame is None:
            return None
            
        # Get current dimensions of video display area
        vid_width  = self.display_width
        vid_height = self.display_height
        
        if vid_width <= 1 or vid_height <= 1:  # Not yet properly initialized
            vid_width, vid_height = 900, 500  # Default values
            
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate scaling factors
        scale_w = vid_width / w
        scale_h = vid_height / h
        
        # Use the smaller scale to ensure the entire image fits
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use higher quality interpolation method
        if scale < 1:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
        # Create a canvas with dark background
        canvas = np.ones((vid_height, vid_width, 3), dtype=np.uint8) * 20  # Very dark gray
        
        # Calculate offset to center the image
        x_offset = (vid_width - new_w) // 2
        y_offset = (vid_height - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
        
    def show_image(self, frame):
        """Display image in the UI with high quality"""
        if frame is None:
            return
            
        # Use the fitting function to scale the image properly
        fitted_frame = self.fit_display_image(frame)
        
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(fitted_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image and then to PhotoImage
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Keep reference to prevent garbage collection
        self.current_image = imgtk
        self.video_label.configure(image=self.current_image)
        
    def _blink_realtime_indicator(self):
        """Make the LIVE indicator blink with smoother transitions"""
        # Only blink if video is actually playing
        if not self.video_paused and not self.video_ended:
            self._blink_state = not self._blink_state
            self.realtime_indicator.config(fg="#e74c3c" if self._blink_state else "#c0392b")
        else:
            # Turn off or grey out when paused/ended
            self.realtime_indicator.config(fg="#7f8c8d", 
                                        text="● PAUSED" if self.video_paused else "● ENDED")
            
        self.window.after(500, self._blink_realtime_indicator)
        
    def show_alert(self, message, color):
        """Display an alert message with fade-out effect"""
        self.alert_label.config(text=message, fg=color)
        
        # Cancel any existing timer
        if hasattr(self, 'alert_timer'):
            self.window.after_cancel(self.alert_timer)
        
        # First fade to a lighter color, then clear
        self.alert_timer = self.window.after(2000, 
                                          lambda: self.alert_label.config(fg="#95a5a6"))
        self.alert_timer = self.window.after(3000, 
                                          lambda: self.alert_label.config(text=""))
    
    def toggle_pause(self):
        """Toggle video pause/play state with visual feedback"""
        self.video_paused = not self.video_paused
        
        if self.video_paused:
            self.play_button.config(text="Resume", bg="#001f3f", activebackground="#27ae60")
            # Add pause indicator to video
            if self.current_frame is not None:
                display_frame = self.current_frame.copy()
                
                # Add semi-transparent overlay
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                
                # Add pause icon
                height, width = display_frame.shape[:2]
                icon_size = min(100, width // 10)
                rect_width = icon_size // 3
                
                # Draw two pause bars
                cv2.rectangle(display_frame, 
                            (width // 2 - icon_size // 2, height // 2 - icon_size // 2),
                            (width // 2 - icon_size // 2 + rect_width, height // 2 + icon_size // 2),
                            (255, 255, 255), -1)
                            
                cv2.rectangle(display_frame, 
                            (width // 2 + icon_size // 2 - rect_width, height // 2 - icon_size // 2),
                            (width // 2 + icon_size // 2, height // 2 + icon_size // 2),
                            (255, 255, 255), -1)
                
                # Add "PAUSED" text
                cv2.putText(display_frame, "PAUSED", 
                          (width // 2 - 70, height // 2 + icon_size + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display the paused frame
                self.show_image(display_frame)
                
            self.show_alert("Video paused", "#f39c12")
        else:
            self.play_button.config(text="Pause", bg="#001f3f", activebackground="#2980b9")
            self.show_alert("Video resumed", "#2ecc71")
        
    def restart_video(self):
        """Restart video from beginning with full reset"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            # Reset video position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_ended = False
            self.video_paused = False
            
            # Update button state
            self.play_button.config(text="Pause", bg="#001f3f", activebackground="#2980b9")
            
            # Reset tracking
            self.frame_count = 0
            self.trackers = {}
            self.next_id = 1
            self.current_occupancy = 0
            self.classroom_box = None
            self.mainframe_box = None
            self.equipment_detections = {"SafetyJacket": [], "Helmet": []}
            self.behavior_detections = []
            self.all_detections = []
            self.person_count_history.clear()
            self.tracked_entries.clear()
            self.tracked_exits.clear()
            
            # Clear logs
            self.detection_text.delete(1.0, tk.END)
            self.auth_text.delete(1.0, tk.END)
            self.alert_text.delete(1.0, tk.END)
            
            # Log restart
            self.alert_text.insert(tk.END, "Video restarted\n", "error")
            self.alert_text.see(tk.END)
            self.show_alert("Video restarted", "#2ecc71")
            self.update_status("Video restarted")
    
    def process_detections(self, results1, results2,results3):
        """Process detection results from both models with improved reliability"""
        human_detections = []
        self.equipment_detections = {"SafetyJacket": [], "Helmet": []}
        self.behavior_detections = []
        self.all_detections = []
        self.human_detections_raw = []  # Store raw detections for direct occupancy check
        
        # Process both models
        for model_idx, results in enumerate([results1, results2,results3]):
            if not results or len(results) == 0:
                continue
                
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = results[0].names.get(class_id, "Unknown")
                conf = float(box.conf[0])
                
                # Extract coordinates and ensure they're valid
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Ensure coordinates are positive and box has width/height
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(x1 + 5, x2), max(y1 + 5, y2)
                except Exception as e:
                    continue  # Skip invalid boxes
                
                # Calculate box area
                area = (x2 - x1) * (y2 - y1)
                model_num = model_idx + 1
                
                # Get preferred model and confidence threshold for this class
                preferred_model, min_conf = CLASS_MODEL_ASSIGNMENT.get(label, (model_num, 0.3))
                
                # Process detection if from preferred model with sufficient confidence
                if model_num == preferred_model and conf >= min_conf:
                    # Handle different detection types
                    if label == "Human" and area > MIN_BOX_AREA:
                        human_detections.append((x1, y1, x2, y2))
                        self.human_detections_raw.append((x1, y1, x2, y2))
                        self.all_detections.append((label, (x1, y1, x2, y2), conf))
                    elif label in ["SafetyJacket", "Helmet"]:
                        self.equipment_detections[label].append(((x1, y1, x2, y2), conf))
                        self.all_detections.append((label, (x1, y1, x2, y2), conf))
                    elif label in ["Cheating", "Eating", "Drinking", "Phone", "PhoneCall", "Texting"]:
                        self.behavior_detections.append((label, (x1, y1, x2, y2), conf))
                        self.all_detections.append((label, (x1, y1, x2, y2), conf))
                    elif label == "Classroom" and area > 1000:
                        self.classroom_history.append((x1, y1, x2, y2))
                        self.missed_classroom_frames = 0
                    elif label == "MainFrame" and area > 5000:
                        self.mainframe_box = (x1, y1, x2, y2)
                        self.all_detections.append((label, (x1, y1, x2, y2), conf))
                    else:
                        self.all_detections.append((label, (x1, y1, x2, y2), conf))
        
        # Merge overlapping human detections for better tracking
        merged_detections = []
        used = [False] * len(human_detections)
        
        for i, box1 in enumerate(human_detections):
            if used[i]:
                continue
                
            final_box = list(box1)
            used[i] = True
            
            for j, box2 in enumerate(human_detections[i+1:], i+1):
                if used[j]:
                    continue
                    
                # Check if boxes significantly overlap
                iou = self.calculate_iou(final_box, box2)
                if iou > 0.1:  # Even slight overlap suggests same person
                    final_box[0] = min(final_box[0], box2[0])
                    final_box[1] = min(final_box[1], box2[1])
                    final_box[2] = max(final_box[2], box2[2])
                    final_box[3] = max(final_box[3], box2[3])
                    used[j] = True
                    
            merged_detections.append(tuple(final_box))
        # if we just got at least one new box this frame, smooth it:
        if self.classroom_history:
            xs1, ys1, xs2, ys2 = zip(*self.classroom_history)
            self.classroom_box = (
            int(median(xs1)),
            int(median(ys1)),
            int(median(xs2)),
            int(median(ys2))
            )
        else:
            self.missed_classroom_frames += 1
            if self.missed_classroom_frames < 5:
                pass
            else:
                self.classroom_box = None
         
         
        return merged_detections
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        if isinstance(box1, (list, tuple)) and len(box1) == 4 and isinstance(box2, (list, tuple)) and len(box2) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0.0
        return 0.0
        
    def update_tracking(self, human_detections):
        """Update tracking with improved matching logic"""
        # Store matched trackers and detections
        matched_trackers = set()
        matched_detections = set()
        
        # Process existing trackers first
        for tid in list(self.trackers.keys()):
            tracker = self.trackers[tid]
            best_match = -1
            best_score = float('inf')  # Lower is better
            
            # Find best matching detection for this tracker
            for i, det in enumerate(human_detections):
                if i in matched_detections:
                    continue
                    
                # Calculate IoU between tracker box and detection
                iou = tracker.calculate_iou(tracker.box, det)
                
                # Skip if IoU is too low
                if iou < 0.05:  # Reduced threshold to capture more matches
                    continue
                    
                # Calculate appearance similarity (lower is better)
                appearance_diff = tracker.appearance_similarity(det)
                
                # Calculate center distance
                tracker_center = tracker.get_center()
                det_center = ((det[0] + det[2]) // 2, (det[1] + det[3]) // 2)
                dist = math.sqrt((tracker_center[0] - det_center[0])**2 + 
                               (tracker_center[1] - det_center[1])**2)
                
                # Normalize distance score (0-1, lower is better)
                dist_score = min(1.0, dist / 300.0)  # Increased range for better matching
                
                # Combined score - weighted for better matching
                score = 0.4 * (1.0 - iou) + 0.3 * appearance_diff + 0.3 * dist_score
                
                # Update best match if this is better
                if score < best_score and score < 0.8:  # Increased threshold for more matches
                    best_score = score
                    best_match = i
            
            # If match found, update tracker
            if best_match >= 0:
                tracker.update(human_detections[best_match])
                matched_trackers.add(tid)
                matched_detections.add(best_match)
            else:
                # No match, predict new position
                tracker.predict()
                
                # Remove if missing too long
                if tracker.missed_frames > MAX_AGE:
                    # Log exit from classroom if needed
                    if tracker.in_classroom:
                        self.show_alert(f"Person left (ID: {tid}) - track lost", "black")
                        self.alert_text.insert(tk.END, 
                                             f"[TRACK LOST] Person ID {tid} track lost while in classroom\n", 
                                             "warning")
                    del self.trackers[tid]
        
        # Create new trackers for unmatched detections
        for i, det in enumerate(human_detections):
            if i in matched_detections:
                continue
                
            # Check if detection is valid
            if (det[2] - det[0]) * (det[3] - det[1]) > MIN_BOX_AREA:
                # Check if too close to an existing new tracker
                det_center = ((det[0] + det[2]) // 2, (det[1] + det[3]) // 2)
                is_duplicate = False
                
                for tid, tracker in self.trackers.items():
                    if tracker.hits < MIN_HITS:  # Only check new trackers
                        tracker_center = tracker.get_center()
                        dist = math.sqrt((tracker_center[0] - det_center[0])**2 + 
                                      (tracker_center[1] - det_center[1])**2)
                        if dist < 100:  # Close detection - likely same person
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    # Create new tracker
                    tid = self.next_id
                    self.next_id += 1
                    self.trackers[tid] = HumanTracker(tid, det)
        
        # Update occupancy status
        self.update_occupancy()
        
    def update_occupancy(self):
        """Calculate classroom occupancy with improved stability"""
        # Direct detection count - people detected in classroom
        raw_count = 0
        for box in self.human_detections_raw:
            if self.is_in_classroom(box):
                raw_count += 1
        
        # Tracking-based count - more stable
        in_classroom_count = 0
        
        # Process each tracker
        for tid, tracker in self.trackers.items():
            # Skip unstable trackers
            if tracker.hits < MIN_HITS:
                continue
                
            # Update equipment status and mainframe proximity
            tracker.update_safety_equipment(self.equipment_detections)
            tracker.near_mainframe = tracker.is_near_mainframe(self.mainframe_box)
            
            # Check if person is in classroom
            in_class_now = self.is_in_classroom(tracker.box)
            
            # Handle entry/exit with hysteresis
            if in_class_now:
                # Person currently in classroom
                if not tracker.in_classroom:
                    # Was outside, now inside - wait for stability
                    tracker.classroom_transition_time += 1
                    if tracker.classroom_transition_time >= CLASSROOM_ENTRY_FRAMES:
                        tracker.in_classroom = True
                        tracker.classroom_transition_time = 0
                        
                        # Only log entry once
                        if tid not in self.tracked_entries:
                            self.show_alert(f"Person entered (ID: {tid})", "black")
                            self.alert_text.insert(tk.END, 
                                                 f"[ENTRY] Person ID {tid} entered classroom\n", 
                                                 "error")
                            self.tracked_entries.add(tid)
                else:
                    # Still in classroom - reset transition counter
                    tracker.classroom_transition_time = 0
                    
                # Count toward occupancy if officially in classroom
                if tracker.in_classroom:
                    in_classroom_count += 1
            else:
                # Person currently outside classroom
                if tracker.in_classroom:
                    # Was inside, now outside - wait for stability
                    tracker.classroom_transition_time += 1
                    if tracker.classroom_transition_time >= CLASSROOM_EXIT_FRAMES:
                        tracker.in_classroom = False
                        tracker.classroom_transition_time = 0
                        
                        # Only log exit once
                        if tid not in self.tracked_exits:
                            self.show_alert(f"Person left (ID: {tid})", "black")
                            self.alert_text.insert(tk.END, 
                                                 f"[EXIT] Person ID {tid} left classroom\n", 
                                                 "warning")
                            self.tracked_exits.add(tid)
                else:
                    # Still outside - reset transition counter
                    tracker.classroom_transition_time = 0
        
        # Reconcile tracking and detection counts - improved blending
        if abs(in_classroom_count - raw_count) > 1 and raw_count > 0:
            # If significant difference, blend the counts with weighted average
            final_count = int((3*in_classroom_count + raw_count) / 4)
        else:
            # Otherwise use tracking count (more stable)
            final_count = in_classroom_count
        
        # Add current count to history
        self.person_count_history.append(final_count)
        
        # Use median for stability when we have enough history
        if len(self.person_count_history) >= 5:
            # Sort counts and pick median
            sorted_counts = sorted(self.person_count_history)
            self.current_occupancy = sorted_counts[len(sorted_counts) // 2]
        else:
            # Not enough history, use current count
            self.current_occupancy = final_count
            
        # Update display with new occupancy
        self.occupancy_label.config(text=f"Occupancy Count: {self.current_occupancy}")
        
    def is_in_classroom(self, box):
        if not USE_CLASSROOM_LOGIC:
            return True 
        """Check if a box is inside the classroom with improved logic"""
        if self.classroom_box is None:
            # No classroom detected, assume everything is inside
            return True
            
        # Calculate center point of the box
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        
        # Add margin for stability (more generous)
        margin = 50
        
        # Extract classroom coordinates
        c_x1, c_y1, c_x2, c_y2 = self.classroom_box
        
        # Check if center point is inside classroom (with margin)
        is_inside = (c_x1 - margin <= center_x <= c_x2 + margin and 
                   c_y1 - margin <= center_y <= c_y2 + margin)
        
        # Also check overlap percentage as alternative method
        x1 = max(box[0], c_x1)
        y1 = max(box[1], c_y1)
        x2 = min(box[2], c_x2)
        y2 = min(box[3], c_y2)
        
        # Calculate overlap area (if any)
        if x2 > x1 and y2 > y1:
            overlap_area = (x2 - x1) * (y2 - y1)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            overlap_percentage = overlap_area / box_area
            
            # If significant overlap, consider inside
            if overlap_percentage > 0.3:
                return True
                
        return is_inside
                
    def draw_detections(self, frame):
        """Draw detection results on frame with improved visualization"""
        display_frame = frame.copy()
        
        # Draw classroom boundary if available
        if self.classroom_box is not None and VISIBLE_CLASSES.get("Classroom", True):
            x1, y1, x2, y2 = self.classroom_box
            color = COLOR_MAP.get("Classroom", (200, 200, 200))
            thickness = BBOX_THICKNESS.get("Classroom", BBOX_THICKNESS["default"])
            
            # Draw main boundary
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with background for better visibility
            label = "Classroom"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x1, y1-text_h-5), (x1+text_w+10, y1), color, -1)
            cv2.putText(display_frame, label, (x1+5, y1-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw margin area with dashed line
            margin = 50
            expanded = (x1-margin, y1-margin, x2+margin, y2+margin)
            
            # Draw dashed line for margin
            dash_length = 15
            for i in range(0, 360, dash_length):
                angle_rad = i * math.pi / 180
                
                # Calculate points along the rectangle
                if i < 90:  # Top edge
                    progress = i / 90
                    pt1 = (int(expanded[0] + progress * (expanded[2] - expanded[0])), expanded[1])
                    pt2 = (int(pt1[0] + dash_length/360 * (expanded[2] - expanded[0])), expanded[1])
                elif i < 180:  # Right edge
                    progress = (i - 90) / 90
                    pt1 = (expanded[2], int(expanded[1] + progress * (expanded[3] - expanded[1])))
                    pt2 = (expanded[2], int(pt1[1] + dash_length/360 * (expanded[3] - expanded[1])))
                elif i < 270:  # Bottom edge
                    progress = (i - 180) / 90
                    pt1 = (int(expanded[2] - progress * (expanded[2] - expanded[0])), expanded[3])
                    pt2 = (int(pt1[0] - dash_length/360 * (expanded[2] - expanded[0])), expanded[3])
                else:  # Left edge
                    progress = (i - 270) / 90
                    pt1 = (expanded[0], int(expanded[3] - progress * (expanded[3] - expanded[1])))
                    pt2 = (expanded[0], int(pt1[1] - dash_length/360 * (expanded[3] - expanded[1])))
                
                # Draw dash only for even numbered segments
                if (i // dash_length) % 2 == 0:
                    cv2.line(display_frame, pt1, pt2, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Draw MainFrame boundary if available
        if self.mainframe_box is not None and VISIBLE_CLASSES.get("MainFrame", True):
            x1, y1, x2, y2 = self.mainframe_box
            color = COLOR_MAP.get("MainFrame", (255, 0, 255))
            thickness = BBOX_THICKNESS.get("MainFrame", BBOX_THICKNESS["default"])
            
            # Draw main boundary
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with background
            label = "MainFrame"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x1, y1-text_h-5), (x1+text_w+10, y1), color, -1)
            cv2.putText(display_frame, label, (x1+5, y1-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw proximity detection area - smoother circle
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw authorization zone with semi-transparent overlay
            radius = 200
            overlay = display_frame.copy()
            cv2.circle(overlay, (cx, cy), radius, (180, 0, 180), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
            
            # Draw circle outline
            cv2.circle(display_frame, (cx, cy), radius, (255, 0, 255), 1, cv2.LINE_AA)
            
            # Add "Authorization Zone" text
            zone_text = "Authorization Zone"
            (text_w, text_h), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(display_frame, zone_text, 
                      (cx - text_w//2, cy + radius + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
        
        # Draw equipment and behavior detections
        for label, box, conf in self.all_detections:
            if VISIBLE_CLASSES.get(label, True) and label not in ["Human", "MainFrame", "Classroom"]:
                color = COLOR_MAP.get(label, (255, 255, 255))
                thickness = BBOX_THICKNESS.get(label, BBOX_THICKNESS["default"])
                x1, y1, x2, y2 = box
                
                # Draw box with semi-transparent fill for better visibility
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
                
                # Draw box outline
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Create background for text
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
                cv2.rectangle(display_frame, (x1, y1-text_h-5), (x1+text_w+5, y1), color, -1)
                
                # Draw text
                cv2.putText(display_frame, label_text, (x1+2, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness)
        
        # Draw human trackers
        for tid, tracker in self.trackers.items():
            if tracker.box is not None and tracker.hits >= MIN_HITS and VISIBLE_CLASSES.get("Human", True):
                # Use the smoothed box if available for better visualization
                if hasattr(tracker, 'smoothed_box') and tracker.smoothed_box:
                    x1, y1, x2, y2 = map(int, tracker.smoothed_box)
                else:
                    x1, y1, x2, y2 = map(int, tracker.box)
                
                # Choose color based on status
                if tracker.near_mainframe:
                    color = (0, 255, 0) if tracker.authorized else (0, 0, 255)
                elif tracker.in_classroom:
                    color = (255, 165, 0)  # Orange for in-classroom
                else:
                    color = (200, 200, 200)  # Grey for outside
                
                # Draw bounding box with thicker lines
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add semi-transparent fill
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
                
                # Draw ID with better visibility
                id_text = f"ID:{tid}"
                (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1-text_h-5), (x1+text_w+10, y1), color, -1)
                cv2.putText(display_frame, id_text, (x1+5, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Show authorization status if near mainframe
                if tracker.near_mainframe:
                    auth_text = "AUTHORIZED" if tracker.authorized else "UNAUTHORIZED"
                    text_color = (0, 255, 0) if tracker.authorized else (0, 0, 255)
                    
                    # Draw authorization status with background
                    (text_w, text_h), _ = cv2.getTextSize(auth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_frame, (x1, y2+5), (x1+text_w+10, y2+5+text_h+5), (0, 0, 0), -1)
                    cv2.putText(display_frame, auth_text, (x1+5, y2+text_h+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    
                    # Draw equipment status with clearer indication
                    y_offset = 40
                    for eq_type, has_eq in tracker.safety_equipment.items():
                        eq_color = (0, 255, 0) if has_eq else (0, 0, 255)
                        eq_status = "✓" if has_eq else "✗"
                        eq_text = f"{eq_type}: {eq_status}"
                        
                        # Create background for better visibility
                        (text_w, text_h), _ = cv2.getTextSize(eq_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(display_frame, (x1, y2+y_offset), (x1+text_w+5, y2+y_offset+text_h+5), 
                                    (0, 0, 0), -1)
                                    
                        cv2.putText(display_frame, eq_text, 
                                  (x1+2, y2+y_offset+text_h), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, eq_color, 1, cv2.LINE_AA)
                        y_offset += 25
                        
                # Indicate classroom presence
                elif tracker.in_classroom:
                    # Create background for better visibility
                    in_class_text = "IN CLASS"
                    (text_w, text_h), _ = cv2.getTextSize(in_class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display_frame, (x1, y2+5), (x1+text_w+5, y2+5+text_h+5), (0, 0, 0), -1)
                    
                    cv2.putText(display_frame, in_class_text, (x1+2, y2+5+text_h), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Display info with semi-transparent background
        info_overlay = display_frame.copy()
        cv2.rectangle(info_overlay, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(info_overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(display_frame, f"Time: {current_time}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(display_frame, f"Occupancy: {self.current_occupancy}", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(display_frame, f"Frame: {self.frame_count}", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return display_frame

    def update_detection_display(self):
        """Update detection info display with improved formatting"""
        self.detection_text.delete(1.0, tk.END)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.detection_text.insert(tk.END, f"[{current_time}] Frame: {self.frame_count}\n\n", "info")
        
        # First show important environment elements
        if self.classroom_box is not None:
            self.detection_text.insert(tk.END, "")
            
        if self.mainframe_box is not None:
            self.detection_text.insert(tk.END,"")
            
        self.detection_text.insert(tk.END, f"Current occupancy: {self.current_occupancy}\n\n", "info")
        
        # Show human trackers
        self.detection_text.insert(tk.END, "--- HUMANS ---\n", "info")
        
        humans_shown = False
        for tid, tracker in sorted(self.trackers.items()):
            if tracker.box is not None and tracker.hits >= MIN_HITS:
                humans_shown = True
                center_x, center_y = tracker.get_center()
                
                # Set color based on status
                tag = f"human"
                if tracker.in_classroom:
                    status_tag = "in_class"
                    self.detection_text.tag_configure(status_tag, foreground="darkblue")
                else:
                    status_tag = "outside"
                    self.detection_text.tag_configure(status_tag, foreground="darkblue")
                
                # Build status text
                status_text = "IN CLASS" if tracker.in_classroom else "OUTSIDE"
                
                # Insert base human info
                self.detection_text.insert(tk.END, f"Human ID:{tid} at ({center_x},{center_y}) ", tag)
                self.detection_text.insert(tk.END, f"{status_text}\n", status_tag)
                
                # Show authorization info if near mainframe
                if tracker.near_mainframe:
                    auth_tag = "authorized" if tracker.authorized else "unauthorized"
                    auth_status = "AUTHORIZED" if tracker.authorized else "UNAUTHORIZED"
                    
                    self.detection_text.insert(tk.END, f"  MainFrame: {auth_status}", auth_tag)
                    
                    # Show equipment status
                    eq_info = []
                    for eq_type, has_eq in tracker.safety_equipment.items():
                        status = "YES" if has_eq else "NO"
                        eq_info.append(f"{eq_type}: {status}")
                    
                    self.detection_text.insert(tk.END, f" ({', '.join(eq_info)})\n", "equipment")
        
        if not humans_shown:
            self.detection_text.insert(tk.END, "  No tracked humans\n", "info")
            
        # Show equipment detections
        self.detection_text.insert(tk.END, "\n--- SAFETY EQUIPMENT ---\n", "info")
        
        equipment_shown = False
        for eq_type in ["SafetyJacket", "Helmet"]:
            for (box, conf) in self.equipment_detections[eq_type]:
                equipment_shown = True
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                self.detection_text.insert(tk.END, 
                                        f"{eq_type} at ({center_x},{center_y}) Conf:{conf:.2f}\n", 
                                        "equipment")
        
        if not equipment_shown:
            self.detection_text.insert(tk.END, "  No safety equipment detected\n", "info")
            
        # Show behavior detections
        self.detection_text.insert(tk.END, "\n--- BEHAVIORS ---\n", "info")
        
        behaviors_shown = False
        for label, box, conf in sorted(self.behavior_detections, key=lambda x: x[0]):
            behaviors_shown = True
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            self.detection_text.insert(tk.END, 
                                     f"{label} at ({center_x},{center_y}) Conf:{conf:.2f}\n", 
                                     "behavior")
                                     
        if not behaviors_shown:
            self.detection_text.insert(tk.END, "  No behaviors detected\n", "info")
    
    def update_logs(self):

        # Ensure latest log is visible
        self.auth_text.see(tk.END)
        self.detection_text.see(tk.END)
        self.alert_text.see(tk.END)

        """Update authorization and behavior logs with improved formatting"""
        # Clear logs
        self.auth_text.delete(1.0, tk.END)
        
        # Track authorization info
        auth_info = []
        
        # Process trackers for authorization
        for tid, tracker in sorted(self.trackers.items()):
            if tracker.near_mainframe and tracker.hits >= MIN_HITS:
                status = "AUTHORIZED" if tracker.authorized else "UNAUTHORIZED"
                tag = "authorized" if tracker.authorized else "unauthorized"
                
                # Format equipment status
                eq_status = []
                for eq_type, has_eq in tracker.safety_equipment.items():
                    status_text = "✓" if has_eq else "✗"
                    eq_status.append(f"{eq_type}: {status_text}")
                
                # Build message and save for sorting
                message = f"Person ID {tid}: {status}\n  {' | '.join(eq_status)}\n"
                auth_info.append((tid, tracker.authorized, message, tag))
                     #--- send email on unauthorized access ---
                if not tracker.authorized:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    details = {
                        "Alert Type": "Unauthorized",
                        "Time":       timestamp,
                        "Person ID":  f"{tracker.id}",
                        "Location":   f"{tracker.get_center()}",
                        "SafetyJacket": "YES" if tracker.safety_equipment["SafetyJacket"] else "NO",
                        "Helmet":       "YES" if tracker.safety_equipment["Helmet"]       else "NO"
                        }
                    self.send_alert_email("Unauthorized", details, tracker.box)
        # Sort by authorized status then ID
        auth_info.sort(key=lambda x: (not x[1], x[0]))
        
        # Display sorted authorization info
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.auth_text.insert(tk.END, f"[{current_time}] MainFrame Access Status:\n\n", "info")
        
        for _, _, message, tag in auth_info:
            self.auth_text.insert(tk.END, message, tag)
            self.auth_text.insert(tk.END, "\n")  # Add extra space between entries
        
        # Show placeholder if no one near mainframe
        if not auth_info:
            self.auth_text.insert(tk.END, "No persons near MainFrame\n", "info")
            
        # Don't clear alert log to maintain history
        # Just add new alerts
        
        # Process behavior alerts
        if self.behavior_detections:
            # Add timestamp to behavior alerts
            current_time = time.strftime("%H:%M:%S", time.localtime())
            
            # Sort by type for better readability
            sorted_behaviors = sorted(self.behavior_detections, key=lambda x: x[0])
            
            # Group behaviors of the same type
            behavior_groups = {}
            for label, box, conf in sorted_behaviors:
                if label not in behavior_groups:
                    behavior_groups[label] = []
                behavior_groups[label].append((box, conf))
            
            # Report behaviors by group
            for label, detections in behavior_groups.items():
                alert_tag    = f"alert_{label.lower()}"
                current_time = time.strftime("%H:%M:%S", time.localtime())
                if label in ["Cheating", "Phone", "PhoneCall", "Texting"]:
                    self.alert_text.insert(
                        tk.END,
                        f"[{current_time}] ⚠️ IMPORTANT: {label} behavior detected ",
                        "warning")
                    self.alert_text.insert(
                        tk.END,
                        f"({len(detections)} instances)\n",
                        alert_tag)
                    highest_box, highest_conf = max(detections, key=lambda x: x[1])
                    cx = (highest_box[0] + highest_box[2]) // 2
                    cy = (highest_box[1] + highest_box[3]) // 2
                    associated = "Unknown"
                    for tid, tracker in self.trackers.items():
                        if tracker.box and math.hypot(
                            cx - tracker.get_center()[0],
                            cy - tracker.get_center()[1]
                            ) < 100:
                            associated = f"Person ID {tid}" + (
                                " (in classroom)" if tracker.in_classroom else ""
                                )
                            break
                    details = {
                        "Alert Type":        label,
                        "Time":              current_time,
                        "Confidence":        f"{highest_conf:.2f}",
                        "Total Instances":   len(detections),
                        "Location":          f"({cx}, {cy})",
                        "Associated Person": associated
                        }
                    self.send_alert_email(label, details, highest_box)
                else:
                    self.alert_text.insert(
                        tk.END,
                        f"[{current_time}] {label} behavior detected ",
                        "info"
                        )
                    self.alert_text.insert(
                        tk.END,
                        f"({len(detections)} instances)\n",
                        alert_tag
                        )
                    for box, conf in detections:
                        cx = (box[0] + box[2]) // 2
                        cy = (box[1] + box[3]) // 2
                        association = "unknown location"
                        for tid, tracker in self.trackers.items():
                            if tracker.box and math.hypot(
                                cx - tracker.get_center()[0],
                                cy - tracker.get_center()[1]
                                ) < 100:
                                    association = f"Person ID {tid}" + (
                                    " (in classroom)" if tracker.in_classroom else ""
                                    )
                                    break
                        self.alert_text.insert(
                            tk.END,
                            f"  - At ({cx},{cy}) - {association} - conf:{conf:.2f}\n",
                            
                            )
                    self.alert_text.insert(tk.END, "\n")
                    
    def capture_alert_screenshot(self,alert_type,region=None):
        if self.current_frame is None:
            return None 
        screenshot=self.draw_detections(self.current_frame.copy())
        if region is not None:
            x1,y1,x2,y2=region
            cv2.rectangle(screenshot,(x1,y1),(x2,y2),(0,0,225),(3))
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = os.path.join(ALERT_DIR, f"{alert_type}_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, screenshot)
                self.alert_text.insert(tk.END, f"Alert screenshot saved: {filename}\n", "info")
            except Exception as e:
                self.alert_text.insert(tk.END, f"Error saving screenshot: {str(e)}\n", "error")
        return screenshot
    def send_alert_email(self,alert_type,details,region=None):
        if not EMAIL_ENABLED:
            return
        if not EMAIL_ALERT_CLASSES.get(alert_type,False):
            return
        screenshot=self.capture_alert_screenshot(alert_type,region)
        if screenshot is not None:
            if self.email_sender.queue_email(alert_type, screenshot, details):
                self.alert_text.insert(tk.END, f"Email alert queued for {alert_type}\n", "error")
            else:
                self.alert_text.insert(tk.END, f"Email alert skipped (cooldown) for {alert_type}\n", "info")
        
    def update_frame(self, cap):
        """Process a video frame with improved error handling and video display"""
        if self.video_paused or self.video_ended:
            self.window.after(100, lambda: self.update_frame(cap))
            return
            
        try:
            # Read frame
            ret, frame = cap.read()
            if not ret or frame is None:
                self.video_ended = True
                self.alert_text.insert(tk.END, "Video processing completed\n", "info")
                self.update_status("Video ended")
                self.show_alert("Video ended", "#e74c3c")
                return
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Update frame counter
            self.frame_count += 1
            
            # Skip first few frames for initialization
            if self.frame_count < 3:
                self.window.after(30, lambda: self.update_frame(cap))
                return
            
            # Measure processing time
            start_time = time.time()
            
            # Process detection models with error handling
            results1 = None
            results2 = None
            results3= None
            
            try:
                if self.model1:
                    results1 = self.model1.predict(frame.copy(), verbose=False)
            except Exception as e:
                self.alert_text.insert(tk.END, f"Model 1 detection error: {str(e)}\n", "error")
                
            try:
                if self.model2:
                    results2 = self.model2.predict(frame.copy(), verbose=False)
            except Exception as e:
                self.alert_text.insert(tk.END, f"Model 2 detection error: {str(e)}\n", "error")
            
            try:
                if self.model3:
                    results3=self.model3.predict(frame.copy(),verbose=False)
            except Exception as e:
                self.alert_text.insert(tk.END,f"Model 2 detection error: {str(e)}\n", "error")
            
            # Process detections and update tracking
            human_detections = self.process_detections(results1, results2,results3)
            self.update_tracking(human_detections)
            # ————— START LOGGING EACH DETECTION —————
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for label, (x1, y1, x2, y2), conf in self.all_detections:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                self.detections_log.append({
                    "timestamp": ts,
                    "frame":     self.frame_count,
                    "class":     label,
                    "conf":      conf,
                    "x1":        x1, "y1": y1,
                    "x2":        x2, "y2": y2,
                    "center_x":  cx, "center_y": cy
                    })
                # ————— END LOGGING EACH DETECTION —————
            # Draw visualization
            display_frame = self.draw_detections(frame.copy())
            
            # Update displays
            self.show_image(display_frame)
            self.update_detection_display()
            self.update_logs()
            
            # Update status with frame rate info
            process_time = (time.time() - start_time) * 1000
            fps = 1000 / max(1, process_time)
            self.update_status(f"Processing frame {self.frame_count} - {fps:.1f} FPS")
            
            # Calculate delay based on processing time to maintain stable framerate
            target_fps = 30  # Target 30 FPS
            delay = max(1, int(1000/target_fps - process_time))
            
            self.window.after(delay, lambda: self.update_frame(cap))
            
        except Exception as e:
            error_msg = f"Frame {self.frame_count} error: {str(e)}\n"
            self.alert_text.insert(tk.END, error_msg, "error")
            traceback.print_exc()
            self.window.after(100, lambda: self.update_frame(cap))
    
    def on_closing(self):
        """Handle window closing event properly"""
            # ————— START DUMP LOG TO EXCEL —————
        if self.detections_log:
            df = pd.DataFrame(self.detections_log)
            out_path = os.path.join(ALERT_DIR, "detection_log.xlsx")
            df.to_excel(out_path, index=False)
            print(f"Saved detection log to {out_path}")
    # ————— END DUMP LOG TO EXCEL —————
        try:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if hasattr(self,'email_sender'):
                self.email_sender.shutdown() 
        except:
            pass
        self.window.destroy()
    
    def run(self, video_path):
        """Start processing video with proper initialization and error handling"""
        try:
            # show “Webcam #0” instead of crashing when video_path is int
            if isinstance(video_path, int):
                source_name = f"Webcam #{video_path}"
            else:
                source_name = os.path.basename(video_path)
                self.update_status(f"Opening video: {source_name}")            
            # Try different approaches to open the video
            self.cap = None
            
            # Try with default backend
            try:
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    self.cap = None
            except Exception as e:
                self.alert_text.insert(tk.END, f"Error with default backend: {str(e)}\n", "error")
            
            # Try with FFMPEG backend if default failed
            if self.cap is None:
                try:
                    self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened():
                        self.cap = None
                except Exception as e:
                    self.alert_text.insert(tk.END, f"Error with FFMPEG backend: {str(e)}\n", "error")
            
            # Try with DirectShow on Windows
            if self.cap is None and os.name == 'nt':
                try:
                    self.cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = None
                except Exception as e:
                    self.alert_text.insert(tk.END, f"Error with DirectShow backend: {str(e)}\n", "error")
            
            if self.cap is None:
                self.alert_text.insert(tk.END, f"Error: Could not open video {video_path}\n", "error")
                self.update_status(f"Failed to open video")
                return
            
            # Get video info
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / max(1, fps)
            
            # Update file info display
            # Update file info display
            if isinstance(video_path, int):
                info_text = f"Webcam #{video_path}"
            else:
                info_text = f""

            self.file_info_label.config(text=info_text)
            # Reset tracking state
            self.trackers = {}
            self.next_id = 1
            self.current_occupancy = 0
            self.classroom_box = None
            self.mainframe_box = None
            self.equipment_detections = {"SafetyJacket": [], "Helmet": []}
            self.behavior_detections = []
            self.all_detections = []
            self.human_detections_raw = []
            self.frame_count = 0
            self.person_count_history.clear()
            self.tracked_entries.clear()
            self.tracked_exits.clear()
            
            # Log video info
            self.alert_text.insert(tk.END, 
                                 f"Video loaded: {width}x{height} @ {fps:.1f} fps ({frame_count} frames, {duration:.1f}s)\n", 
                                 "info")
            self.alert_text.see(tk.END)
            self.update_status(f"Video loaded: {width}x{height} @ {fps:.1f} fps")
            
            # Start processing
            self.video_ended = False
            self.video_paused = False
            
            # Make sure buttons have correct state
            self.play_button.config(text="Pause", bg="#001f3f", activebackground="#2980b9")
            
            # Force UI update before starting processing
            self.window.update_idletasks()
            
            # Start video processing with a slight delay to ensure UI is ready
            self.window.after(500, lambda: self.update_frame(self.cap))
            
            # Start UI loop
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.window.mainloop()
            
        except Exception as e:
            error_msg = f"Error initializing video: {str(e)}\n"
            self.alert_text.insert(tk.END, error_msg, "error")
            self.update_status(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            # Ensure we clean up
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()

if __name__ == "__main__":
    try:
        # Create and configure the application
        monitor = ClassroomMonitor()
        
        # Set up closing handler
        monitor.window.protocol("WM_DELETE_WINDOW", monitor.on_closing)
        
        # Start video processing
        monitor.run(VIDEO_PATH)
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
