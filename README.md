# Classroom Occupancy Monitoring and Behavior Analysis System

Welcome to the repository for our Classroom Occupancy Monitoring and Behavior Analysis System.
This AI-driven project focuses on enhancing safety, improving learning environments, and optimizing resource utilization within educational institutions using cutting-edge computer vision technologies.
This repository contains all source codes, reports, documentation, and additional resources developed over the course of these two projects.


## 📁 Project Structure

/SDP I

    - Object Detection System (mug, cup, bottle, human, restricted area detection)

    
/SDP II

    - Behavior Analysis System (eating, drinking, phone calls, texting, object detection)
    - Cheating Detection System (cheating class, phone calls, texting, object detection)
    - Attendance Monitoring System (student occupancy and logging)

## 📜 Project Overview

This system is built in two phases:
### SDP I (Phase 1):
Focused on Object Detection, where the system detects:

Unauthorized objects (cups & mugs) inside labs

Unauthorized access to restricted (mainframe) areas

### SDP II (Phase 2):
Focused on Behavior Analysis and Occupancy Monitoring, introducing:

Behavior System: Detects disruptive activities like eating, drinking, texting, and phone calls

Cheating Detection System: Identifies phone calls, texting during exams, and cheating class behaviors

Attendance System: Tracks the number of students present for attendance 

### Key Features:
Real-time object and behavior detection using YOLOv8s

Illegal object and restricted area alerts

Automated behavior monitoring to minimize manual supervision

Occupancy tracking for attendance and analytics

Dashboard for live monitoring, event logs, and location mapping

Email alerts to notify mentors instantly about significant events or policy violations

## ⚙️ Technologies Used
Python

YOLOv8s (You Only Look Once v8 small version - for fast and accurate detections)

OpenCV (computer vision operations)

MakeSense.ai (data annotation)

TensorFlow/PyTorch (training and fine-tuning models)

Flask / Streamlit (for dashboards — if applicable)

## 📊 Dashboard
A user-friendly dashboard provides:
Real-time alerts for detected illegal objects and behaviors

Event logging with time stamps for tracking detected incidents

Location coordinates for detected illegal activities within the classroom or restricted areas

Occupancy reports for tracking student attendance and room usage

Email alerts to notify administrators or staff about significant events or unauthorized activities


## ✍️ Authors
Alanood Alharmoodi       100053854 

Rodha Alhosani           100058376 

Shaikha Alhammadi        100058710 

Yasmine Benkhelifa       100059531

