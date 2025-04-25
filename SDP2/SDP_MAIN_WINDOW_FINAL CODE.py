from tkinter import Tk, Button, Label, Frame
from PIL import Image, ImageTk
import os
import subprocess

root = Tk()
root.title("CS06 SDP Monitoring")
root.geometry("800x600")
root.configure(bg="#f3f6fb")  # light background

# Main container frame
main_frame = Frame(root, bg="#f3f6fb")
main_frame.pack(expand=True)

# Load the new high-res logo
logo_path = r"D:\SDP2 NEW\university_cheating_sys_rodha\logo.png"
if os.path.exists(logo_path):
    logo_img = Image.open(logo_path)
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(root, image=logo_photo, bg="#f3f6fb")
    logo_label.image = logo_photo
    logo_label.place(x=20, y=10)
else:
    logo_label = Label(root, text="Logo", font=("Arial", 12), bg="#f3f6fb")
    logo_label.place(x=20, y=10)

# Title
title_label = Label(main_frame, text="CS06 SDP \n Classroom Occupancy Monitoring and Behavior Analysis System", font=("Segoe UI", 19, "bold"), bg="#f3f6fb", fg="#001f3f")
title_label.pack(pady=(120, 30))

# Button styles
button_style = {
    "font": ("Segoe UI", 16, "bold"),
    "width": 20,
    "bg": "#001f3f",  # Dark blue
    "fg": "white",
    "activebackground": "#003366",
    "activeforeground": "white",
    "bd": 0,
    "highlightthickness": 0,
    "padx": 10,
    "pady": 6
}

def launch_behavior_system():
    subprocess.Popen(["python", r"D:\SDP2 NEW\BEHAVIOR_SYSTEM_FINAL_CODE.py"])

def launch_cheating_system():
    subprocess.Popen(["python", r"D:\SDP2 NEW\CHEATING_SYSTEM_FINAL_CODE.py"])

def launch_attendance_system():
    subprocess.Popen(["python", r"D:\SDP2 NEW\ATTENDACNE_FINAL FINAL.py"])

button1 = Button(main_frame, text="Behavior System", command=launch_behavior_system, **button_style)
button1.pack(pady=10)

button2 = Button(main_frame, text="Cheating System", command=launch_cheating_system, **button_style)
button2.pack(pady=10)
button3 = Button(main_frame, text="Attendance System", command=launch_attendance_system, **button_style)
button3.pack(pady=10)

# Footer
footer = Label(
    root,
    text="Developed by:\nAlanood Alharmoodi - Rodha Alhosani\nYasmine Benkhelifa - Shaikha Alhammadi",
    font=("Segoe UI", 15),
    bg="#f3f6fb",
    fg="#001f3f",
    justify="center"
)
footer.pack(side="bottom", pady=10)
root.mainloop()
