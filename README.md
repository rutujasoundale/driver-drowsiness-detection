🚗 AI Driver Drowsiness Detection System
📌 Overview

This project implements a **real-time Driver Drowsiness Detection System** using Computer Vision.
The system monitors the driver's face through a webcam and detects **eye closure, yawning, and fatigue** to alert the driver and help prevent road accidents.

The system works by detecting **facial landmarks**, calculating **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**, and triggering alerts when signs of drowsiness are detected.

---

🎯 Features

Real-time face detection
Eye landmark detection
Eye Aspect Ratio (EAR) calculation
Yawning detection using Mouth Aspect Ratio (MAR)
Driver drowsiness alert
Facial landmark visualization
Webcam-based monitoring

---

🧠 How the System Works

The system follows this pipeline:

Webcam → Frame Capture → Face Detection → Facial Landmark Detection →
Eye Landmark Extraction → EAR Calculation →
Mouth Landmark Extraction → MAR Calculation →
Drowsiness Detection → Alert System

---

👁 Eye Aspect Ratio (EAR)

EAR is used to determine whether the eyes are open or closed.

EAR Formula:

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 ||p1 - p4||)

If EAR drops below a threshold for multiple frames, the system assumes the driver is sleepy.

---

👄 Mouth Aspect Ratio (MAR)

MAR is used to detect yawning.

MAR Formula:

MAR = (||p3 - p7|| + ||p4 - p6||) / (2 ||p1 - p5||)

If MAR exceeds the threshold, yawning is detected.

---

🧰 Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Computer Vision

---

📂 Project Structure

```
driver-drowsiness-detection/
│
│   ├── main.py                # Main program to run the system
│   ├── vision_agent.py        # Vision agent decision logic
│   ├── eye_detection.py       # EAR calculation
│   ├── yawn_detection.py      # MAR calculation
│   └── utils.py               # helper functions (distance, landmarks)
│
│
├── screenshots/
│   └── output_example.png     # sample output screenshot
│
│
├── requirements.txt           # python dependencies
├── README.md                  # project documentation
├── .gitignore                 # ignored files
└── LICENSE                    # optional license```

---

# ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/driver-drowsiness-detection.git
```

### 2. Navigate to project folder

```
cd driver-drowsiness-detection
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the Python script:

```
python main.py
```
```
The webcam will open and start detecting:

* Face
* Eye closure
* Yawning
* Drowsiness

---

# 🚨 Alerts

The system displays alerts when:

Eyes closed for multiple frames → **DROWSINESS ALERT**

Wide mouth opening → **Yawning Detected**

---

# ⚠️ Limitations

* Performance depends on lighting conditions
* Glasses may affect eye detection
* Large head rotations may reduce accuracy
* Camera must clearly capture the driver's face

---

# 🚀 Future Improvements

* Head pose detection
* Blink rate monitoring
* Mobile phone usage detection
* Alarm sound alerts
* Deep learning fatigue classification

---

# 📚 Applications

* Driver safety systems
* Smart vehicles
* Fleet monitoring
* Transportation safety
* AI-based vehicle monitoring

---

# 👩‍💻 Author

Rutuja Soundale

---

# ⭐ If you like this project

Give the repository a **star ⭐ on GitHub**.
#If you want you can contribute to improve this project mainly with vision-agent framework , Feel free to raise issues

