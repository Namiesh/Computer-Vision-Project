# Gesture Drawing - Computer Vision Virtual Drawing Application

An interactive, gesture-controlled drawing application built in Python using **OpenCV**, **MediaPipe**, and **NumPy**. This application enables users to draw in real time using their webcam feed by tracking their hand gestures and finger coordinates.

---

## 📂 Project Architecture & File Structure

The project strictly follows modular, object-oriented design principles:

```
CV_PROJECT/
│
├── hand_tracker.py     # Class HandTracker: MediaPipe initialization, landmark extraction, and finger state tracking
├── canvas_ui.py        # Class CanvasUI: UI buttons, color palette, canvas drawing buffer, and frame blending
├── main.py             # Controller script: Webcam loop, gesture mode routing, and keyboard event handling
├── requirements.txt    # Project dependencies
└── README.md           # Documentation and run instructions
```

---

## 🚀 Key Features

- **21-Point Hand Landmark Tracking**: Powered by MediaPipe Hands for high precision and low latency.
- **Gesture Mode Switching**:
  - ✌️ **Selection Mode (Index + Middle Fingers UP)**: Pauses drawing. Hover your index finger over the top UI header to switch tools (`Blue`, `Green`, `Red`, `Yellow`, `Eraser`) or click `Clear Screen`.
  - ☝️ **Drawing / Erasing Mode (ONLY Index Finger UP)**: Draws smooth continuous colored strokes (or erases lines when the Eraser tool is active).
  - ✊ **Standby / Pause (Fingers DOWN / Fist)**: Safely pauses drawing without connecting unwanted stroke artifacts.
- **Top UI Toolbar**: 6 distinct bounding boxes (`CLEAR SCREEN`, `BLUE`, `GREEN`, `RED`, `YELLOW`, `ERASER`) with active tool highlighting.
- **Mirror View**: The webcam frame is flipped horizontally for intuitive, natural interaction.
- **Layer Blending**: Renders vibrant, anti-aliased drawing strokes directly over the live webcam feed.
- **HUD & Status Bar**: Displays real-time FPS, active interaction mode (`DRAWING`, `ERASING`, `SELECTION`), and keyboard shortcuts.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8 to 3.12.
- A functional webcam.

### 2. Clone / Open Project Directory
```bash
cd CV_PROJECT
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Running the Application

Execute the controller script:

```bash
python main.py
```

### Keyboard Shortcuts:
- **`E`**: Switch active tool to **Eraser**.
- **`C`**: **Clear** the entire canvas.
- **`Q` or `ESC`**: Exit application cleanly.
