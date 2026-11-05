🛡️ Integrated Violence Detection System  
Real-time Audio + Video Threat Detection using YOLO, Mediapipe & Deep Learning

This project is a fusion of computer vision, sound analysis, and smart heuristics to detect violent behaviour in real time.
It listens. It watches. It thinks.
And when the threat spikes, it shouts.

Perfect for demos, research, personal security experiments, or just geeking out with AI systems.

🚀 What This System Does

The system continuously monitors:

🎥 Visual Activity

Uses YOLOv8 for high-speed person detection

Tracks body keypoints (wrists, shoulders, head) using Mediapipe Pose

Measures wrist speed to detect violent swings

Smooths jitter using rolling averages

Generates per-person violence risk %

🎤 Audio Activity

Captures live audio via sounddevice

Computes MFCC features

Classifies violence with a custom PyTorch model

Produces a confidence score

Adds cooldowns to prevent spam

🎯 Smart Threat Fusion

Both streams combine into a single threat score:

combined = 0.6 * visual_threat + 0.4 * audio_threat


If the threat goes too high →
Notifications pop. Alarm plays. A hero is born.

🧠 Features at a Glance

Real-time camera feed processing

Multi-person detection (up to 3 targets)

Wrist-speed–based violence scoring

Stylish risk bars + integrated threat panel overlay

Audio+Video fusion detection

Smart alerting system with cooldowns

Optional alarm sound

FPS monitoring

Smooth, color-coded UI

Fully modular and customizable

🛠️ Requirements
Python Libraries

You’ll need to install these:

pip install opencv-python numpy sounddevice librosa torch ultralytics mediapipe

Additional Notes

Works best on macOS/Linux (notification + alarm)

Windows also works, but alert sounds may need adaptation

Requires a webcam and optionally a microphone

YOLO model file auto-downloads on first run

Audio model must be placed here:

fight-detection/models/fight_cnn.pth

📦 Running the System

Just run:

python main.py


You'll see:

Bounding boxes

Skeletal landmarks

Risk meters

Threat panel

Audio status

FPS

Press q or ESC to exit.

🧩 Folder Structure (Recommended)
.
├── main.py
├── long_alarm.wav
├── fight-detection/
│   └── models/
│       └── fight_cnn.pth
└── README.md

⚙️ Customization

You can tweak almost everything:

Change violence thresholds

Turn audio detection on/off

Adjust YOLO confidence

Track more people

Modify risk weighting

Add logs, analytics, cloud uploads — whatever you like

The code is intentionally readable and hack-friendly.

🎬 Demo Output (Example)
ALERT: Combined threat: 70% | Visual: 100% | Audio: 0%
🚨 VIOLENCE ALERT! Combined threat: 70% | Visual: 100% | Audio: 0%

🤝 Contributing

Feel free to fork, improve, add new techniques, or even plug in your own fight-detection model.
If you build something cool, tag me — I'd love to see it spin into new directions.

📜 License

Open for personal and research use.
If you use it commercially, modify responsibly.
