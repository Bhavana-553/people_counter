import os
from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model (make sure yolov8n.pt is in your project folder or accessible)
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return "✅ Flask is running. Go to /video_feed to see the people counter stream."

def generate_frames():
    cap = cv2.VideoCapture("mall_video.mp4")  # Use your video file path here

    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        # Run detection on current frame
        results = model(frame)

        # Annotate frame with detection boxes
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or fallback to 10000
    app.run(host="0.0.0.0", port=port)
