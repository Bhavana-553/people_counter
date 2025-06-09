from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Make sure this model file is accessible
people_count = 0
tracked_ids = set()

@app.route('/')
def index():
    return "✅ Flask is running. Go to /video_feed to see the people counter stream."

def generate_frames():
    global people_count, tracked_ids

    cap = cv2.VideoCapture("mall_video.mp4")  # Replace with your relative or absolute path

    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return

    people_count = 0
    tracked_ids.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        results = model.track(source=frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for pid in ids:
                tracked_ids.add(pid)
            people_count = len(tracked_ids)
        else:
            people_count = 0
            tracked_ids.clear()

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
    app.run(host="0.0.0.0", port=10000)