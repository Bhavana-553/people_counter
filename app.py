from flask import Flask, render_template, request, redirect, session, jsonify, Response
import cv2
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'secretkey'  # Use a secure key in production

# ✅ Dummy login credentials
users = {"bhavana": "#bhav"}

# Global variables
people_count = 0
tracked_ids = set()
last_seen = defaultdict(int)
frame_counter = 0
max_missing_frames = 10  # Number of frames before removing an ID

# ✅ Load YOLOv8 model (make sure yolov8n.pt is in the correct path)
model = YOLO('yolov8n.pt')


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        session['user'] = username
        return redirect('/dashboard')
    return "Login Failed. <a href='/'>Try again</a>"


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html', username=session['user'])


@app.route('/count')
def count():
    global people_count
    return jsonify({"count": people_count})


def generate_frames():
    global people_count, tracked_ids, last_seen, frame_counter

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return

    tracked_ids.clear()
    last_seen.clear()
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        frame_counter += 1
        frame = cv2.resize(frame, (640, 480))

        results = model.track(frame, persist=True, classes=[0], conf=0.4, iou=0.5, verbose=False)
        boxes = results[0].boxes

        current_ids = set()

        if boxes.id is not None and len(boxes.id) > 0:
            ids = boxes.id.cpu().numpy().astype(int)
            current_ids.update(ids)
            for pid in ids:
                last_seen[pid] = frame_counter  # Update last seen

        # Remove IDs that haven't been seen recently
        gone_ids = {pid for pid in tracked_ids if frame_counter - last_seen[pid] > max_missing_frames}
        tracked_ids.difference_update(gone_ids)

        # Add current IDs
        tracked_ids.update(current_ids)

        people_count = len(tracked_ids)

        # Draw annotations
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"People Count: {people_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
