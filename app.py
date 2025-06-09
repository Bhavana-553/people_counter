import os
from flask import Flask, render_template, request, redirect, session, jsonify, Response, send_from_directory
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)
app.secret_key = 'secretkey'  # Use a secure key in production

# ✅ Dummy login credentials
users = {"bhavana": "#bhav"}

# Global variables
people_count = 0
tracked_ids = set()

# ✅ Load YOLOv8 model (make sure yolov8n.pt is in the project folder or accessible path)
model = YOLO('yolov8n.pt')

@app.route('/')
def serve_index():
    return send_from_directory('people_counter', 'index.html')  # Serving frontend directly

@app.route('/index', methods=['POST'])
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
    global people_count, tracked_ids

    cap = cv2.VideoCapture(r"C:\Users\bhavh\OneDrive\Documents\people_-counter\mall_video.mp4")


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

        # Run detection
        results = model.track(source=frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for pid in ids:
                tracked_ids.add(pid)
            people_count = len(tracked_ids)
        else:
            people_count = 0
            tracked_ids.clear()

        # Annotate and encode frame
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
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
