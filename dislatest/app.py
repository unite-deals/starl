from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file

def measure_distance(frame, known_width, focal_length):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the object of interest)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        _, radius = cv2.minEnclosingCircle(largest_contour)
        object_width = 2 * radius  # Calculate object width from the enclosing circle

        # Calculate the object distance
        distance = (known_width * focal_length) / object_width
        return distance

    return None

def generate_frames():
    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            break

        # Perform object distance measurement
        known_width = 20  # Width of the object in cm
        focal_length = 500  # Focal length in pixels
        distance = measure_distance(frame, known_width, focal_length)

        # Draw the measured distance on the frame
        if distance is not None:
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
