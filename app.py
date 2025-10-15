from flask import Flask, render_template, Response, jsonify, request
import cv2
from deepface import DeepFace
import json
from datetime import datetime

app = Flask(__name__)

# Define emotion opposites
EMOTION_OPPOSITES = {
    'happy': 'sad',
    'sad': 'happy',
    'angry': 'calm',
    'disgust': 'delight',
    'fear': 'courage',
    'surprise': 'expectation',
    'neutral': 'excitement'
}

def get_opposite_emotion(emotion):
    return EMOTION_OPPOSITES.get(emotion.lower(), 'neutral')

# Global variable to store the latest emotion data
latest_emotion_data = {
    'emotion': '',
    'opposite': '',
    'timestamp': None
}

def gen_frames():
    global latest_emotion_data
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            # Analyze face for emotions
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if result and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                opposite = get_opposite_emotion(emotion)
                
                # Update the latest emotion data
                latest_emotion_data = {
                    'emotion': emotion,
                    'opposite': opposite,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Draw rectangle and text
                x, y, w, h = result[0]['region']['x'], result[0]['region']['y'], result[0]['region']['w'], result[0]['region']['h']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"You: {emotion}", (x, y-40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Try: {opposite}", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify(latest_emotion_data)

if __name__ == '__main__':
    app.run(debug=True)