from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

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

def gen_frames():
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

if __name__ == '__main__':
    app.run(debug=True)