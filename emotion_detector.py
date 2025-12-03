import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the emotion detection model"""
        try:
            # Try to load the downloaded model first
            if os.path.exists("models/emotion_model.h5"):
                self.model = load_model("models/emotion_model.h5")
                print("Loaded emotion detection model")
            elif os.path.exists("models/simple_emotion_model.h5"):
                self.model = load_model("models/simple_emotion_model.h5")
                print("Loaded simple emotion detection model")
            else:
                print("No model found, creating a basic one...")
                self.create_basic_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_basic_model()
    
    def create_basic_model(self):
        """Create a basic emotion detection model"""
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Initialize with random weights (for demo purposes)
        self.model = model
        print("Created basic emotion detection model")
    
    def detect_emotion(self, frame):
        """Detect emotions in a frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            emotions_data = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.reshape(1, 48, 48, 1)
                face_roi = face_roi.astype('float32') / 255.0
                
                # Predict emotion
                if self.model is not None:
                    predictions = self.model.predict(face_roi, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    emotion = self.emotion_labels[emotion_idx]
                    confidence = float(predictions[0][emotion_idx])
                else:
                    # Fallback to basic emotion detection
                    emotion, confidence = self.basic_emotion_detection(face_roi)
                
                emotions_data.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw emotion text
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            return frame, emotions_data
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return frame, []
    
    def basic_emotion_detection(self, face_roi):
        """Basic emotion detection using simple heuristics"""
        # This is a simplified version for demo purposes
        # In a real implementation, you'd use a trained model
        
        # Calculate some basic features
        mean_intensity = np.mean(face_roi)
        std_intensity = np.std(face_roi)
        
        # Simple heuristic-based emotion detection
        if mean_intensity < 0.3:
            return "Sad", 0.7
        elif mean_intensity > 0.7:
            return "Happy", 0.7
        elif std_intensity > 0.2:
            return "Surprise", 0.6
        else:
            return "Neutral", 0.8
    
    def get_engagement_score(self, emotions_data):
        """Calculate engagement score based on emotions"""
        if not emotions_data:
            return 0
        
        # Weight different emotions for engagement
        emotion_weights = {
            'Happy': 1.0,
            'Surprise': 0.8,
            'Neutral': 0.5,
            'Sad': 0.2,
            'Angry': 0.1,
            'Fear': 0.1,
            'Disgust': 0.1
        }
        
        total_score = 0
        total_confidence = 0
        
        for emotion_data in emotions_data:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            weight = emotion_weights.get(emotion, 0.5)
            total_score += weight * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            return total_score / total_confidence
        return 0

# Test the emotion detector
if __name__ == "__main__":
    detector = EmotionDetector()
    print("Emotion detector initialized successfully!")
