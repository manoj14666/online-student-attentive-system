import cv2
import numpy as np
import os
import random

class SimpleEmotionDetector:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        cascades = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_profileface.xml'
        ]
        self.face_cascades = []
        for c in cascades:
            path = cv2.data.haarcascades + c
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                print(f"Loaded cascade: {c}")
                self.face_cascades.append(clf)
            else:
                print(f"Warning: Could not load cascade {c}")
        if not self.face_cascades:
            print("Error: No face cascades loaded; face detection will fail")
        self.emotion_history = []
    
    def detect_emotion(self, frame):
        """Detect emotions in a frame using simple computer vision techniques"""
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame received")
                return frame, []

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image contrast
            gray = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            print(f"Image shape after preprocessing: {gray.shape}")
            
            # Optionally upscale small frames to aid detection
            scale_up = 1.0
            h0, w0 = gray.shape[:2]
            if max(w0, h0) < 500:
                scale_up = 2.0
                gray = cv2.resize(gray, (int(w0 * scale_up), int(h0 * scale_up)))
                print(f"Upscaled frame for detection: {gray.shape}")

            # Try different detection parameters
            detection_params = [
                {'scale': 1.05, 'neighbors': 3, 'size': (30, 30)},
                {'scale': 1.1, 'neighbors': 4, 'size': (40, 40)},
                {'scale': 1.2, 'neighbors': 5, 'size': (50, 50)}
            ]
            
            faces = []
            for clf in self.face_cascades:
                for params in detection_params:
                    current_faces = clf.detectMultiScale(
                        gray,
                        scaleFactor=params['scale'],
                        minNeighbors=params['neighbors'],
                        minSize=params['size'],
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    print(f"Cascade try -> faces: {len(current_faces)} (scale={params['scale']}, neighbors={params['neighbors']})")
                    if len(current_faces) > 0:
                        faces = current_faces
                        break
                if len(faces) > 0:
                    break

            # Map faces back to original scale if upscaled
            if scale_up != 1.0 and len(faces) > 0:
                mapped = []
                for (x, y, w, h) in faces:
                    mapped.append((int(x/scale_up), int(y/scale_up), int(w/scale_up), int(h/scale_up)))
                faces = mapped
            
            emotions_data = []
            
            for (x, y, w, h) in faces:
                # Add padding to face region
                padding = int(0.1 * w)  # 10% padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Simple emotion detection based on facial features
                emotion, confidence = self.simple_emotion_detection(face_roi)
                
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
    
    def simple_emotion_detection(self, face_roi):
        """Simple emotion detection using basic computer vision"""
        try:
            if face_roi is None or face_roi.size == 0:
                return "Unknown", 0.0

            # Resize face to standard size
            face_resized = cv2.resize(face_roi, (48, 48))
            
            # Apply histogram equalization for better feature detection
            face_resized = cv2.equalizeHist(face_resized)
            
            # Calculate basic features
            mean_intensity = np.mean(face_resized)
            std_intensity = np.std(face_resized)
            
            # Improved edge detection
            edges = cv2.Canny(face_resized, 30, 150)
            edge_density = np.sum(edges > 0) / (48 * 48)
            
            # Calculate facial regions
            eye_region = face_resized[10:24, :]
            mouth_region = face_resized[30:42, :]
            forehead_region = face_resized[0:10, :]
            
            # Calculate regional features
            eye_mean = np.mean(eye_region)
            mouth_mean = np.mean(mouth_region)
            forehead_mean = np.mean(forehead_region)
            
            # Calculate vertical symmetry
            left_half = face_resized[:, :24]
            right_half = face_resized[:, 24:]
            symmetry_score = 1 - (np.mean(np.abs(left_half - np.fliplr(right_half))) / 255)
            
            # Enhanced heuristic-based emotion detection with better feature analysis
            # Normalize features to 0-1 range for consistent scoring
            normalized_std = min(std_intensity / 50.0, 1.0)  # Cap at 50 for std
            normalized_edge = min(edge_density * 10, 1.0)  # Normalize edge density
            
            # Calculate mouth curvature (smile indicator)
            mouth_diff = (mouth_mean - mean_intensity) / 255.0  # Positive = brighter mouth
            eye_diff = (eye_mean - mean_intensity) / 255.0
            forehead_diff = (forehead_mean - mean_intensity) / 255.0
            
            # Calculate emotion scores with improved feature analysis and better thresholds
            # Analyze mouth curvature (smile vs frown)
            mouth_curvature = (mouth_mean - mean_intensity) / 255.0
            
            # Analyze eyebrow region (forehead) for anger/frown
            eyebrow_tension = abs(forehead_diff)
            
            # Analyze eye region for surprise/sadness
            eye_aperture = abs(eye_diff)
            
            # Calculate emotion scores with more balanced and accurate thresholds
            emotion_scores = {
                'Happy': max(0, (mouth_curvature * 3.0 + normalized_edge * 0.8 + symmetry_score * 0.4)) 
                         if (mouth_curvature > 0.1 and normalized_edge > 0.12 and symmetry_score > 0.45 and eye_aperture < 0.1) else 0,
                
                'Sad': max(0, (abs(mouth_curvature) * 2.0 + (1 - symmetry_score) * 0.5 + eye_aperture * 0.6)) 
                        if (mouth_curvature < -0.06 and (eye_diff < 0 or eyebrow_tension > 0.05)) else 0,
                
                'Surprise': max(0, (normalized_std * 1.0 + eye_aperture * 0.8 + normalized_edge * 0.4)) 
                           if (normalized_std > 0.28 and eye_aperture > 0.08 and mouth_curvature > -0.05) else 0,
                
                'Neutral': max(0, (symmetry_score * 0.8 + (1 - normalized_std * 2) * 0.4 + (1 - abs(mouth_curvature) * 5) * 0.3)) 
                          if (abs(mouth_curvature) < 0.05 and normalized_std < 0.22 and symmetry_score > 0.4) else 0,
                
                'Angry': max(0, ((1 - symmetry_score) * 0.8 + eyebrow_tension * 0.6 + abs(mouth_curvature) * 0.4)) 
                         if (forehead_diff < -0.06 and mouth_curvature < 0 and symmetry_score < 0.7) else 0,
                
                'Fear': max(0, (normalized_std * 0.6 + eye_aperture * 0.5 + abs(mouth_curvature) * 0.3)) 
                        if (normalized_std > 0.22 and eye_aperture > 0.06 and mouth_curvature < -0.03) else 0,
                
                'Disgust': max(0, ((1 - symmetry_score) * 0.6 + abs(mouth_curvature) * 0.5 + eyebrow_tension * 0.3)) 
                          if (mouth_curvature < -0.04 and forehead_diff < -0.03 and symmetry_score < 0.75) else 0
            }
            
            # Get the emotion with highest score
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion, score = max_emotion
            
            # Calculate confidence based on the difference between highest and second highest score
            sorted_scores = sorted(emotion_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > 0:
                score_diff = sorted_scores[0] - sorted_scores[1]
                confidence = 0.5 + min(score_diff * 2, 0.5)  # Confidence based on score separation
            else:
                confidence = 0.6  # Default confidence if scores are too close
            
            # Normalize confidence to 0.5-0.9 range (more conservative)
            confidence = min(0.9, max(0.5, confidence))
            
            # If all scores are very low or too close, return neutral with moderate confidence
            if score < 0.18 or (len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.05):
                return "Neutral", 0.65
                
            return emotion, confidence
                
        except Exception as e:
            print(f"Error in simple emotion detection: {e}")
            return "Neutral", 0.5
    
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
    detector = SimpleEmotionDetector()
    print("Simple emotion detector initialized successfully!")
