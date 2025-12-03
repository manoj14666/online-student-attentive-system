import cv2
import numpy as np
import os
import random
from datetime import datetime, timedelta
from collections import deque
import math

class EnhancedEmotionDetector:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Attention monitoring parameters
        self.attention_history = deque(maxlen=300)  # 5 minutes at 1 FPS
        self.emotion_history = deque(maxlen=300)
        self.face_presence_history = deque(maxlen=300)
        self.head_pose_history = deque(maxlen=60)  # 1 minute history for head pose
        
        # Attention thresholds
        self.NEUTRAL_DURATION_THRESHOLD = 300  # 5 minutes in seconds
        self.FACE_ABSENCE_THRESHOLD = 30  # 30 seconds
        self.BLINK_THRESHOLD = 0.3  # Eye aspect ratio threshold for blink detection
        self.HEAD_TURN_THRESHOLD = 30  # degrees
        
        # Initialize face landmarks detector (using simple approach)
        self.face_landmarks_detector = None
        self._initialize_landmarks_detector()
    
    def _initialize_landmarks_detector(self):
        """Initialize face landmarks detector using OpenCV's built-in methods"""
        try:
            # Try to load dlib's face landmark predictor if available
            import dlib
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.face_landmarks_detector = dlib.shape_predictor(predictor_path)
                print("Dlib face landmarks detector loaded successfully")
            else:
                print("Dlib landmarks not available, using simple eye detection")
        except ImportError:
            print("Dlib not available, using simple eye detection")
    
    def detect_emotion_and_attention(self, frame):
        """Enhanced emotion detection with attention monitoring"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            emotions_data = []
            attention_data = {
                'face_detected': len(faces) > 0,
                'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
                'eye_gaze': {'left_eye_open': True, 'right_eye_open': True, 'gaze_direction': 'center'},
                'blink_count': 0,
                'attention_score': 0,
                'status': 'Unknown'
            }
            
            if len(faces) > 0:
                # Process the largest face (assuming main subject)
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence = self.simple_emotion_detection(face_roi)
                
                # Detect attention features
                attention_features = self.detect_attention_features(frame, largest_face, gray)
                attention_data.update(attention_features)
                
                emotions_data.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'attention_data': attention_data
                })
                
                # Draw visualizations
                self.draw_attention_visualizations(frame, largest_face, attention_data)
                
                # Draw emotion text
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw attention score
                cv2.putText(frame, f"Attention: {attention_data['attention_score']:.1f}%", 
                           (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw status
                status_color = self.get_status_color(attention_data['status'])
                cv2.putText(frame, f"Status: {attention_data['status']}", 
                           (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            else:
                # No face detected
                attention_data['face_detected'] = False
                attention_data['status'] = 'No Face Detected'
                attention_data['attention_score'] = 0
            
            # Update history
            self.update_attention_history(attention_data, emotions_data)
            
            # Calculate overall attention score
            attention_data['attention_score'] = self.calculate_attention_score()
            attention_data['status'] = self.determine_attention_status()
            
            return frame, emotions_data, attention_data
            
        except Exception as e:
            print(f"Error in enhanced emotion detection: {e}")
            return frame, [], {'face_detected': False, 'attention_score': 0, 'status': 'Error'}
    
    def detect_attention_features(self, frame, face_bbox, gray):
        """Detect various attention-related features"""
        x, y, w, h = face_bbox
        face_roi = gray[y:y+h, x:x+w]
        
        attention_data = {
            'head_pose': self.estimate_head_pose(face_roi),
            'eye_gaze': self.detect_eye_gaze(face_roi),
            'blink_count': self.detect_blinks(face_roi),
            'face_quality': self.assess_face_quality(face_roi)
        }
        
        return attention_data
    
    def estimate_head_pose(self, face_roi):
        """Estimate head pose using simple computer vision techniques"""
        try:
            # Simple head pose estimation based on face symmetry and features
            h, w = face_roi.shape
            
            # Calculate face center
            center_x = w // 2
            center_y = h // 2
            
            # Detect eyes for pose estimation
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Calculate eye centers
                left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate eye line angle
                eye_line_angle = math.atan2(right_eye_center[1] - left_eye_center[1], 
                                          right_eye_center[0] - left_eye_center[0])
                
                # Convert to degrees
                roll = math.degrees(eye_line_angle)
                
                # Estimate yaw based on eye positions relative to face center
                eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
                face_center_offset = eye_center_x - center_x
                yaw = (face_center_offset / center_x) * 30  # Rough estimation
                
                # Estimate pitch based on eye position relative to face center
                eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
                vertical_offset = eye_center_y - center_y
                pitch = (vertical_offset / center_y) * 20  # Rough estimation
                
                return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
            
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
            
        except Exception as e:
            print(f"Error in head pose estimation: {e}")
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
    
    def detect_eye_gaze(self, face_roi):
        """Detect eye gaze direction and eye openness"""
        try:
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Calculate eye aspect ratio for each eye
                left_ear = self.calculate_eye_aspect_ratio(left_eye, face_roi)
                right_ear = self.calculate_eye_aspect_ratio(right_eye, face_roi)
                
                # Determine if eyes are open
                left_eye_open = left_ear > self.BLINK_THRESHOLD
                right_eye_open = right_ear > self.BLINK_THRESHOLD
                
                # Simple gaze direction estimation
                gaze_direction = self.estimate_gaze_direction(left_eye, right_eye, face_roi)
                
                return {
                    'left_eye_open': left_eye_open,
                    'right_eye_open': right_eye_open,
                    'gaze_direction': gaze_direction,
                    'left_ear': left_ear,
                    'right_ear': right_ear
                }
            
            return {
                'left_eye_open': True,
                'right_eye_open': True,
                'gaze_direction': 'center',
                'left_ear': 0.5,
                'right_ear': 0.5
            }
            
        except Exception as e:
            print(f"Error in eye gaze detection: {e}")
            return {
                'left_eye_open': True,
                'right_eye_open': True,
                'gaze_direction': 'center',
                'left_ear': 0.5,
                'right_ear': 0.5
            }
    
    def calculate_eye_aspect_ratio(self, eye_bbox, face_roi):
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            x, y, w, h = eye_bbox
            eye_region = face_roi[y:y+h, x:x+w]
            
            # Calculate vertical and horizontal measurements
            vertical_1 = eye_region[h//4, w//4]
            vertical_2 = eye_region[3*h//4, w//4]
            vertical_3 = eye_region[h//4, 3*w//4]
            vertical_4 = eye_region[3*h//4, 3*w//4]
            
            horizontal_1 = eye_region[h//2, 0]
            horizontal_2 = eye_region[h//2, w-1]
            
            # Calculate EAR
            vertical_mean = (vertical_1 + vertical_2 + vertical_3 + vertical_4) / 4
            horizontal_mean = (horizontal_1 + horizontal_2) / 2
            
            if horizontal_mean > 0:
                ear = vertical_mean / horizontal_mean
            else:
                ear = 0.5
            
            return ear
            
        except Exception as e:
            return 0.5
    
    def estimate_gaze_direction(self, left_eye, right_eye, face_roi):
        """Estimate gaze direction based on eye positions"""
        try:
            h, w = face_roi.shape
            face_center_x = w // 2
            
            # Calculate eye centers
            left_eye_center = left_eye[0] + left_eye[2] // 2
            right_eye_center = right_eye[0] + right_eye[2] // 2
            eye_center_x = (left_eye_center + right_eye_center) // 2
            
            # Determine gaze direction
            offset = eye_center_x - face_center_x
            threshold = w * 0.1  # 10% of face width
            
            if offset > threshold:
                return 'right'
            elif offset < -threshold:
                return 'left'
            else:
                return 'center'
                
        except Exception as e:
            return 'center'
    
    def detect_blinks(self, face_roi):
        """Detect blinks using eye aspect ratio"""
        try:
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            if len(eyes) >= 2:
                # Calculate average EAR for both eyes
                total_ear = 0
                for eye in eyes:
                    ear = self.calculate_eye_aspect_ratio(eye, face_roi)
                    total_ear += ear
                
                avg_ear = total_ear / len(eyes)
                
                # Count blinks based on EAR threshold
                if avg_ear < self.BLINK_THRESHOLD:
                    return 1
                else:
                    return 0
            
            return 0
            
        except Exception as e:
            return 0
    
    def assess_face_quality(self, face_roi):
        """Assess the quality of face detection"""
        try:
            # Calculate image quality metrics
            mean_intensity = np.mean(face_roi)
            std_intensity = np.std(face_roi)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            
            # Calculate brightness and contrast
            brightness = mean_intensity / 255.0
            contrast = std_intensity / 255.0
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': laplacian_var,
                'quality_score': min(1.0, (brightness * 0.3 + contrast * 0.3 + min(1.0, laplacian_var/1000) * 0.4))
            }
            
        except Exception as e:
            return {
                'brightness': 0.5,
                'contrast': 0.5,
                'sharpness': 0,
                'quality_score': 0.5
            }
    
    def update_attention_history(self, attention_data, emotions_data):
        """Update attention monitoring history"""
        current_time = datetime.now()
        
        # Update face presence history
        self.face_presence_history.append({
            'timestamp': current_time,
            'face_detected': attention_data['face_detected']
        })
        
        # Update emotion history
        if emotions_data:
            emotion = emotions_data[0]['emotion']
            self.emotion_history.append({
                'timestamp': current_time,
                'emotion': emotion
            })
        
        # Update head pose history
        if attention_data.get('head_pose'):
            self.head_pose_history.append({
                'timestamp': current_time,
                'head_pose': attention_data['head_pose']
            })
        
        # Update overall attention history
        self.attention_history.append({
            'timestamp': current_time,
            'attention_data': attention_data,
            'emotions_data': emotions_data
        })
    
    def calculate_attention_score(self):
        """Calculate overall attention score based on multiple factors"""
        try:
            if not self.attention_history:
                return 0
            
            # Base score from face detection
            recent_faces = list(self.face_presence_history)[-30:]  # Last 30 seconds
            face_detection_rate = sum(1 for f in recent_faces if f['face_detected']) / len(recent_faces) if recent_faces else 0
            
            # Score from head pose (looking towards screen)
            head_pose_score = 1.0
            if self.head_pose_history:
                recent_poses = list(self.head_pose_history)[-10:]  # Last 10 seconds
                for pose_data in recent_poses:
                    pose = pose_data['head_pose']
                    # Penalize large head turns
                    if abs(pose['yaw']) > self.HEAD_TURN_THRESHOLD:
                        head_pose_score -= 0.2
                    if abs(pose['pitch']) > self.HEAD_TURN_THRESHOLD:
                        head_pose_score -= 0.1
            
            # Score from eye gaze
            eye_gaze_score = 1.0
            recent_attention = list(self.attention_history)[-10:]  # Last 10 seconds
            for att_data in recent_attention:
                if att_data['attention_data'].get('eye_gaze'):
                    gaze = att_data['attention_data']['eye_gaze']
                    if not gaze['left_eye_open'] or not gaze['right_eye_open']:
                        eye_gaze_score -= 0.3
                    if gaze['gaze_direction'] != 'center':
                        eye_gaze_score -= 0.2
            
            # Score from emotion engagement
            emotion_score = 0.5  # Default neutral
            if self.emotion_history:
                recent_emotions = list(self.emotion_history)[-30:]  # Last 30 seconds
                emotion_counts = {}
                for emo_data in recent_emotions:
                    emotion = emo_data['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Weight emotions for engagement
                emotion_weights = {
                    'Happy': 1.0,
                    'Surprise': 0.8,
                    'Neutral': 0.5,
                    'Sad': 0.2,
                    'Angry': 0.1,
                    'Fear': 0.1,
                    'Disgust': 0.1
                }
                
                total_weight = 0
                weighted_sum = 0
                for emotion, count in emotion_counts.items():
                    weight = emotion_weights.get(emotion, 0.5)
                    weighted_sum += weight * count
                    total_weight += count
                
                if total_weight > 0:
                    emotion_score = weighted_sum / total_weight
            
            # Calculate final attention score
            attention_score = (
                face_detection_rate * 0.3 +
                max(0, head_pose_score) * 0.25 +
                max(0, eye_gaze_score) * 0.25 +
                emotion_score * 0.2
            ) * 100
            
            return min(100, max(0, attention_score))
            
        except Exception as e:
            print(f"Error calculating attention score: {e}")
            return 0
    
    def determine_attention_status(self):
        """Determine attention status based on various factors"""
        try:
            attention_score = self.calculate_attention_score()
            
            # Check for face absence
            if self.face_presence_history:
                recent_faces = list(self.face_presence_history)[-30:]  # Last 30 seconds
                face_detection_rate = sum(1 for f in recent_faces if f['face_detected']) / len(recent_faces)
                
                if face_detection_rate < 0.1:  # Less than 10% face detection
                    return "Absent / Disengaged"
            
            # Check for prolonged neutral emotion
            if self.emotion_history:
                recent_emotions = list(self.emotion_history)[-300:]  # Last 5 minutes
                neutral_count = sum(1 for e in recent_emotions if e['emotion'] == 'Neutral')
                
                if len(recent_emotions) > 0 and neutral_count / len(recent_emotions) > 0.8:
                    return "Low Engagement"
            
            # Determine status based on attention score
            if attention_score >= 80:
                return "Attentive"
            elif attention_score >= 50:
                return "Partially Attentive"
            elif attention_score >= 20:
                return "Distracted"
            else:
                return "Inattentive"
                
        except Exception as e:
            print(f"Error determining attention status: {e}")
            return "Unknown"
    
    def get_status_color(self, status):
        """Get color for status display"""
        color_map = {
            'Attentive': (0, 255, 0),  # Green
            'Partially Attentive': (0, 255, 255),  # Yellow
            'Distracted': (0, 165, 255),  # Orange
            'Inattentive': (0, 0, 255),  # Red
            'Absent / Disengaged': (128, 0, 128),  # Purple
            'Low Engagement': (255, 0, 255),  # Magenta
            'No Face Detected': (128, 128, 128),  # Gray
            'Error': (255, 255, 255)  # White
        }
        return color_map.get(status, (255, 255, 255))
    
    def draw_attention_visualizations(self, frame, face_bbox, attention_data):
        """Draw attention-related visualizations on the frame"""
        try:
            x, y, w, h = face_bbox
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw head pose indicators
            if attention_data.get('head_pose'):
                pose = attention_data['head_pose']
                
                # Draw pose text
                pose_text = f"Pitch: {pose['pitch']:.1f}° Yaw: {pose['yaw']:.1f}° Roll: {pose['roll']:.1f}°"
                cv2.putText(frame, pose_text, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw eye gaze indicators
            if attention_data.get('eye_gaze'):
                gaze = attention_data['eye_gaze']
                
                # Draw eye status
                eye_text = f"Eyes: {'Open' if gaze['left_eye_open'] and gaze['right_eye_open'] else 'Closed'}"
                cv2.putText(frame, eye_text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw gaze direction
                gaze_text = f"Gaze: {gaze['gaze_direction']}"
                cv2.putText(frame, gaze_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw face quality indicator
            if attention_data.get('face_quality'):
                quality = attention_data['face_quality']
                quality_text = f"Quality: {quality['quality_score']:.2f}"
                cv2.putText(frame, quality_text, (x+w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        except Exception as e:
            print(f"Error drawing attention visualizations: {e}")
    
    def simple_emotion_detection(self, face_roi):
        """Simple emotion detection using basic computer vision (from original)"""
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_roi, (48, 48))
            
            # Calculate basic features
            mean_intensity = np.mean(face_resized)
            std_intensity = np.std(face_resized)
            
            # Calculate edge density (smile detection)
            edges = cv2.Canny(face_resized, 50, 150)
            edge_density = np.sum(edges > 0) / (48 * 48)
            
            # Calculate mouth region (lower half of face)
            mouth_region = face_resized[24:48, :]
            mouth_mean = np.mean(mouth_region)
            
            # Simple heuristic-based emotion detection
            if edge_density > 0.1 and mouth_mean > mean_intensity + 10:
                return "Happy", 0.8
            elif mean_intensity < 80:
                return "Sad", 0.7
            elif std_intensity > 30:
                return "Surprise", 0.6
            elif edge_density < 0.05:
                return "Neutral", 0.8
            elif mouth_mean < mean_intensity - 10:
                return "Angry", 0.6
            else:
                # Add some randomness for demo purposes
                emotions = ["Happy", "Neutral", "Surprise"]
                weights = [0.4, 0.4, 0.2]
                emotion = np.random.choice(emotions, p=weights)
                confidence = random.uniform(0.6, 0.9)
                return emotion, confidence
                
        except Exception as e:
            print(f"Error in simple emotion detection: {e}")
            return "Neutral", 0.5
    
    def get_engagement_score(self, emotions_data):
        """Calculate engagement score based on emotions (from original)"""
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
    
    def get_attention_summary(self):
        """Get a summary of attention monitoring data"""
        try:
            current_score = self.calculate_attention_score()
            current_status = self.determine_attention_status()
            
            # Calculate statistics
            total_records = len(self.attention_history)
            face_detection_rate = 0
            if self.face_presence_history:
                face_detection_rate = sum(1 for f in self.face_presence_history if f['face_detected']) / len(self.face_presence_history)
            
            # Emotion distribution
            emotion_distribution = {}
            if self.emotion_history:
                for emo_data in self.emotion_history:
                    emotion = emo_data['emotion']
                    emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            return {
                'current_attention_score': current_score,
                'current_status': current_status,
                'face_detection_rate': face_detection_rate,
                'total_records': total_records,
                'emotion_distribution': emotion_distribution,
                'monitoring_duration_minutes': len(self.attention_history) / 60 if self.attention_history else 0
            }
            
        except Exception as e:
            print(f"Error getting attention summary: {e}")
            return {
                'current_attention_score': 0,
                'current_status': 'Error',
                'face_detection_rate': 0,
                'total_records': 0,
                'emotion_distribution': {},
                'monitoring_duration_minutes': 0
            }

# Test the enhanced emotion detector
if __name__ == "__main__":
    detector = EnhancedEmotionDetector()
    print("Enhanced emotion detector with attention monitoring initialized successfully!")
