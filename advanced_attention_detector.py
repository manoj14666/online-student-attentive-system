import cv2
import numpy as np
from collections import deque
import math
from simple_emotion_detector import SimpleEmotionDetector

class AdvancedAttentionDetector(SimpleEmotionDetector):
    def __init__(self):
        super().__init__()
        
        # Load eye cascade for blink detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # History for temporal analysis
        self.blink_history = deque(maxlen=30)  # ~1 second at 30fps
        self.eye_aspect_ratio_history = deque(maxlen=30)
        self.head_pose_history = deque(maxlen=60)  # ~2 seconds
        self.mouth_history = deque(maxlen=30)  # For yawn detection
        self.emotion_history = deque(maxlen=90)  # ~3 seconds
        
        # Eye Aspect Ratio (EAR) constants for blink detection
        self.EAR_THRESHOLD = 0.25  # Below this = eye closed
        self.EAR_CONSEC_FRAMES = 3  # Frames for blink confirmation
        self.BLINK_RATE_THRESHOLD = 0.5  # Blinks per second (excessive if > 0.5)
        
        # Yawn detection constants
        self.YAWN_THRESHOLD = 0.6  # Mouth aspect ratio threshold
        self.YAWN_CONSEC_FRAMES = 10  # Frames mouth open for yawn
        
        # Head pose thresholds (degrees)
        self.HEAD_YAW_THRESHOLD = 25  # Looking left/right
        self.HEAD_PITCH_DOWN_THRESHOLD = 15  # Looking down
        self.HEAD_PITCH_UP_THRESHOLD = -10  # Looking up
        
        # Eye gaze constants
        self.GAZE_CENTER_THRESHOLD = 0.3  # Eye position relative to face center
        
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if len(eye_points) < 6:
            return 0.3  # Default if not enough points
        
        # Vertical distances
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if horizontal == 0:
            return 0.3
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def detect_eyes_detailed(self, face_roi, gray_frame):
        """Detect eyes with detailed information for gaze and blink"""
        try:
            h, w = face_roi.shape[:2] if len(face_roi.shape) == 2 else face_roi.shape[:2]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, 
                scaleFactor=1.1, 
                minNeighbors=3,
                minSize=(int(w*0.15), int(h*0.12))
            )
            
            if len(eyes) < 2:
                # Try with more lenient parameters
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(int(w*0.1), int(h*0.08))
                )
            
            left_eye = None
            right_eye = None
            left_ear = 0.3
            right_ear = 0.3
            
            if len(eyes) >= 2:
                # Sort by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Calculate EAR for each eye using simple approximation
                # Get eye regions
                lx, ly, lw, lh = left_eye
                rx, ry, rw, rh = right_eye
                
                left_eye_roi = face_roi[ly:ly+lh, lx:lx+lw]
                right_eye_roi = face_roi[ry:ry+rh, rx:rx+rw]
                
                # Simple EAR calculation using eye region height/width ratio
                if lw > 0 and lh > 0:
                    left_ear = min(lh / (lw * 2.0), 0.4)  # Normalized
                if rw > 0 and rh > 0:
                    right_ear = min(rh / (rw * 2.0), 0.4)  # Normalized
                
            elif len(eyes) == 1:
                # Only one eye detected - use it
                ex, ey, ew, eh = eyes[0]
                if ex < w / 2:
                    left_eye = eyes[0]
                    left_ear = min(eh / (ew * 2.0), 0.4) if ew > 0 else 0.3
                else:
                    right_eye = eyes[0]
                    right_ear = min(eh / (ew * 2.0), 0.4) if ew > 0 else 0.3
            
            # Determine if eyes are open (more strict threshold)
            # Use lower threshold for detection since we need both eyes
            eye_open_threshold = self.EAR_THRESHOLD * 1.2  # Slightly higher for more accuracy
            left_open = left_ear > eye_open_threshold
            right_open = right_ear > eye_open_threshold
            
            # If eyes detected but EAR is very low, they're likely closed
            if len(eyes) >= 1 and (left_ear < 0.15 or right_ear < 0.15):
                left_open = False
                right_open = False
            
            # Calculate average EAR for blink detection
            avg_ear = (left_ear + right_ear) / 2.0 if len(eyes) >= 2 else max(left_ear, right_ear)
            
            # If no eyes detected but we have a face, likely closed
            if len(eyes) == 0:
                left_open = False
                right_open = False
                avg_ear = 0.15  # Low value indicating closed
            
            # Calculate eye positions relative to face center for gaze
            eye_gaze = 'center'
            if left_eye is not None and right_eye is not None:
                left_center_x = left_eye[0] + left_eye[2] // 2
                right_center_x = right_eye[0] + right_eye[2] // 2
                eye_center_x = (left_center_x + right_center_x) / 2.0
                face_center_x = w / 2.0
                
                offset_ratio = (eye_center_x - face_center_x) / face_center_x
                
                if offset_ratio > self.GAZE_CENTER_THRESHOLD:
                    eye_gaze = 'right'
                elif offset_ratio < -self.GAZE_CENTER_THRESHOLD:
                    eye_gaze = 'left'
                else:
                    eye_gaze = 'center'
            
            return {
                'left_eye': left_eye,
                'right_eye': right_eye,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'left_open': left_open,
                'right_open': right_open,
                'avg_ear': avg_ear,
                'gaze_direction': eye_gaze,
                'eyes_detected': len(eyes)
            }
        except Exception as e:
            print(f"Error in eye detection: {e}")
            # Return closed eyes on error (safer default)
            return {
                'left_eye': None,
                'right_eye': None,
                'left_ear': 0.15,
                'right_ear': 0.15,
                'left_open': False,
                'right_open': False,
                'avg_ear': 0.15,
                'gaze_direction': 'unknown',
                'eyes_detected': 0
            }
    
    def detect_blink(self, eye_data):
        """Detect blinks using EAR history"""
        try:
            avg_ear = eye_data['avg_ear']
            self.eye_aspect_ratio_history.append(avg_ear)
            
            # Check for blink (EAR drops below threshold)
            if len(self.eye_aspect_ratio_history) >= self.EAR_CONSEC_FRAMES:
                recent_ears = list(self.eye_aspect_ratio_history)[-self.EAR_CONSEC_FRAMES:]
                
                # Blink if recent frames are below threshold
                if all(ear < self.EAR_THRESHOLD for ear in recent_ears[-3:]):
                    # Check if it was above threshold before (blink start)
                    if len(recent_ears) > 3 and recent_ears[-4] > self.EAR_THRESHOLD:
                        self.blink_history.append(1)
                        return True
            
            self.blink_history.append(0)
            return False
        except Exception:
            return False
    
    def calculate_blink_rate(self):
        """Calculate blinks per second"""
        if len(self.blink_history) < 10:
            return 0.0
        
        # Count blinks in recent history (assuming ~30fps)
        recent_blinks = sum(list(self.blink_history)[-30:])
        return recent_blinks / 30.0  # Blinks per second
    
    def detect_yawn(self, face_roi):
        """Detect yawning by analyzing mouth region"""
        try:
            h, w = face_roi.shape[:2] if len(face_roi.shape) == 2 else face_roi.shape[:2]
            
            # Mouth region is typically in lower 1/3 of face
            mouth_region = face_roi[int(h*0.6):h, :]
            
            if mouth_region.size == 0:
                return False, 0.0
            
            # Calculate mouth aspect ratio (MAR)
            # Vertical: height of mouth region
            # Horizontal: width of face
            mouth_height = mouth_region.shape[0]
            mouth_width = w
            
            if mouth_width == 0:
                return False, 0.0
            
            mar = mouth_height / mouth_width
            
            self.mouth_history.append(mar)
            
            # Check if mouth has been open for consecutive frames
            if len(self.mouth_history) >= self.YAWN_CONSEC_FRAMES:
                recent_mar = list(self.mouth_history)[-self.YAWN_CONSEC_FRAMES:]
                if all(m > self.YAWN_THRESHOLD for m in recent_mar):
                    return True, mar
            
            return False, mar
        except Exception as e:
            print(f"Yawn detection error: {e}")
            return False, 0.0
    
    def estimate_head_pose_advanced(self, face_roi, eyes_data):
        """Advanced head pose estimation using facial features"""
        try:
            h, w = face_roi.shape[:2] if len(face_roi.shape) == 2 else face_roi.shape[:2]
            
            face_center_x = w / 2.0
            face_center_y = h / 2.0
            
            pitch = 0.0
            yaw = 0.0
            roll = 0.0
            
            # Estimate roll from eye alignment
            if eyes_data['left_eye'] is not None and eyes_data['right_eye'] is not None:
                left_eye = eyes_data['left_eye']
                right_eye = eyes_data['right_eye']
                
                left_eye_y = left_eye[1] + left_eye[3] // 2
                right_eye_y = right_eye[1] + right_eye[3] // 2
                
                # Calculate roll (tilt) from eye alignment
                eye_dy = right_eye_y - left_eye_y
                eye_dx = (right_eye[0] + right_eye[2] // 2) - (left_eye[0] + left_eye[2] // 2)
                
                if eye_dx != 0:
                    roll = math.degrees(math.atan2(eye_dy, eye_dx))
                
                # Estimate yaw from eye positions relative to face center
                eye_center_x = ((left_eye[0] + left_eye[2] // 2) + (right_eye[0] + right_eye[2] // 2)) / 2.0
                eye_offset = (eye_center_x - face_center_x) / face_center_x
                yaw = eye_offset * 45  # Scale to degrees
                
                # Estimate pitch from eye position relative to face center
                eye_center_y = (left_eye_y + right_eye_y) / 2.0
                vertical_offset = (eye_center_y - face_center_y) / face_center_y
                pitch = vertical_offset * 30  # Scale to degrees
            
            # Store in history
            self.head_pose_history.append({'pitch': pitch, 'yaw': yaw, 'roll': roll})
            
            return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    
    def detect_emotion_and_attention(self, frame):
        """Comprehensive emotion and attention detection"""
        try:
            # Detect basic emotions
            processed_frame, emotions_data = self.detect_emotion(frame)
            
            if not emotions_data:
                return processed_frame, emotions_data, {
                    'face_detected': False,
                    'attention_score': 0.0,
                    'status': 'Absent / Disengaged',
                    'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
                    'eye_gaze': {'direction': 'unknown', 'left_open': False, 'right_open': False},
                    'blink_rate': 0.0,
                    'yawn_detected': False,
                    'face_quality': {'score': 0.0}
                }
            
            # Get first face for detailed analysis
            face_bbox = emotions_data[0].get('bbox', (0, 0, 0, 0))
            x, y, w, h = face_bbox
            
            # Extract face region
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes with detailed information
            eyes_data = self.detect_eyes_detailed(face_roi, gray)
            
            # Detect blinks
            blink_detected = self.detect_blink(eyes_data)
            blink_rate = self.calculate_blink_rate()
            
            # Detect yawn
            yawn_detected, yawn_intensity = self.detect_yawn(face_roi)
            
            # Estimate head pose
            head_pose = self.estimate_head_pose_advanced(face_roi, eyes_data)
            self.head_pose_history.append(head_pose)
            
            # Calculate attention score
            attention_score = self.calculate_attention_score(
                emotions_data,
                eyes_data,
                head_pose,
                blink_rate,
                yawn_detected
            )
            
            # Determine attention status
            attention_status = self.determine_attention_status(attention_score, eyes_data, head_pose, yawn_detected)
            
            # Calculate face quality
            face_quality = self.calculate_face_quality(face_roi)
            
            # Store emotion in history
            if emotions_data:
                self.emotion_history.append(emotions_data[0]['emotion'])
            
            attention_data = {
                'face_detected': True,
                'attention_score': attention_score,
                'status': attention_status,
                'head_pose': head_pose,
                'eye_gaze': {
                    'direction': eyes_data['gaze_direction'],
                    'left_open': eyes_data['left_open'],
                    'right_open': eyes_data['right_open'],
                    'left_ear': eyes_data['left_ear'],
                    'right_ear': eyes_data['right_ear']
                },
                'blink_rate': blink_rate,
                'blink_detected': blink_detected,
                'yawn_detected': yawn_detected,
                'yawn_intensity': yawn_intensity,
                'face_quality': face_quality
            }
            
            return processed_frame, emotions_data, attention_data
            
        except Exception as e:
            print(f"Error in comprehensive detection: {e}")
            return frame, [], {
                'face_detected': False,
                'attention_score': 0.0,
                'status': 'Error',
                'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
                'eye_gaze': {'direction': 'unknown', 'left_open': False, 'right_open': False},
                'blink_rate': 0.0,
                'yawn_detected': False,
                'face_quality': {'score': 0.0}
            }
    
    def calculate_attention_score(self, emotions_data, eyes_data, head_pose, blink_rate, yawn_detected):
        """Calculate comprehensive attention score (0-100)"""
        try:
            base_score = 50.0  # Start with neutral
            
            # Face presence (20 points)
            if emotions_data:
                base_score += 20.0
            
            # Eye openness (20 points) - critical check
            both_open = eyes_data['left_open'] and eyes_data['right_open']
            avg_ear = eyes_data.get('avg_ear', 0.3)
            eyes_detected_count = eyes_data.get('eyes_detected', 0)
            
            # More strict eye closure detection
            if both_open and avg_ear > self.EAR_THRESHOLD and eyes_detected_count >= 2:
                base_score += 20.0
            elif (eyes_data['left_open'] or eyes_data['right_open']) and avg_ear > self.EAR_THRESHOLD * 0.8:
                base_score += 5.0  # One eye open
            else:
                # Eyes closed - severe penalty
                base_score -= 50.0
                base_score = max(0, base_score)  # Don't go below 0
            
            # Eye gaze direction (15 points)
            if eyes_data['gaze_direction'] == 'center':
                base_score += 15.0
            elif eyes_data['gaze_direction'] in ['left', 'right']:
                base_score -= 10.0  # Looking away
            
            # Head pose (15 points)
            yaw_abs = abs(head_pose['yaw'])
            pitch = head_pose['pitch']
            
            if yaw_abs < 10 and -5 < pitch < 10:  # Looking forward
                base_score += 15.0
            elif yaw_abs < 20 and -10 < pitch < 15:  # Slightly off
                base_score += 5.0
            elif yaw_abs > 25 or pitch < -15 or pitch > 20:  # Looking away
                base_score -= 20.0
            
            # Blink rate penalty (10 points)
            if blink_rate > self.BLINK_RATE_THRESHOLD:
                base_score -= 15.0  # Excessive blinking
            elif blink_rate > self.BLINK_RATE_THRESHOLD * 0.7:
                base_score -= 5.0  # Slightly excessive
            
            # Yawn penalty (10 points)
            if yawn_detected:
                base_score -= 15.0
            
            # Emotion-based adjustment (10 points)
            if emotions_data:
                emotion = emotions_data[0]['emotion']
                if emotion == 'Happy':
                    base_score += 10.0
                elif emotion == 'Neutral':
                    base_score += 5.0
                elif emotion in ['Sad', 'Angry']:
                    base_score -= 5.0
                elif emotion == 'Surprise':
                    base_score += 3.0
            
            # Normalize to 0-100
            attention_score = max(0.0, min(100.0, base_score))
            
            return attention_score
            
        except Exception as e:
            print(f"Error calculating attention score: {e}")
            return 50.0
    
    def determine_attention_status(self, attention_score, eyes_data, head_pose, yawn_detected):
        """Determine attention status based on multiple factors"""
        try:
            # Check for critical conditions first - eyes closed is highest priority
            # Check both explicit eye_open flags and EAR values
            both_eyes_closed = (not eyes_data['left_open'] and not eyes_data['right_open'])
            # Also check if EAR is very low (eyes likely closed)
            avg_ear = eyes_data.get('avg_ear', 0.3)
            eyes_likely_closed = avg_ear < self.EAR_THRESHOLD
            
            if both_eyes_closed or (eyes_data['eyes_detected'] < 2 and eyes_likely_closed):
                return "Distracted (Eyes Closed)"
            
            if yawn_detected:
                return "Drowsy / Fatigued"
            
            yaw_abs = abs(head_pose['yaw'])
            pitch = head_pose['pitch']
            
            if yaw_abs > 30 or pitch < -20:
                return "Inattentive (Looking Away)"
            
            # Adjust attention score based on eye closure
            if not eyes_data['left_open'] or not eyes_data['right_open']:
                # One eye closed or both - force distracted status
                if attention_score > 30:
                    attention_score = 25  # Force below threshold
            
            if attention_score >= 75:
                return "Attentive"
            elif attention_score >= 50:
                return "Partially Attentive"
            elif attention_score >= 30:
                return "Distracted"
            else:
                return "Inattentive"
                
        except Exception as e:
            print(f"Error determining attention status: {e}")
            return "Unknown"
    
    def calculate_face_quality(self, face_roi):
        """Calculate face quality score"""
        try:
            if face_roi.size == 0:
                return {'score': 0.0}
            
            # Calculate brightness
            mean_brightness = np.mean(face_roi)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Calculate contrast
            std_brightness = np.std(face_roi)
            contrast_score = min(std_brightness / 50.0, 1.0)
            
            # Calculate sharpness (using Laplacian variance)
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            sharpness_score = min(laplacian.var() / 100.0, 1.0)
            
            # Combined quality score
            quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3)
            
            return {
                'score': quality_score,
                'brightness': brightness_score,
                'contrast': contrast_score,
                'sharpness': sharpness_score
            }
        except Exception:
            return {'score': 0.5}

