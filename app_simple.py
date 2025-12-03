from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
from datetime import datetime
import json

from database import db, User, Session, EmotionData, Feedback, ClassRoom, AttentionAlert, AttentionSummary
from simple_emotion_detector import SimpleEmotionDetector
from advanced_attention_detector import AdvancedAttentionDetector
from audio_processor import AudioProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize advanced detector and audio processor
try:
    emotion_detector = AdvancedAttentionDetector()
    print("Advanced attention detector initialized")
except Exception as e:
    print(f"Failed to initialize advanced detector: {e}, falling back to simple detector")
    emotion_detector = SimpleEmotionDetector()

audio_processor = AudioProcessor()

# Eye cascade for simple eye-open detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility to convert NumPy types to native Python for JSON
def _to_native_number(value):
    try:
        # Convert numpy types to native python
        if isinstance(value, (np.generic,)):
            return value.item()
    except Exception:
        pass
    return value

def _serialize_emotions(emotions_data):
    serialized = []
    for e in emotions_data:
        x, y, w, h = e.get('bbox', (0, 0, 0, 0))
        serialized.append({
            'emotion': str(e.get('emotion', '')),
            'confidence': float(_to_native_number(e.get('confidence', 0.0))),
            'bbox': [int(_to_native_number(x)), int(_to_native_number(y)), int(_to_native_number(w)), int(_to_native_number(h))]
        })
    return serialized

def check_and_create_attention_alerts(session_id, attention_data):
    """Check attention data and create alerts if necessary"""
    try:
        session = Session.query.get(session_id)
        if not session:
            return
        
        # Check for low attention score
        if attention_data['attention_score'] < 30:
            alert = AttentionAlert(
                session_id=session_id,
                student_id=session.user_id,
                alert_type='low_attention',
                alert_message=f"Student attention score is low: {attention_data['attention_score']:.1f}%",
                attention_score=attention_data['attention_score']
            )
            db.session.add(alert)
        
        # Check for face absence
        if not attention_data['face_detected']:
            alert = AttentionAlert(
                session_id=session_id,
                student_id=session.user_id,
                alert_type='face_absent',
                alert_message="Student face not detected - possible disengagement",
                attention_score=0.0
            )
            db.session.add(alert)
        
        # Check for distracted status
        if attention_data['status'] in ['Distracted', 'Inattentive']:
            alert = AttentionAlert(
                session_id=session_id,
                student_id=session.user_id,
                alert_type='distracted',
                alert_message=f"Student appears {attention_data['status'].lower()}",
                attention_score=attention_data['attention_score']
            )
            db.session.add(alert)
        
        db.session.commit()
        
    except Exception as e:
        print(f"Error creating attention alerts: {e}")
        db.session.rollback()

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_interface'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            else:
                return redirect(url_for('student_interface'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/student')
@login_required
def student_interface():
    if current_user.role != 'student':
        flash('Access denied')
        return redirect(url_for('index'))
    
    try:
        # Get active session or create new one
        active_session = Session.query.filter_by(user_id=current_user.id, is_active=True).first()
        if not active_session:
            active_session = Session(
                user_id=current_user.id,
                session_name=f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            db.session.add(active_session)
            db.session.commit()
            print(f"Created new session with ID: {active_session.id}")  # Debug print
        else:
            print(f"Found existing session with ID: {active_session.id}")  # Debug print
        
        return render_template('student.html', session_id=active_session.id)
        
    except Exception as e:
        print(f"Error in student interface: {e}")
        db.session.rollback()
        flash('Error accessing student interface')
        return redirect(url_for('index'))

@app.route('/teacher')
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher':
        flash('Access denied')
        return redirect(url_for('index'))
    
    # Get all students
    students = User.query.filter_by(role='student').all()
    
    # Get recent sessions
    recent_sessions = Session.query.filter_by(is_active=True).all()
    
    return render_template('teacher.html', students=students, sessions=recent_sessions)

@app.route('/api/emotion_data', methods=['POST'])
@login_required
def process_emotion_data():
    try:
        print("Received emotion data request")  # Debug print
        
        # Check if request has JSON data
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
            
        data = request.json
        if not data:
            print("No JSON data received")
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        image_data = data.get('image')
        session_id = data.get('session_id')
        
        print(f"Session ID received: {session_id}")  # Debug print
        
        if not image_data:
            print("Missing image data")
            return jsonify({'success': False, 'error': 'Missing image data'}), 400
            
        if not session_id:
            print("Missing session ID")
            return jsonify({'success': False, 'error': 'Missing session ID'}), 400
        
        # Decode base64 image
        try:
            # Remove header if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode image")
                return jsonify({'success': False, 'error': 'Failed to decode image'})
                
            # Check if image is empty
            if frame.size == 0:
                print("Empty image received")
                return jsonify({'success': False, 'error': 'Empty image received'})
                
            # Print image shape for debugging
            print(f"Image shape: {frame.shape}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})
            print(f"Image shape: {frame.shape}")
        
        # Detect emotions and attention with advanced features
        try:
            # Use advanced detector if available
            if hasattr(emotion_detector, 'detect_emotion_and_attention'):
                processed_frame, emotions_data, attention_data = emotion_detector.detect_emotion_and_attention(frame)
                
                face_detected = attention_data.get('face_detected', len(emotions_data) > 0)
                attention_score = float(_to_native_number(attention_data.get('attention_score', 0)))
                attention_status = attention_data.get('status', 'Unknown')
                
                # Extract detailed features
                head_pose = attention_data.get('head_pose', {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0})
                eye_gaze = attention_data.get('eye_gaze', {})
                blink_rate = attention_data.get('blink_rate', 0.0)
                blink_detected = attention_data.get('blink_detected', False)
                yawn_detected = attention_data.get('yawn_detected', False)
                face_quality = attention_data.get('face_quality', {'score': 0.8})
                
                # Calculate engagement score
                engagement_score = float(_to_native_number(emotion_detector.get_engagement_score(emotions_data)))
                
            else:
                # Fallback to basic detection
                processed_frame, emotions_data = emotion_detector.detect_emotion(frame)
                
                if not isinstance(emotions_data, list):
                    emotions_data = []
                
                face_detected = len(emotions_data) > 0
                engagement_score = float(_to_native_number(emotion_detector.get_engagement_score(emotions_data)))
                attention_score = engagement_score * 100 if face_detected else 0
                attention_status = "Attentive" if attention_score > 70 else "Partially Attentive" if attention_score > 40 else "Distracted"
                
                # Basic eye detection
                head_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
                eye_gaze = {'direction': 'center', 'left_open': True, 'right_open': True}
                blink_rate = 0.0
                blink_detected = False
                yawn_detected = False
                face_quality = {'score': 0.8}
                
                # Simple eye-open check
                try:
                    if face_detected and len(emotions_data) > 0:
                        x, y, w, h = emotions_data[0].get('bbox', (0, 0, 0, 0))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        roi = gray[y:y+h, x:x+w]
                        if roi.size > 0:
                            eyes = eye_cascade.detectMultiScale(roi, 1.1, 3)
                            eyes_open = len(eyes) >= 2
                            if not eyes_open:
                                attention_status = "Distracted"
                                attention_score = min(attention_score, 10.0)
                            eye_gaze['left_open'] = eyes_open
                            eye_gaze['right_open'] = eyes_open
                except Exception as _e:
                    pass
            
            print(f"Emotions detected: {emotions_data}")
            print(f"Attention score: {attention_score}, Status: {attention_status}")
            print(f"Blink rate: {blink_rate}, Yawn: {yawn_detected}")
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error in emotion detection: {str(e)}',
                'face_detected': False,
                'emotions': [],
                'engagement_score': 0,
                'attention_score': 0,
                'attention_status': 'Error'
            })
        
        # Save emotion data to database with advanced features
        for emotion_data in emotions_data:
            emotion_record = EmotionData(
                session_id=session_id,
                emotion=emotion_data['emotion'],
                confidence=emotion_data['confidence'],
                engagement_score=engagement_score,
                face_detected=face_detected,
                attention_score=attention_score,
                attention_status=attention_status,
                head_pitch=float(_to_native_number(head_pose.get('pitch', 0.0))),
                head_yaw=float(_to_native_number(head_pose.get('yaw', 0.0))),
                head_roll=float(_to_native_number(head_pose.get('roll', 0.0))),
                eye_gaze_direction=eye_gaze.get('direction', 'center'),
                left_eye_open=eye_gaze.get('left_open', True),
                right_eye_open=eye_gaze.get('right_open', True),
                blink_detected=blink_detected,
                face_quality_score=float(_to_native_number(face_quality.get('score', 0.8)))
            )
            db.session.add(emotion_record)
        
        db.session.commit()
        
        # Check for attention alerts
        attention_data = {
            'attention_score': attention_score,
            'status': attention_status,
            'face_detected': face_detected
        }
        check_and_create_attention_alerts(session_id, attention_data)
        
        # Prepare JSON-safe payload
        serialized_emotions = _serialize_emotions(emotions_data)

        # Emit to teacher dashboard with comprehensive attention data
        socketio.emit('emotion_update', {
            'user_id': int(_to_native_number(current_user.id)),
            'username': current_user.username,
            'emotions': serialized_emotions,
            'engagement_score': engagement_score,
            'attention_score': attention_score,
            'attention_status': attention_status,
            'face_detected': face_detected,
            'head_pose': head_pose,
            'eye_gaze': eye_gaze,
            'blink_rate': float(_to_native_number(blink_rate)),
            'yawn_detected': yawn_detected,
            'face_quality': face_quality,
            'timestamp': datetime.now().isoformat()
        }, room='teacher_room')

        # Real-time teacher alert on distraction/inattention/absence
        try:
            if (not face_detected) or (attention_score < 30) or (attention_status in ['Distracted', 'Inattentive', 'Absent / Disengaged', 'Low Engagement']):
                socketio.emit('student_attention_alert', {
                    'user_id': int(_to_native_number(current_user.id)),
                    'username': current_user.username,
                    'attention_score': float(_to_native_number(attention_score)),
                    'attention_status': attention_status,
                    'face_detected': face_detected,
                    'timestamp': datetime.now().isoformat()
                }, room='teacher_room')
        except Exception:
            pass
        
        response_data = {
            'success': True,
            'emotions': serialized_emotions,
            'engagement_score': float(_to_native_number(engagement_score)),
            'attention_score': float(_to_native_number(attention_score)),
            'attention_status': attention_status,
            'face_detected': face_detected,
            'head_pose': head_pose,
            'eye_gaze': eye_gaze,
            'blink_rate': float(_to_native_number(blink_rate)),
            'yawn_detected': yawn_detected,
            'face_quality': face_quality
        }
        print("Sending response:", response_data)  # Debug print
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing emotion data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/audio_data', methods=['POST'])
@login_required
def process_audio_data():
    """Process audio data for voice activity and noise detection"""
    try:
        data = request.json
        if not data or 'audio_data' not in data:
            return jsonify({'success': False, 'error': 'No audio data received'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'Missing session ID'}), 400
        
        # Decode base64 audio data
        try:
            audio_base64 = data['audio_data']
            if ',' in audio_base64:
                audio_base64 = audio_base64.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            
            # Process audio
            voice_active, noise_level, rms_level = audio_processor.process_audio_chunk(audio_array)
            is_noisy = audio_processor.is_noisy_environment(noise_level)
            voice_status = audio_processor.get_voice_activity_status(voice_active, noise_level)
            
            print(f"Audio processed - Voice: {voice_active}, Noise: {noise_level:.3f}, Status: {voice_status}")  # Debug
            
            # Emit to teacher dashboard
            socketio.emit('audio_update', {
                'user_id': int(_to_native_number(current_user.id)),
                'username': current_user.username,
                'voice_active': voice_active,
                'noise_level': float(_to_native_number(noise_level)),
                'is_noisy': is_noisy,
                'voice_status': voice_status,
                'timestamp': datetime.now().isoformat()
            }, room='teacher_room')
            
            return jsonify({
                'success': True,
                'voice_active': voice_active,
                'noise_level': float(_to_native_number(noise_level)),
                'is_noisy': is_noisy,
                'voice_status': voice_status
            })
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        print(f"Error in audio endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send_feedback', methods=['POST'])
@login_required
def send_feedback():
    # Debug auth state
    try:
        print(f"[DEBUG] send_feedback: authenticated={current_user.is_authenticated}, role={getattr(current_user, 'role', None)}, user_id={getattr(current_user, 'id', None)}")
    except Exception:
        pass

    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    if current_user.role != 'teacher':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json or {}
        student_id = int(data['student_id'])
        message = data['message']
        feedback_type = data.get('feedback_type', 'general')
        session_id = data.get('session_id')

        # Resolve a session_id if not provided (use active or most recent session)
        if not session_id:
            active_session = Session.query.filter_by(user_id=student_id, is_active=True).first()
            if active_session:
                session_id = active_session.id
            else:
                last_session = Session.query.filter_by(user_id=student_id).order_by(Session.start_time.desc()).first()
                session_id = last_session.id if last_session else None
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session found for student'}), 400
        
        feedback = Feedback(
            teacher_id=current_user.id,
            student_id=student_id,
            session_id=session_id,
            message=message,
            feedback_type=feedback_type
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        # Emit to student
        socketio.emit('feedback_received', {
            'message': message,
            'feedback_type': feedback_type,
            'teacher': current_user.username,
            'timestamp': datetime.now().isoformat()
        }, room=f'student_{student_id}')
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"[DEBUG] send_feedback error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/student_stats/<int:student_id>')
@login_required
def get_student_stats(student_id):
    if current_user.role != 'teacher':
        return jsonify({'error': 'Unauthorized'})
    
    # Get emotion data for the student's recent sessions
    sessions = Session.query.filter_by(user_id=student_id).order_by(Session.start_time.desc()).limit(5).all()
    
    stats = []
    for session in sessions:
        emotion_data = EmotionData.query.filter_by(session_id=session.id).all()
        
        if emotion_data:
            emotions = [data.emotion for data in emotion_data]
            avg_engagement = sum(data.engagement_score for data in emotion_data) / len(emotion_data)
            avg_attention = sum(data.attention_score for data in emotion_data) / len(emotion_data)
            
            # Get attention status distribution
            status_counts = {}
            for data in emotion_data:
                status = data.attention_status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            stats.append({
                'session_id': session.id,
                'session_name': session.session_name,
                'start_time': session.start_time.isoformat(),
                'emotions': emotions,
                'avg_engagement': avg_engagement,
                'avg_attention': avg_attention,
                'attention_status_distribution': status_counts,
                'total_records': len(emotion_data)
            })
    
    return jsonify(stats)

@app.route('/api/attention_alerts')
@login_required
def get_attention_alerts():
    if current_user.role != 'teacher':
        return jsonify({'error': 'Unauthorized'})
    
    # Get recent unacknowledged alerts
    alerts = AttentionAlert.query.filter_by(is_acknowledged=False).order_by(AttentionAlert.timestamp.desc()).limit(50).all()
    
    alert_data = []
    for alert in alerts:
        student = User.query.get(alert.student_id)
        alert_data.append({
            'id': alert.id,
            'student_name': student.username if student else 'Unknown',
            'student_id': alert.student_id,
            'alert_type': alert.alert_type,
            'alert_message': alert.alert_message,
            'attention_score': alert.attention_score,
            'timestamp': alert.timestamp.isoformat(),
            'session_id': alert.session_id
        })
    
    return jsonify(alert_data)

@app.route('/api/acknowledge_alert/<int:alert_id>', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    if current_user.role != 'teacher':
        return jsonify({'error': 'Unauthorized'})
    
    try:
        alert = AttentionAlert.query.get(alert_id)
        if not alert:
            return jsonify({'error': 'Alert not found'})
        
        alert.is_acknowledged = True
        alert.acknowledged_by = current_user.id
        alert.acknowledged_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/attention_summary/<int:session_id>')
@login_required
def get_attention_summary(session_id):
    if current_user.role != 'teacher':
        return jsonify({'error': 'Unauthorized'})
    
    try:
        session = Session.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'})
        
        # Get emotion data for the session
        emotion_data = EmotionData.query.filter_by(session_id=session_id).all()
        
        if not emotion_data:
            return jsonify({'error': 'No data found for this session'})
        
        # Calculate summary statistics
        attention_scores = [data.attention_score for data in emotion_data]
        avg_attention = sum(attention_scores) / len(attention_scores)
        min_attention = min(attention_scores)
        max_attention = max(attention_scores)
        
        # Face detection rate
        face_detected_count = sum(1 for data in emotion_data if data.face_detected)
        face_detection_rate = face_detected_count / len(emotion_data)
        
        # Emotion distribution
        emotion_counts = {}
        for data in emotion_data:
            emotion = data.emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Attention status distribution
        status_counts = {}
        for data in emotion_data:
            status = data.attention_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary = {
            'session_id': session_id,
            'student_id': session.user_id,
            'avg_attention_score': avg_attention,
            'min_attention_score': min_attention,
            'max_attention_score': max_attention,
            'total_records': len(emotion_data),
            'face_detection_rate': face_detection_rate,
            'emotion_distribution': emotion_counts,
            'attention_status_distribution': status_counts,
            'current_attention_score': avg_attention,
            'current_status': 'Active' if face_detection_rate > 0.5 else 'Inactive',
            'monitoring_duration_minutes': len(emotion_data) / 60 if emotion_data else 0
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            join_room('teacher_room')
        else:
            join_room(f'student_{current_user.id}')
        print(f"User {current_user.username} connected")
        # Notify teachers about student presence
        if current_user.role == 'student':
            socketio.emit('student_presence', {
                'user_id': int(_to_native_number(current_user.id)),
                'username': current_user.username,
                'status': 'online',
                'timestamp': datetime.now().isoformat()
            }, room='teacher_room')

@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            leave_room('teacher_room')
        else:
            leave_room(f'student_{current_user.id}')
        print(f"User {current_user.username} disconnected")
        # Notify teachers about student presence
        if current_user.role == 'student':
            socketio.emit('student_presence', {
                'user_id': int(_to_native_number(current_user.id)),
                'username': current_user.username,
                'status': 'offline',
                'timestamp': datetime.now().isoformat()
            }, room='teacher_room')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create default teacher account if none exists
        if not User.query.filter_by(role='teacher').first():
            teacher = User(
                username='teacher',
                email='teacher@example.com',
                password_hash=generate_password_hash('password'),
                role='teacher'
            )
            db.session.add(teacher)
            db.session.commit()
            print("Default teacher account created: username='teacher', password='password'")
    
    socketio.run(app, debug=True, host='localhost', port=5000)
