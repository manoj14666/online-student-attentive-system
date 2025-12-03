from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import numpy as np
from datetime import datetime
import json

from database import db, User, Session, EmotionData, Feedback, ClassRoom, AttentionAlert, AttentionSummary

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

# Simple emotion detector without OpenCV
class SimpleEmotionDetector:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def detect_emotion(self, frame):
        """Mock emotion detection for testing"""
        import random
        emotions_data = [{
            'emotion': random.choice(self.emotion_labels),
            'confidence': random.uniform(0.6, 0.9),
            'bbox': (100, 100, 200, 200)
        }]
        return frame, emotions_data
    
    def get_engagement_score(self, emotions_data):
        """Calculate engagement score based on emotions"""
        if not emotions_data:
            return 0
        
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

# Initialize emotion detector
emotion_detector = SimpleEmotionDetector()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility to convert NumPy types to native Python for JSON
def _to_native_number(value):
    try:
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
    
    # Get active session or create new one
    active_session = Session.query.filter_by(user_id=current_user.id, is_active=True).first()
    if not active_session:
        active_session = Session(
            user_id=current_user.id,
            session_name=f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        db.session.add(active_session)
        db.session.commit()
    
    return render_template('student.html', session_id=active_session.id)

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
        data = request.json
        image_data = data['image']
        session_id = data['session_id']
        
        # Mock image processing for testing
        import random
        
        # Detect emotions
        processed_frame, emotions_data = emotion_detector.detect_emotion(None)
        
        # Calculate engagement score
        engagement_score = float(_to_native_number(emotion_detector.get_engagement_score(emotions_data)))
        
        # Calculate simple attention score based on face detection and engagement
        face_detected = len(emotions_data) > 0
        attention_score = engagement_score * 100 if face_detected else 0
        attention_status = "Attentive" if attention_score > 70 else "Partially Attentive" if attention_score > 40 else "Distracted"
        
        # Save emotion data to database
        for emotion_data in emotions_data:
            emotion_record = EmotionData(
                session_id=session_id,
                emotion=emotion_data['emotion'],
                confidence=emotion_data['confidence'],
                engagement_score=engagement_score,
                face_detected=face_detected,
                attention_score=attention_score,
                attention_status=attention_status,
                head_pitch=0.0,
                head_yaw=0.0,
                head_roll=0.0,
                eye_gaze_direction='center',
                left_eye_open=True,
                right_eye_open=True,
                blink_detected=False,
                face_quality_score=0.8
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

        # Emit to teacher dashboard with attention data
        socketio.emit('emotion_update', {
            'user_id': int(_to_native_number(current_user.id)),
            'username': current_user.username,
            'emotions': serialized_emotions,
            'engagement_score': engagement_score,
            'attention_score': attention_score,
            'attention_status': attention_status,
            'face_detected': face_detected,
            'timestamp': datetime.now().isoformat()
        }, room='teacher_room')
        
        return jsonify({
            'success': True,
            'emotions': serialized_emotions,
            'engagement_score': engagement_score,
            'attention_score': attention_score,
            'attention_status': attention_status,
            'face_detected': face_detected
        })
        
    except Exception as e:
        print(f"Error processing emotion data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/send_feedback', methods=['POST'])
@login_required
def send_feedback():
    if current_user.role != 'teacher':
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    try:
        data = request.json
        student_id = data['student_id']
        message = data['message']
        feedback_type = data.get('feedback_type', 'general')
        session_id = data.get('session_id')
        
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
        return jsonify({'success': False, 'error': str(e)})

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

@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        if current_user.role == 'teacher':
            leave_room('teacher_room')
        else:
            leave_room(f'student_{current_user.id}')
        print(f"User {current_user.username} disconnected")

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
    
    print("Starting Enhanced Emotion Detection System with Attention Monitoring...")
    print("Access the application at: http://localhost:5000")
    print("Teacher login: username='teacher', password='password'")
    socketio.run(app, debug=True, host='localhost', port=5000)



