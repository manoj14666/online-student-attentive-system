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

# Import detectors
from simple_emotion_detector import SimpleEmotionDetector

# Try to import enhanced detector
try:
    from enhanced_emotion_detector import EnhancedEmotionDetector
    use_enhanced = True
except ImportError:
    EnhancedEmotionDetector = None
    use_enhanced = False

# Initialize the appropriate detector
if use_enhanced and EnhancedEmotionDetector:
    emotion_detector = EnhancedEmotionDetector()
    print("Using EnhancedEmotionDetector with eye tracking and attention monitoring")
else:
    emotion_detector = SimpleEmotionDetector()
    print("Using SimpleEmotionDetector (basic emotion detection only)")

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

# Utility to convert NumPy types to native Python for JSON
def _to_native_number(value):
    try:
        import numpy as _np
        if isinstance(value, (_np.generic,)):
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect emotions and attention features
        try:
            # Check if enhanced detector supports advanced features
            if hasattr(emotion_detector, 'detect_emotion_and_attention'):
                processed_frame, emotions_data, attention_data = emotion_detector.detect_emotion_and_attention(frame)
                
                # Extract attention features
                face_detected = attention_data.get('face_detected', len(emotions_data) > 0)
                attention_score = attention_data.get('attention_score', 0)
                attention_status = attention_data.get('status', 'Unknown')
                
                # Get detailed head pose and eye gaze
                head_pose = attention_data.get('head_pose', {})
                eye_gaze = attention_data.get('eye_gaze', {})
                face_quality = attention_data.get('face_quality', {})
                blink_detected = attention_data.get('blink_count', 0) > 0
            else:
                # Fallback to basic detection
                processed_frame, emotions_data = emotion_detector.detect_emotion(frame)
                face_detected = len(emotions_data) > 0
                attention_score = 0
                attention_status = "Unknown"
                head_pose = {}
                eye_gaze = {}
                face_quality = {}
                blink_detected = False
        except Exception as e:
            print(f"Enhanced detection failed: {e}, using fallback")
            processed_frame, emotions_data = emotion_detector.detect_emotion(frame)
            face_detected = len(emotions_data) > 0
            attention_score = 0
            attention_status = "Unknown"
            head_pose = {}
            eye_gaze = {}
            face_quality = {}
            blink_detected = False

        # Calculate engagement score (0-1)
        engagement_score = float(_to_native_number(emotion_detector.get_engagement_score(emotions_data)))

        # Save emotion data to database with attention features
        for ed in emotions_data:
            record = EmotionData(
                session_id=session_id,
                emotion=ed['emotion'],
                confidence=float(_to_native_number(ed['confidence'])),
                engagement_score=engagement_score,
                face_detected=face_detected,
                attention_score=attention_score,
                attention_status=attention_status,
                head_pitch=head_pose.get('pitch', 0.0),
                head_yaw=head_pose.get('yaw', 0.0),
                head_roll=head_pose.get('roll', 0.0),
                eye_gaze_direction=eye_gaze.get('gaze_direction', 'center'),
                left_eye_open=eye_gaze.get('left_eye_open', True),
                right_eye_open=eye_gaze.get('right_eye_open', True),
                blink_detected=blink_detected,
                face_quality_score=face_quality.get('quality_score', 0.8)
            )
            db.session.add(record)

        db.session.commit()

        # JSON-safe payload
        serialized_emotions = _serialize_emotions(emotions_data)

        # Emit to teacher dashboard
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
    print(f"[DEBUG] Current user authenticated: {current_user.is_authenticated}")
    print(f"[DEBUG] Current user role: {current_user.role if current_user.is_authenticated else 'None'}")
    
    if not current_user.is_authenticated:
        print("[DEBUG] User not authenticated")
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    if current_user.role != 'teacher':
        print(f"[DEBUG] User is not teacher, role: {current_user.role}")
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json
        student_id = data['student_id']
        message = data['message']
        feedback_type = data.get('feedback_type', 'general')
        session_id = data.get('session_id')
        
        print(f"[DEBUG] Sending feedback to student {student_id} from teacher {current_user.id}")
        
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
        
        print(f"[DEBUG] Feedback sent successfully")
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"[DEBUG] Error in send_feedback: {e}")
        import traceback
        traceback.print_exc()
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
            
            stats.append({
                'session_id': session.id,
                'session_name': session.session_name,
                'start_time': session.start_time.isoformat(),
                'emotions': emotions,
                'avg_engagement': avg_engagement,
                'total_records': len(emotion_data)
            })
    
    return jsonify(stats)

@app.route('/api/attention_alerts')
@login_required
def get_attention_alerts():
    if current_user.role != 'teacher':
        return jsonify({'error': 'Unauthorized'})
    
    # Get recent unacknowledged alerts (if AttentionAlert table exists)
    try:
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
    except Exception as e:
        # Table might not exist yet, return empty list
        print(f"AttentionAlert table not available: {e}")
        return jsonify([])

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

        # Ensure new columns exist for backward-compatible upgrade on SQLite
        try:
            from sqlalchemy import text
            conn = db.engine.connect()
            # Get existing columns
            cols = conn.execute(text("PRAGMA table_info(emotion_data)")).fetchall()
            existing = {c[1] for c in cols}  # column name is at index 1

            alters = []
            if 'attention_score' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN attention_score REAL DEFAULT 0.0")
            if 'attention_status' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN attention_status TEXT DEFAULT 'Unknown'")
            if 'head_pitch' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN head_pitch REAL DEFAULT 0.0")
            if 'head_yaw' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN head_yaw REAL DEFAULT 0.0")
            if 'head_roll' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN head_roll REAL DEFAULT 0.0")
            if 'eye_gaze_direction' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN eye_gaze_direction TEXT DEFAULT 'center'")
            if 'left_eye_open' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN left_eye_open INTEGER DEFAULT 1")
            if 'right_eye_open' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN right_eye_open INTEGER DEFAULT 1")
            if 'blink_detected' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN blink_detected INTEGER DEFAULT 0")
            if 'face_quality_score' not in existing:
                alters.append("ALTER TABLE emotion_data ADD COLUMN face_quality_score REAL DEFAULT 0.5")

            for stmt in alters:
                conn.execute(text(stmt))
            conn.close()
        except Exception as _e:
            # Safe to print; non-fatal
            print(f"Schema check/upgrade skipped or failed: {_e}")
        
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
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
