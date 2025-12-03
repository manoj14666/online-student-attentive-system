from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
from datetime import datetime
import json

from database import db, User, Session, EmotionData, Feedback, ClassRoom
from emotion_detector import EmotionDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize emotion detector
emotion_detector = EmotionDetector()

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
        
        # Detect emotions
        processed_frame, emotions_data = emotion_detector.detect_emotion(frame)
        
        # Calculate engagement score
        engagement_score = emotion_detector.get_engagement_score(emotions_data)
        
        # Save emotion data to database
        for emotion_data in emotions_data:
            emotion_record = EmotionData(
                session_id=session_id,
                emotion=emotion_data['emotion'],
                confidence=emotion_data['confidence'],
                engagement_score=engagement_score,
                face_detected=True
            )
            db.session.add(emotion_record)
        
        db.session.commit()
        
        # Emit to teacher dashboard
        socketio.emit('emotion_update', {
            'user_id': current_user.id,
            'username': current_user.username,
            'emotions': emotions_data,
            'engagement_score': engagement_score,
            'timestamp': datetime.now().isoformat()
        }, room='teacher_room')
        
        return jsonify({
            'success': True,
            'emotions': emotions_data,
            'engagement_score': engagement_score
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
            
            stats.append({
                'session_id': session.id,
                'session_name': session.session_name,
                'start_time': session.start_time.isoformat(),
                'emotions': emotions,
                'avg_engagement': avg_engagement,
                'total_records': len(emotion_data)
            })
    
    return jsonify(stats)

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
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
