from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'teacher' or 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = db.relationship('Session', backref='user', lazy=True)
    # Disambiguate Feedback relations (two FKs to User)
    feedbacks_sent = db.relationship(
        'Feedback',
        foreign_keys='Feedback.teacher_id',
        backref='teacher',
        lazy=True
    )
    feedbacks_received = db.relationship(
        'Feedback',
        foreign_keys='Feedback.student_id',
        backref='student',
        lazy=True
    )

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    emotion_data = db.relationship('EmotionData', backref='session', lazy=True)

class EmotionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    emotion = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    engagement_score = db.Column(db.Float, nullable=False)
    face_detected = db.Column(db.Boolean, default=True)
    
    # Enhanced attention monitoring fields
    attention_score = db.Column(db.Float, default=0.0)
    attention_status = db.Column(db.String(50), default='Unknown')
    head_pitch = db.Column(db.Float, default=0.0)
    head_yaw = db.Column(db.Float, default=0.0)
    head_roll = db.Column(db.Float, default=0.0)
    eye_gaze_direction = db.Column(db.String(20), default='center')
    left_eye_open = db.Column(db.Boolean, default=True)
    right_eye_open = db.Column(db.Boolean, default=True)
    blink_detected = db.Column(db.Boolean, default=False)
    face_quality_score = db.Column(db.Float, default=0.5)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    feedback_type = db.Column(db.String(20), nullable=False)  # 'encouragement', 'warning', 'general'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

class ClassRoom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    students = db.relationship('User', secondary='classroom_students', backref='classrooms')

# Association table for many-to-many relationship between classrooms and students
classroom_students = db.Table('classroom_students',
    db.Column('classroom_id', db.Integer, db.ForeignKey('class_room.id'), primary_key=True),
    db.Column('student_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

class AttentionAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # 'low_attention', 'face_absent', 'distracted', etc.
    alert_message = db.Column(db.Text, nullable=False)
    attention_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    acknowledged_at = db.Column(db.DateTime)
    
    # Relationships
    session = db.relationship('Session', backref='attention_alerts')
    student = db.relationship('User', foreign_keys=[student_id], backref='attention_alerts_received')
    acknowledged_by_user = db.relationship('User', foreign_keys=[acknowledged_by], backref='attention_alerts_acknowledged')

class AttentionSummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    summary_period_start = db.Column(db.DateTime, nullable=False)
    summary_period_end = db.Column(db.DateTime, nullable=False)
    avg_attention_score = db.Column(db.Float, nullable=False)
    min_attention_score = db.Column(db.Float, nullable=False)
    max_attention_score = db.Column(db.Float, nullable=False)
    total_records = db.Column(db.Integer, nullable=False)
    face_detection_rate = db.Column(db.Float, nullable=False)
    emotion_distribution = db.Column(db.Text)  # JSON string
    attention_status_distribution = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    session = db.relationship('Session', backref='attention_summaries')
    student = db.relationship('User', backref='attention_summaries')
