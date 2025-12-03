#!/usr/bin/env python3

try:
    print("Starting application...")
    
    # Test imports
    print("Testing imports...")
    from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
    print("✓ Flask imported")
    
    from flask_socketio import SocketIO, emit, join_room, leave_room
    print("✓ Flask-SocketIO imported")
    
    from flask_login import LoginManager, login_user, logout_user, login_required, current_user
    print("✓ Flask-Login imported")
    
    from werkzeug.security import generate_password_hash, check_password_hash
    print("✓ Werkzeug imported")
    
    import base64
    import cv2
    import numpy as np
    from datetime import datetime
    import json
    print("✓ Standard libraries imported")
    
    from database import db, User, Session, EmotionData, Feedback, ClassRoom, AttentionAlert, AttentionSummary
    print("✓ Database models imported")
    
    from simple_emotion_detector import SimpleEmotionDetector
    print("✓ Emotion detector imported")
    
    print("All imports successful!")
    
    # Test app creation
    print("Creating Flask app...")
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-change-this'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_detection.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    print("✓ Flask app created")
    
    # Test extensions
    print("Initializing extensions...")
    db.init_app(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    print("✓ Extensions initialized")
    
    # Test emotion detector
    print("Initializing emotion detector...")
    emotion_detector = SimpleEmotionDetector()
    print("✓ Emotion detector initialized")
    
    print("All components initialized successfully!")
    print("Application is ready to run!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()



