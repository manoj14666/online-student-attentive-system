#!/usr/bin/env python3
"""
Simple Setup script for Online Class Facial Emotion Detection Project
This script helps set up the project environment with minimal dependencies.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_simple_requirements():
    """Install simplified requirements"""
    print("\n📦 Installing simplified requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    directories = [
        "static/css",
        "static/js", 
        "static/images",
        "templates",
        "models",
        "uploads"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_database():
    """Initialize the database"""
    print("\n🗄️ Setting up database...")
    try:
        from app_simple import app, db
        with app.app_context():
            db.create_all()
            print("✓ Database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\n👥 Creating sample data...")
    try:
        from app_simple import app, db, User
        from werkzeug.security import generate_password_hash
        
        with app.app_context():
            # Check if default teacher exists
            if not User.query.filter_by(username='teacher').first():
                teacher = User(
                    username='teacher',
                    email='teacher@example.com',
                    password_hash=generate_password_hash('password'),
                    role='teacher'
                )
                db.session.add(teacher)
                db.session.commit()
                print("✓ Default teacher account created")
            
            # Check if sample student exists
            if not User.query.filter_by(username='student').first():
                student = User(
                    username='student',
                    email='student@example.com',
                    password_hash=generate_password_hash('password'),
                    role='student'
                )
                db.session.add(student)
                db.session.commit()
                print("✓ Sample student account created")
        
        return True
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def print_instructions():
    """Print setup completion instructions"""
    print("\n" + "="*60)
    print("🎉 SIMPLE SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Run the simplified application:")
    print("   python app_simple.py")
    print("\n2. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n3. Login with demo accounts:")
    print("   Teacher: username='teacher', password='password'")
    print("   Student: username='student', password='password'")
    print("\n4. Or register new accounts using the registration form")
    print("\n📚 Features:")
    print("• Real-time facial emotion detection (simplified)")
    print("• Teacher dashboard for monitoring students")
    print("• Instant feedback system")
    print("• Engagement analytics")
    print("• Session management")
    print("\n🔧 Troubleshooting:")
    print("• Make sure your camera is connected and accessible")
    print("• Allow camera permissions in your browser")
    print("• Check that all dependencies are installed")
    print("\n📖 For more information, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Online Class Facial Emotion Detection - Simple Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install simplified requirements
    if not install_simple_requirements():
        print("\n❌ Setup failed at package installation.")
        print("Please install packages manually: pip install -r requirements_simple.txt")
        return False
    
    # Setup database
    if not setup_database():
        print("\n❌ Setup failed at database initialization.")
        return False
    
    # Create sample data
    if not create_sample_data():
        print("\n⚠️ Warning: Sample data creation failed, but setup can continue.")
    
    # Print instructions
    print_instructions()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
