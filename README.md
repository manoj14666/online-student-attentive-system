# Online Class Facial Emotion Detection with Teacher Feedback

## Project Overview
This project provides real-time facial emotion detection for online classes, allowing teachers to monitor student engagement and provide timely feedback.

## Features
- Real-time facial emotion detection using webcam
- Teacher dashboard for monitoring multiple students
- Feedback system for teachers to provide guidance
- Analytics and reporting for class engagement
- User authentication and session management

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the emotion detection model:
   ```bash
   python download_model.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. **Students**: Access the student interface and allow camera permissions
2. **Teachers**: Login to the teacher dashboard to monitor student emotions
3. **Feedback**: Teachers can send real-time feedback to students

## Technology Stack
- Backend: Flask, SocketIO
- Frontend: HTML, CSS, JavaScript
- AI/ML: TensorFlow, OpenCV, face_recognition
- Database: SQLite

## Project Structure
```
├── app.py                 # Main Flask application
├── emotion_detector.py    # Emotion detection logic
├── models/               # AI models directory
├── static/               # CSS, JS, images
├── templates/            # HTML templates
├── database.py           # Database models
└── requirements.txt      # Python dependencies
```

## Contributing
This is a community service project for educational purposes.

## License
MIT License
