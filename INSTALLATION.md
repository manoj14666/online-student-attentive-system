# Online Class Facial Emotion Detection - Installation Guide

## System Requirements

- Python 3.7 or higher
- Webcam/Camera access
- Modern web browser (Chrome, Firefox, Safari, Edge)
- At least 4GB RAM
- Internet connection (for initial model download)

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Clone or download the project files**
2. **Run the setup script:**
   ```bash
   python setup.py
   ```
3. **Start the application:**
   ```bash
   python app.py
   ```
4. **Open your browser and go to:** `http://localhost:5000`

### Option 2: Manual Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the emotion detection model:**
   ```bash
   python download_model.py
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

## Detailed Installation Steps

### Step 1: Python Environment Setup

1. **Check Python version:**
   ```bash
   python --version
   ```
   Ensure you have Python 3.7 or higher.

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv emotion_detection_env
   
   # On Windows:
   emotion_detection_env\Scripts\activate
   
   # On macOS/Linux:
   source emotion_detection_env/bin/activate
   ```

### Step 2: Install Dependencies

The project requires several Python packages. Install them using:

```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- Flask (web framework)
- OpenCV (computer vision)
- TensorFlow (machine learning)
- Flask-SocketIO (real-time communication)
- SQLAlchemy (database)

### Step 3: Download AI Model

The emotion detection uses a pre-trained model:

```bash
python download_model.py
```

If the download fails, the application will create a basic model automatically.

### Step 4: Database Setup

The application uses SQLite database which is created automatically on first run.

### Step 5: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Browser Setup

### Camera Permissions

1. **Allow camera access** when prompted by your browser
2. **Ensure your camera is not being used** by other applications
3. **Test camera functionality** in your browser settings

### Browser Compatibility

- **Chrome** (recommended): Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support

## Demo Accounts

After setup, you can use these demo accounts:

**Teacher Account:**
- Username: `teacher`
- Password: `password`

**Student Account:**
- Username: `student`
- Password: `password`

Or register new accounts using the registration form.

## Usage Instructions

### For Students

1. **Login** with your student account
2. **Allow camera access** when prompted
3. **Click "Start Detection"** to begin emotion monitoring
4. **Receive feedback** from your teacher in real-time
5. **View your engagement stats** and session analytics

### For Teachers

1. **Login** with your teacher account
2. **Monitor students** on the dashboard
3. **Send feedback** to individual students
4. **View analytics** and engagement reports
5. **Track class performance** over time

## Troubleshooting

### Common Issues

**1. Camera not working:**
- Check camera permissions in browser settings
- Ensure camera is not being used by other applications
- Try refreshing the page

**2. Model download fails:**
- Check internet connection
- The app will create a basic model automatically
- You can manually download models later

**3. Dependencies installation fails:**
- Update pip: `pip install --upgrade pip`
- Try installing packages individually
- Check Python version compatibility

**4. Database errors:**
- Delete `emotion_detection.db` file and restart
- Check file permissions in the project directory

**5. Port already in use:**
- Change port in `app.py`: `socketio.run(app, debug=True, port=5001)`
- Or kill the process using port 5000

### Performance Optimization

**For better performance:**
- Close unnecessary applications
- Use a modern browser
- Ensure good lighting for camera
- Use a stable internet connection

**For large classes:**
- Consider using a more powerful server
- Monitor system resources
- Adjust detection frequency in the code

## Development Setup

### For Developers

1. **Fork the repository**
2. **Create a development branch**
3. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```
4. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

### Customization

**To modify emotion detection:**
- Edit `emotion_detector.py`
- Adjust confidence thresholds
- Add new emotion categories

**To customize UI:**
- Modify templates in `templates/` directory
- Update CSS in `static/css/style.css`
- Add JavaScript functionality

## Security Considerations

- **Change default passwords** in production
- **Use HTTPS** for production deployment
- **Implement proper authentication** for real use
- **Secure database** with proper permissions
- **Regular security updates** for dependencies

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## Support

For issues and questions:
1. Check this documentation
2. Review error messages carefully
3. Check browser console for JavaScript errors
4. Verify all dependencies are installed correctly

## License

This project is created for educational purposes as part of a community service project.
