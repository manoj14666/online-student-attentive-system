# Enhanced Emotion Detection System with Attention Monitoring

## Overview

This enhanced emotion detection system now includes comprehensive attention monitoring features that go beyond simple facial emotion detection. The system can track student attention levels, detect disengagement, and provide real-time alerts to teachers.

## New Attention Monitoring Features

### 🔹 Head Pose Estimation
- **Purpose**: Track whether the student's face is directed toward the screen
- **Implementation**: Uses computer vision techniques to estimate head pitch, yaw, and roll
- **Alert Trigger**: If the student frequently looks away (down, sideways, or phone direction), they are marked as "Inattentive"

### 🔹 Eye Gaze Tracking
- **Purpose**: Monitor eye focus and detect blinking patterns
- **Features**:
  - Simple gaze estimation using eye landmarks
  - Eye openness detection (left and right eyes)
  - Blink detection using Eye Aspect Ratio (EAR)
  - Gaze direction estimation (left, center, right)
- **Alert Trigger**: If eyes are closed, blinking excessively, or looking away for extended periods, attention score is reduced

### 🔹 Face Presence Monitoring
- **Purpose**: Detect if the student is present in the camera view
- **Features**:
  - Real-time face detection
  - Face absence duration tracking
  - Face quality assessment (brightness, contrast, sharpness)
- **Alert Trigger**: If face is missing for more than 30 seconds, triggers "Student not detected — possible disengagement" alert

### 🔹 Prolonged Neutral Emotion Detection
- **Purpose**: Identify students with consistently low emotional engagement
- **Implementation**: Monitors emotion history over time windows
- **Alert Trigger**: If detected emotion stays Neutral or "No Face Detected" for > 5 minutes, flags as "Low Engagement"

## Behavioral Interpretation

The system combines multiple factors to calculate a comprehensive "Attentiveness Score":

### Attention Score Components (0-100%)
1. **Face Detection Rate** (30% weight): Percentage of time face is detected
2. **Head Pose Score** (25% weight): Penalty for large head turns away from screen
3. **Eye Gaze Score** (25% weight): Penalty for closed eyes or off-center gaze
4. **Emotion Engagement** (20% weight): Weighted average of emotional responses

### Attention Status Categories
- **Attentive** (80-100%): Student is focused and engaged
- **Partially Attentive** (50-79%): Some attention but may be distracted
- **Distracted** (20-49%): Student appears distracted or unfocused
- **Inattentive** (0-19%): Very low attention, likely disengaged
- **Absent / Disengaged**: Face not detected for extended periods
- **Low Engagement**: Prolonged neutral emotions

## Enhanced Teacher Dashboard

### Real-time Monitoring
The teacher dashboard now displays:

| Student | Current Emotion | Attention (%) | Status |
|---------|----------------|---------------|---------|
| StudentA | Happy | 95% | Attentive |
| StudentB | Neutral | 40% | Partially Attentive |
| StudentC | No Face | 0% | Absent / Disengaged |

### Attention Alerts Panel
- **Real-time Alerts**: Shows active attention alerts with acknowledgment buttons
- **Alert Types**:
  - Low Attention: Attention score below 30%
  - Face Absent: No face detected
  - Distracted: Student appears distracted or inattentive
- **Alert Management**: Teachers can acknowledge alerts to remove them from the active list

### Enhanced Analytics
- **Average Attention Score**: Real-time average attention across all students
- **Active Alerts Counter**: Number of unacknowledged attention alerts
- **Face Detection Rate**: Percentage of time faces are detected
- **Attention Status Distribution**: Breakdown of student attention states

## Technical Implementation

### Database Schema Updates
New fields added to `EmotionData` table:
- `attention_score`: Calculated attention score (0-100)
- `attention_status`: Current attention status
- `head_pitch`, `head_yaw`, `head_roll`: Head pose angles
- `eye_gaze_direction`: Gaze direction (left/center/right)
- `left_eye_open`, `right_eye_open`: Eye openness status
- `blink_detected`: Whether a blink was detected
- `face_quality_score`: Image quality assessment

New tables:
- `AttentionAlert`: Stores attention-related alerts
- `AttentionSummary`: Aggregated attention data for sessions

### API Endpoints
- `GET /api/attention_alerts`: Retrieve active attention alerts
- `POST /api/acknowledge_alert/<alert_id>`: Acknowledge an alert
- `GET /api/attention_summary/<session_id>`: Get detailed attention summary for a session

### Enhanced Emotion Detector
The `EnhancedEmotionDetector` class includes:
- Real-time attention monitoring
- Historical data tracking (5-minute windows)
- Multi-factor attention score calculation
- Automatic alert generation
- Visual attention indicators on video feed

## Usage Instructions

### For Teachers
1. **Monitor Real-time**: Watch the attention monitoring panel for live updates
2. **Review Alerts**: Check the attention alerts panel for students needing attention
3. **Acknowledge Alerts**: Click the checkmark to acknowledge and remove alerts
4. **Send Feedback**: Use enhanced feedback system to address attention issues

### For Students
1. **Position Camera**: Ensure face is clearly visible in camera
2. **Maintain Focus**: Keep eyes directed toward screen
3. **Stay Present**: Avoid leaving camera view for extended periods
4. **Monitor Status**: Check attention status in student interface

## Configuration

### Attention Thresholds
- `NEUTRAL_DURATION_THRESHOLD`: 300 seconds (5 minutes)
- `FACE_ABSENCE_THRESHOLD`: 30 seconds
- `BLINK_THRESHOLD`: 0.3 (Eye Aspect Ratio)
- `HEAD_TURN_THRESHOLD`: 30 degrees

### Alert Conditions
- Low attention score: < 30%
- Face absence: > 30 seconds
- Distracted status: Attention status = "Distracted" or "Inattentive"

## Benefits

### For Educators
- **Early Intervention**: Detect disengagement before it becomes a problem
- **Data-Driven Insights**: Understand student attention patterns
- **Efficient Monitoring**: Monitor multiple students simultaneously
- **Targeted Feedback**: Send specific feedback based on attention data

### For Students
- **Self-Awareness**: Understand their own attention patterns
- **Improved Focus**: Visual feedback encourages better attention
- **Fair Assessment**: Attention data provides context for performance

## Privacy Considerations

- **Local Processing**: All video processing happens locally
- **No Video Storage**: Only attention metrics are stored, not video frames
- **Consent**: Students should be informed about attention monitoring
- **Data Retention**: Attention data can be configured for automatic deletion

## Future Enhancements

- **Machine Learning**: Train models on attention patterns for better accuracy
- **Integration**: Connect with learning management systems
- **Analytics**: Advanced reporting and trend analysis
- **Mobile Support**: Extend to mobile devices and tablets
- **Accessibility**: Support for students with different needs

## Installation and Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run database migrations: `python -c "from app_simple import app, db; app.app_context().push(); db.create_all()"`
3. Start the application: `python app_simple.py`
4. Access teacher dashboard at: `http://localhost:5000/teacher`
5. Access student interface at: `http://localhost:5000/student`

## Support

For technical support or feature requests, please refer to the main project documentation or contact the development team.
