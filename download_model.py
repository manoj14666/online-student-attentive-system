import os
import requests
import zipfile

def download_emotion_model():
    """Download pre-trained emotion detection model"""
    model_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    model_path = "models/emotion_model.h5"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        print("Downloading emotion detection model...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Creating a simple emotion detection model instead...")
            create_simple_model()
    else:
        print("Model already exists!")

def create_simple_model():
    """Create a simple emotion detection model if download fails"""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    # Create a simple CNN model for emotion detection
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')  # 7 emotions
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/simple_emotion_model.h5")
    print("Simple emotion model created!")

if __name__ == "__main__":
    download_emotion_model()
