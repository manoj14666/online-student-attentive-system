import numpy as np
import math

class AudioProcessor:
    """Process audio for voice activity detection and noise level analysis"""
    
    def __init__(self):
        self.audio_history = []
        self.silence_threshold = 0.005  # RMS threshold for silence (lower = more sensitive)
        self.noise_threshold = 0.08  # RMS threshold for excessive noise (lower = more sensitive)
        
        # Voice activity detection parameters
        self.voice_threshold = 0.015  # RMS threshold for voice (lower = more sensitive)
        self.voice_frames = 0
        self.silence_frames = 0
        
    def process_audio_chunk(self, audio_data, sample_rate=44100):
        """
        Process audio chunk (PCM data)
        Returns: voice_active, noise_level, rms_level
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                return False, 0.0, 0.0
            
            # Convert to numpy array if needed
            if isinstance(audio_data, list):
                audio_array = np.array(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] if needed
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                # Assume 16-bit PCM
                audio_array = audio_array / 32768.0
            
            # Calculate RMS (Root Mean Square) for volume level
            rms = np.sqrt(np.mean(audio_array ** 2))
            
            # Also calculate peak for better detection
            peak = np.max(np.abs(audio_array))
            
            # Combine RMS and peak for better voice detection
            combined_level = (rms * 0.7) + (peak * 0.3)
            
            # Store in history (keep last 30 frames ~1 second)
            self.audio_history.append(combined_level)
            if len(self.audio_history) > 30:
                self.audio_history.pop(0)
            
            # Calculate average level for noise level
            avg_level = np.mean(self.audio_history) if self.audio_history else 0.0
            
            # Determine noise level (0.0 = silent, 1.0 = very loud) - more sensitive scaling
            noise_level = min(avg_level * 20.0, 1.0)  # Increased multiplier for better visibility
            
            # Voice activity detection using combined level
            voice_active = False
            if combined_level > self.voice_threshold:
                self.voice_frames += 1
                self.silence_frames = 0
                if self.voice_frames >= 2:  # Reduced from 3 to 2 for faster detection
                    voice_active = True
            else:
                self.silence_frames += 1
                if self.silence_frames >= 5:  # Reduced from 10 for faster response
                    self.voice_frames = 0
            
            return voice_active, noise_level, combined_level
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return False, 0.0, 0.0
    
    def is_noisy_environment(self, noise_level):
        """Determine if environment is too noisy"""
        return noise_level > self.noise_threshold
    
    def get_voice_activity_status(self, voice_active, noise_level):
        """Get human-readable voice activity status"""
        if voice_active:
            return "Speaking"
        elif noise_level > self.noise_threshold:
            return "Noisy Environment"
        elif noise_level < self.silence_threshold:
            return "Silent"
        else:
            return "Background Noise"

