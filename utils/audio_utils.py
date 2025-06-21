import os
import io
from gtts import gTTS
import tempfile

def generate_speech_from_text(text, lang='en', slow=False):
    """
    Generate speech from text using gTTS (Google Text-to-Speech)
    
    Args:
        text (str): Text to convert to speech
        lang (str): Language code (default: 'en')
        slow (bool): Whether to speak slowly (default: False)
        
    Returns:
        io.BytesIO: Audio data as bytes
    """
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save to BytesIO object
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return audio_bytes
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def save_audio_to_file(audio_bytes, output_path):
    """
    Save audio bytes to file
    
    Args:
        audio_bytes (io.BytesIO): Audio data as bytes
        output_path (str): Path to save the audio file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write bytes to file
        with open(output_path, 'wb') as f:
            f.write(audio_bytes.getvalue())
        
        return True
    except Exception as e:
        print(f"Error saving audio to file: {e}")
        return False

def play_audio(audio_bytes):
    """
    Play audio in a temporary file
    
    Args:
        audio_bytes (io.BytesIO): Audio data as bytes
    """
    try:
        import pygame
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes.getvalue())
        
        # Load and play the audio
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        
        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
    except ImportError:
        print("pygame is required to play audio. Install with: pip install pygame")
    except Exception as e:
        print(f"Error playing audio: {e}")

def generate_multilingual_speech(text, language='en'):
    """
    Generate speech in specified language
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (e.g., 'en', 'fr', 'es', 'ja')
        
    Returns:
        io.BytesIO: Audio data as bytes
    """
    # Language mapping to full names for reference
    language_map = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'pt': 'Portuguese',
        'zh-CN': 'Chinese (Simplified)',
        'zh-TW': 'Chinese (Traditional)',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    # Check if language is supported
    if language not in language_map:
        print(f"Language '{language}' not recognized. Using English.")
        language = 'en'
    
    # Generate speech
    return generate_speech_from_text(text, lang=language, slow=False)