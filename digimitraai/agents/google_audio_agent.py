from google.cloud import speech_v1
from google.cloud import texttospeech_v1
from google.cloud import translate_v2 as translate
import os
from typing import Dict, Optional
import tempfile

class GoogleAudioAgent:
    def __init__(self, credentials_path: Optional[str] = None):
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
        self.speech_client = speech_v1.SpeechClient()
        self.tts_client = texttospeech_v1.TextToSpeechClient()
        self.translate_client = translate.Client()
        
        # Language configurations for Indian languages
        self.language_configs = {
            'malayalam': {
                'code': 'ml-IN',
                'name': 'Malayalam',
                'voice_name': 'ml-IN-Standard-A',
                'translate_code': 'ml'
            },
            'hindi': {
                'code': 'hi-IN',
                'name': 'Hindi',
                'voice_name': 'hi-IN-Standard-A',
                'translate_code': 'hi'
            },
            'tamil': {
                'code': 'ta-IN',
                'name': 'Tamil',
                'voice_name': 'ta-IN-Standard-A',
                'translate_code': 'ta'
            },
            'telugu': {
                'code': 'te-IN',
                'name': 'Telugu',
                'voice_name': 'te-IN-Standard-A',
                'translate_code': 'te'
            },
            'english': {
                'code': 'en-IN',
                'name': 'English',
                'voice_name': 'en-IN-Standard-A',
                'translate_code': 'en'
            }
            # Add more Indian languages as needed
        }

    def speech_to_text(self, audio_data: str, source_language: str) -> Dict:
        """Convert speech to text using Google Speech-to-Text"""
        try:
            # Get language configuration
            lang_config = self.language_configs.get(source_language)
            if not lang_config:
                return {
                    "success": False,
                    "error": f"Unsupported language: {source_language}"
                }

            # Read the audio file
            with open(audio_data, 'rb') as audio:
                content = audio.read()

            # Configure audio and recognition
            audio = speech_v1.RecognitionAudio(content=content)
            config = speech_v1.RecognitionConfig(
                language_code=lang_config['code'],
                enable_automatic_punctuation=True,
                # model='command_and_search',
                model=lang_config.get('model', 'command_and_search'),
                use_enhanced=True
                  # Allow language-specific models
                # audio_channel_count=2,  # For stereo audio (can adjust dynamically)
                # enable_word_time_offsets=True,  # Provides timestamps for each word
                # enable_word_confidence=True 
            )

            # Perform the transcription
            response = self.speech_client.recognize(config=config, audio=audio)

            if not response.results:
                return {
                    "success": False,
                    "error": "No speech detected"
                }

            # Get transcribed text
            text = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence

            # If source is not English, translate to English
            if source_language != 'english':
                translation = self.translate_client.translate(
                    text,
                    source_language=lang_config['translate_code'],
                    target_language='en'
                )
                english_text = translation['translatedText']
            else:
                english_text = text

            return {
                "success": True,
                "text": text,  # Original language text
                "original_text": english_text,  # English translation
                "confidence": confidence,
                "language": source_language
            }

        except Exception as e:
            print(f"Speech-to-text error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def text_to_speech(self, text: str, language: str) -> Dict:
        """Convert text to speech using Google Text-to-Speech"""
        try:
            # Get language configuration
            lang_config = self.language_configs.get(language)
            if not lang_config:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}"
                }

            # Set the voice
            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=lang_config['code'],
                name=lang_config['voice_name']
            )

            # Set the audio config
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.MP3,
                speaking_rate=0.9,  # Slightly slower for clarity
                pitch=0.0
            )

            # Set the synthesis input
            synthesis_input = texttospeech_v1.SynthesisInput(text=text)

            # Generate the audio
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Create temporary audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as output:
                output.write(response.audio_content)
                output_path = output.name

            return {
                "success": True,
                "audio_path": output_path,
                "language": language
            }

        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def process_audio_query(self, audio_data: tuple, source_language: str) -> Dict:
        """Process audio from Gradio (returns tuple of (path, sampling_rate))"""
        try:
            if not audio_data:
                return {
                    "success": False,
                    "error": "No audio data provided"
                }
            
            # Gradio returns (file_path, sampling_rate)
            audio_path = audio_data[0] if isinstance(audio_data, tuple) else audio_data
            
            # Convert speech to text
            stt_result = self.speech_to_text(audio_path, source_language)
            
            if not stt_result["success"]:
                return stt_result
                
            return {
                "success": True,
                "text": stt_result["text"],
                "original_text": stt_result["original_text"],
                "confidence": stt_result["confidence"],
                "source_language": source_language
            }
                
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }