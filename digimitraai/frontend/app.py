import gradio as gr
import sys
import os
import traceback
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Union, Any
from pathlib import Path
import scipy.io.wavfile

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.manager_agent import ManagerAgent

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.manager_agent import ManagerAgent

class AadhaarChatInterface:
    def __init__(self):
        self.manager_agent = ManagerAgent()
        self.chat_history = []
        
        # Get supported languages
        self.languages = self.manager_agent.multilingual_agent.supported_languages
        self.language_names = {lang: info['name'] for lang, info in self.languages.items()}

    def process_query(self, message: str, audio_data: any, source_language: str, target_language: str, history: list) -> list:
        try:
            if audio_data:
                # Handle audio input (both upload and recording)
                if isinstance(audio_data, tuple):
                    audio_samples, sampling_rate = audio_data
                    
                    # Create a temporary WAV file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name

                    # Convert and save audio data using scipy
                    try:
                        import scipy.io.wavfile as wav
                        if isinstance(audio_samples, np.ndarray):
                            # Ensure the data is in the right format
                            audio_samples = (audio_samples * 32767).astype(np.int16)
                            wav.write(temp_path, sampling_rate, audio_samples)
                        elif isinstance(audio_samples, str) and os.path.exists(audio_samples):
                            # Copy existing file
                            shutil.copy2(audio_samples, temp_path)
                        
                        # Process audio with temporary file
                        response = self.manager_agent.process_multilingual_query(
                            audio_data=temp_path,
                            source_language=source_language,
                            target_language=target_language
                        )
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    response = self.manager_agent.process_multilingual_query(
                        audio_data=audio_data,
                        source_language=source_language,
                        target_language=target_language
                    )
                
                query = response.get("original_query", "")
                
            elif message:
                # Process text input
                response = self.manager_agent.process_multilingual_query(
                    query=message,
                    source_language=source_language,
                    target_language=target_language
                )
                query = message
            else:
                return history
            
            # Format response for chat history
            if not history:
                history = []
                
            history.append([query, response["answer"]])
            
            # Add debug info if available
            debug_info = ""
            if "confidence" in response:
                debug_info += f"Confidence: {response['confidence']:.2f}\n"
            if "sources" in response:
                debug_info += "Sources:\n" + "\n".join(response.get("sources", []))
            
            if debug_info:
                history.append([None, f"Debug Info:\n{debug_info}"])
                
            return history

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(f"Error details: {str(e)}")
            traceback.print_exc()  # Print full traceback
            if not history:
                history = []
            history.append([message if message else "Audio Query", error_message])
            return history

    def process_uploaded_audio(audio_input, source_lang, target_lang, history):
        if audio_input is not None:
            # audio_input is tuple (file_path, sampling_rate)
            return self.process_query(None, audio_input, source_lang, target_lang, history)
        return history

    def process_recorded_audio(audio_input, source_lang, target_lang, history):
        if audio_input is not None:
            # audio_input is tuple (file_path, sampling_rate)
            return self.process_query(None, audio_input, source_lang, target_lang, history)
        return history
    
    def process_with_audio(self, message, audio_path, src_lang, tgt_lang, history, enable_voice):
        try:
            # Process query
            if audio_path:
                response = self.manager_agent.process_multilingual_query(
                    audio_data=audio_path,
                    source_language=src_lang,
                    target_language=tgt_lang
                )
                query = response.get("original_text", "")
            else:
                response = self.manager_agent.process_multilingual_query(
                    query=message,
                    source_language=src_lang,
                    target_language=tgt_lang
                )
                query = message

            # Update chat history
            history.append((query, response["answer"]))

            # Generate audio if enabled
            audio_path = None
            if enable_voice:
                audio_result = self.manager_agent.google_audio_agent.text_to_speech(
                    response["answer"],
                    tgt_lang
                )
                if audio_result["success"]:
                    audio_path = audio_result["audio_path"]

            return history, audio_path
            
        except Exception as e:
            print(f"Error: {str(e)}")
            history.append((message or "Audio Input", f"Error: {str(e)}"))
            return history, None

    def create_interface(self):
        def clear_fn():
            return [], None

        with gr.Blocks(title="Aadhaar Customer Service Assistant") as interface:
            gr.Markdown("# Aadhaar Customer Service Assistant")
            
            with gr.Row():
                source_language = gr.Dropdown(
                    choices=list(self.language_names.keys()),
                    value="english",
                    label="Input Language"
                )
                target_language = gr.Dropdown(
                    choices=list(self.language_names.keys()),
                    value="english",
                    label="Output Language"
                )

            chatbot = gr.Chatbot([], height=400)
            audio_output = gr.Audio(label="Response Audio", visible=False)

            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Type your question here...",
                    scale=4
                )

            with gr.Row():
                mic_audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record Audio"
                )
                upload_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload Audio File"
                )

            enable_audio = gr.Checkbox(label="Enable Voice Response", value=False)
            clear = gr.Button("Clear")

            # Create dummy components for None inputs
            dummy_text = gr.Textbox(visible=False)
            dummy_audio = gr.Audio(visible=False)

            # Event handlers with proper component references
            txt.submit(
                self.process_with_audio,
                inputs=[txt, dummy_audio, source_language, target_language, chatbot, enable_audio],
                outputs=[chatbot, audio_output]
            )

            mic_audio.stop_recording(
                self.process_with_audio,
                inputs=[dummy_text, mic_audio, source_language, target_language, chatbot, enable_audio],
                outputs=[chatbot, audio_output]
            )

            upload_audio.change(
                self.process_with_audio,
                inputs=[dummy_text, upload_audio, source_language, target_language, chatbot, enable_audio],
                outputs=[chatbot, audio_output]
            )

            enable_audio.change(
                lambda x: gr.update(visible=x),
                inputs=enable_audio,
                outputs=audio_output
            )

            clear.click(
                clear_fn,
                inputs=None,
                outputs=[chatbot, audio_output]
            )

            gr.Markdown("---")
            gr.Markdown("## Book Appointment For Aadhaar Enrollment / Check Aadhar Status")
            
            with gr.Row():
                gr.Button(
                    "Book Appointment / Check Aadhaar Status",
                    link="https://aadharappointmentstub-sfwbp53byxdbpsxhwexvpd.streamlit.app/"
                )

            return interface

def main():
    chat_app = AadhaarChatInterface()
    interface = chat_app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=4
    )

if __name__ == "__main__":
    main()