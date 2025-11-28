
import os
import sys
import json
import time
import tempfile
import traceback
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import deque
import numpy as np
import sounddevice as sd
import librosa
import torch
import gradio as gr
from scipy.io.wavfile import write as write_wav
from midiutil import MIDIFile
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseSettings, Field

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config(BaseSettings):
    """Application configuration"""
    SR: int = Field(16000, env="SR")
    CHUNK_SIZE: int = Field(1024, env="CHUNK_SIZE")
    BUFFER_SECONDS: int = Field(10, env="BUFFER_SECONDS")
    FART_THRESHOLD: float = Field(0.7, env="FART_THRESHOLD")
    MIN_FART_DURATION: float = Field(0.1, env="MIN_FART_DURATION")
    MAX_FART_DURATION: float = Field(3.0, env="MAX_FART_DURATION")

    # Cartman LLM Configuration
    CARTMAN_TEMPERATURE: float = Field(0.9, env="CARTMAN_TEMPERATURE")

    # Audio processing
    PRE_TRIGGER_SECONDS: float = Field(1.0, env="PRE_TRIGGER_SECONDS")
    POST_TRIGGER_SECONDS: float = Field(2.0, env="POST_TRIGGER_SECONDS")

    class Config:
        env_file = ".env"

cfg = Config()

# ============================================================================
# AUDIO BUFFER & CONTINUOUS CAPTURE
# ============================================================================

class CircularAudioBuffer:
    """Ring buffer for continuous audio recording"""
    def __init__(self, sample_rate: int, buffer_seconds: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.available_samples = 0

    def write(self, data: np.ndarray):
        """Write new audio data to buffer"""
        n_samples = len(data)

        if n_samples > self.buffer_size:
            data = data[-self.buffer_size:]
            n_samples = self.buffer_size

        if self.write_pos + n_samples <= self.buffer_size:
            self.buffer[self.write_pos:self.write_pos + n_samples] = data
        else:
            remaining = self.buffer_size - self.write_pos
            self.buffer[self.write_pos:] = data[:remaining]
            self.buffer[:n_samples - remaining] = data[remaining:]

        self.write_pos = (self.write_pos + n_samples) % self.buffer_size
        self.available_samples = min(self.available_samples + n_samples, self.buffer_size)

    def read_recent(self, seconds: float) -> np.ndarray:
        """Read the most recent audio data"""
        samples_needed = int(seconds * self.sample_rate)
        samples_needed = min(samples_needed, self.available_samples)

        start_pos = (self.write_pos - samples_needed) % self.buffer_size

        if start_pos + samples_needed <= self.buffer_size:
            return self.buffer[start_pos:start_pos + samples_needed]
        else:
            first_part = self.buffer[start_pos:]
            remaining = samples_needed - len(first_part)
            return np.concatenate([first_part, self.buffer[:remaining]])

# ============================================================================
# ML MODEL FOR FART DETECTION
# ============================================================================

# Load the audio classification model and feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
audio_model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")

# ============================================================================
# CARTMAN LLM COMMENTARY SYSTEM
# ============================================================================

class CartmanCommentary:
    """South Park's Eric Cartman style commentary generator"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    def generate_commentary(self, features: Dict[str, Any], event_type: str = "fart") -> str:
        """Generate Cartman-style commentary on detected events"""

        prompt = f"The following is a conversation with Cartman from South Park. A fart was just detected. Cartman replies:"

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.chat_model.generate(inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
            return response
        except Exception as e:
            # Fallback responses in Cartman style
            fallbacks = [
                "Respect my authoritah! That was definitely a premium fart!",
                "Screw you guys, I'm going home! But first, let me appreciate this magnificent blast!",
                "You will respect my authoritah! That fart deserves its own NFT!",
                "Whatever! I'm so much cooler than Kyle! This fart proves it!",
                "That's it! This calls for cheesy poofs and premium fart tokens!"
            ]
            return np.random.choice(fallbacks)

# ============================================================================
# CONTINUOUS MONITORING SYSTEM
# ============================================================================

class FartResearchLab:
    """Main class for continuous fart research and analysis"""

    def __init__(self):
        self.audio_buffer = CircularAudioBuffer(
            sample_rate=cfg.SR,
            buffer_seconds=cfg.BUFFER_SECONDS
        )
        self.is_monitoring = False
        self.detection_count = 0
        self.event_history = deque(maxlen=100)
        self.last_detection_time = 0
        self.cartman = CartmanCommentary()

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for continuous audio input"""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono and add to buffer
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        self.audio_buffer.write(audio_data)

        # Real-time analysis on recent audio
        recent_audio = self.audio_buffer.read_recent(1.0)  # Analyze last second
        if len(recent_audio) > 0:
            self.analyze_realtime_audio(recent_audio)

    def analyze_realtime_audio(self, audio_data: np.ndarray):
        """Analyze audio chunk for fart detection"""
        try:
            features = self.extract_audio_features(audio_data)
            confidence = self.predict_fart(audio_data)
            features['confidence'] = confidence

            # Check if fart is detected
            if (confidence > cfg.FART_THRESHOLD and
                features['duration'] >= cfg.MIN_FART_DURATION and
                features['duration'] <= cfg.MAX_FART_DURATION and
                time.time() - self.last_detection_time > 2.0):  # Debouncing

                self.handle_fart_detection(features)

        except Exception as e:
            print(f"Analysis error: {e}")

    def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        features = {
            'rms_mean': float(np.sqrt(np.mean(audio_data**2))),
            'rms_std': float(np.std(audio_data)),
            'zcr_mean': float(librosa.feature.zero_crossing_rate(audio_data)[0].mean()),
            'duration': len(audio_data) / cfg.SR,
            'timestamp': time.time()
        }

        # Spectral features
        if len(audio_data) > 0:
            S = np.abs(librosa.stft(audio_data))
            features.update({
                'spectral_centroid_mean': float(librosa.feature.spectral_centroid(S=S)[0].mean()),
                'spectral_bandwidth_mean': float(librosa.feature.spectral_bandwidth(S=S)[0].mean()),
            })

        return features

    def predict_fart(self, audio_data: np.ndarray) -> float:
        """Use pre-trained model to predict fart probability"""
        try:
            inputs = audio_feature_extractor(audio_data, sampling_rate=cfg.SR, return_tensors="pt")
            with torch.no_grad():
                logits = audio_model(**inputs).logits

            # Get the softmax probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Find the index of the "Fart" class
            fart_class_index = -1
            for i, label in audio_model.config.id2label.items():
                if label == "Fart":
                    fart_class_index = i
                    break

            if fart_class_index != -1:
                return probabilities[0][fart_class_index].item()
            else:
                return 0.0

        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    def handle_fart_detection(self, features: Dict[str, Any]):
        """Handle detected fart event"""
        self.detection_count += 1
        self.last_detection_time = time.time()

        # Capture the full audio event with pre/post trigger
        full_audio = self.capture_full_event()

        # Generate Cartman commentary
        commentary = self.cartman.generate_commentary(features, "fart")

        # Create event record
        event = {
            'id': self.detection_count,
            'timestamp': time.time(),
            'features': features,
            'commentary': commentary,
            'audio_data': full_audio.tolist() if full_audio is not None else []
        }

        self.event_history.append(event)

        # Print to console (in production, this would update UI)
        print(f"\n🎵 FART DETECTED! #{self.detection_count}")
        print(f"   Confidence: {features['confidence']:.1%}")
        print(f"   Duration: {features['duration']:.2f}s")
        print(f"   🗣️ Cartman: {commentary}")
        print("-" * 50)

        # Generate MIDI in background
        if full_audio is not None:
            threading.Thread(target=self.process_event, args=(event,)).start()

    def capture_full_event(self) -> Optional[np.ndarray]:
        """Capture full audio event with pre and post trigger"""
        try:
            total_seconds = cfg.PRE_TRIGGER_SECONDS + cfg.POST_TRIGGER_SECONDS
            return self.audio_buffer.read_recent(total_seconds)
        except Exception as e:
            print(f"Event capture error: {e}")
            return None

    def process_event(self, event: Dict[str, Any]):
        """Process event: generate MIDI"""
        try:
            # Save audio to temporary file
            temp_audio = tempfile.mktemp(suffix=".wav")
            audio_data = np.array(event['audio_data'], dtype=np.float32)
            write_wav(temp_audio, cfg.SR, (audio_data * 32767).astype(np.int16))

            # Generate MIDI
            midi_path = self.audio_to_midi(temp_audio)

            # Cleanup
            os.unlink(temp_audio)
            if midi_path and os.path.exists(midi_path):
                os.unlink(midi_path)

        except Exception as e:
            print(f"Event processing error: {e}")

    def audio_to_midi(self, audio_path: str) -> str:
        """Convert audio to MIDI using onset detection"""
        try:
            y, sr = librosa.load(audio_path, sr=cfg.SR)

            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Create MIDI
            midi = MIDIFile(1)
            midi.addTempo(0, 0, 120)

            for i, onset_time in enumerate(onset_times):
                # Simple pitch mapping based on timing
                pitch = 60 + (i % 24)  # C4 to C6 range
                midi.addNote(0, 0, pitch, onset_time, 0.5, 100)

            midi_path = tempfile.mktemp(suffix=".mid")
            with open(midi_path, "wb") as f:
                midi.writeFile(f)

            return midi_path
        except Exception as e:
            print(f"MIDI generation error: {e}")
            return None

    def start_monitoring(self):
        """Start continuous audio monitoring"""
        if self.is_monitoring:
            return

        print("🎵 Starting Continuous Fart Research Lab...")
        print("🔊 Listening for audio events...")
        print("💬 Cartman commentary activated!")
        print("-" * 50)

        self.is_monitoring = True

        try:
            self.stream = sd.InputStream(
                samplerate=cfg.SR,
                channels=1,
                dtype='float32',
                blocksize=cfg.CHUNK_SIZE,
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self.is_monitoring = False

    def stop_monitoring(self):
        """Stop audio monitoring"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_monitoring = False
        print("🛑 Fart Research Lab stopped")

# ============================================================================
# GRADIO UI WITH REAL-TIME MONITORING
# ============================================================================

class FartResearchUI:
    """Gradio UI for the fart research lab"""

    def __init__(self):
        self.lab = FartResearchLab()
        self.setup_ui()

    def setup_ui(self):
        """Setup Gradio interface"""

        with gr.Blocks(theme=gr.themes.Soft(), title="🎵 Cartman Fart Research Lab") as demo:
            gr.Markdown("""
            # 🎵 Cartman Fart Research Lab
            ## *"Respect My Authoritah of Flatulence Analysis!"*
            Continuous audio monitoring with real-time fart detection and Cartman commentary
            """)

            with gr.Row():
                with gr.Column():
                    start_btn = gr.Button("🚀 Start Monitoring", variant="primary")
                    stop_btn = gr.Button("🛑 Stop Monitoring", variant="secondary")
                    status_text = gr.Textbox(label="Status", value="Ready to start...", interactive=False)

                with gr.Column():
                    detection_count = gr.Number(label="Total Detections", value=0, precision=0)
                    confidence_display = gr.Number(label="Last Confidence", value=0.0)

            with gr.Row():
                event_log = gr.Textbox(
                    label="🎭 Cartman Commentary & Event Log",
                    lines=10,
                    max_lines=15,
                    placeholder="Events and commentary will appear here...",
                    show_copy_button=True
                )

            with gr.Row():
                recent_events = gr.JSON(
                    label="📊 Recent Events",
                    value=[]
                )

            # Event handlers
            start_btn.click(
                fn=self.start_monitoring,
                outputs=[status_text, detection_count]
            )

            stop_btn.click(
                fn=self.stop_monitoring,
                outputs=[status_text]
            )

            # Periodic updates
            demo.load(
                fn=self.update_display,
                outputs=[event_log, recent_events, detection_count, confidence_display],
                every=1.0
            )

            self.demo = demo

    def start_monitoring(self):
        """Start the monitoring system"""
        self.lab.start_monitoring()
        return "🎵 Monitoring active - Listening for farts...", self.lab.detection_count

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.lab.stop_monitoring()
        return "🛑 Monitoring stopped"

    def update_display(self):
        """Update UI with latest events"""
        # Convert event history to display format
        events_list = list(self.lab.event_history)
        recent_events = events_list[-5:]  # Last 5 events

        # Create event log text
        log_text = ""
        for event in reversed(events_list[-10:]):  # Last 10 events in reverse
            timestamp = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
            log_text += f"[{timestamp}] 💨 Fart #{event['id']}\n"
            log_text += f"   Confidence: {event['features']['confidence']:.1%}\n"
            log_text += f"   🗣️ {event['commentary']}\n"
            log_text += "-" * 40 + "\n"

        # Get latest confidence
        latest_confidence = 0.0
        if events_list:
            latest_confidence = events_list[-1]['features']['confidence']

        return log_text, recent_events, self.lab.detection_count, latest_confidence

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        return self.demo.launch(**kwargs)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create and launch the UI
    ui = FartResearchUI()

    print("🎵 Starting Cartman Fart Research Lab...")
    print("💬 Cartman: 'You will respect my authoritah of fart analysis!'")
    print("🌐 UI will open at http://localhost:7860")

    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
