import gradio as gr
import speech_recognition as sr
from pydub import AudioSegment
import os
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import tempfile
import librosa

# Load the audio classification model and feature extractor
audio_feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
audio_model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")

# Load the conversational model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def get_cartman_response(text):
    # Craft a prompt to encourage Cartman-like responses
    prompt = f"The following is a conversation with Cartman from South Park. The user says: '{text}'. Cartman replies:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = chat_model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
        return fp.name

def analyze_audio(audio_path):
    if audio_path is None:
        return "You didn't record anything, you stupid idiot.", None

    try:
        # Convert audio file to WAV format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sound = AudioSegment.from_file(audio_path)
            sound.export(fp.name, format="wav")
            wav_path = fp.name
    except Exception as e:
        return f"Error converting audio file: {e}", None

    try:
        # Fart detection
        audio, sr = librosa.load(wav_path, sr=16000)
        inputs = audio_feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            logits = audio_model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_label = audio_model.config.id2label[predicted_class_ids]

        if predicted_label == "Fart":
            response = "Cartman says: 'Sweet, a fart! That one sounded juicy.'"
            audio_response = text_to_speech(response)
            return response, audio_response

        # Speech recognition
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            response = get_cartman_response(text)
            audio_response = text_to_speech(response)
    except sr.UnknownValueError:
        response = "Cartman says: 'I couldn't understand what you said. Was that a fart?'"
        audio_response = text_to_speech(response)
    except sr.RequestError as e:
        response = f"Could not request results from Google Speech Recognition service; {e}"
        audio_response = None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return response, audio_response

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # Fart Identifier with Cartman
            Record your farts or your voice and get a response from Cartman.
            """
        )
        with gr.Row():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your fart or speech")
        with gr.Row():
            analyze_button = gr.Button("Analyze")
        with gr.Row():
            output_text = gr.Textbox(label="Cartman's Response")
        with gr.Row():
            output_audio = gr.Audio(label="Cartman's Spoken Response")

        analyze_button.click(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[output_text, output_audio]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
