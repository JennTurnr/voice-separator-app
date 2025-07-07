# app.py
import streamlit as st
import torchaudio
import soundfile as sf
from asteroid.models import ConvTasNet
import tempfile
import os
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np

@st.cache_resource
def load_model():
   return ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
model = load_model()

st.title("🎙️ Voice Separator Web App")
st.write("Upload a WAV/MP3 file or record your voice. We'll separate each speaker into their own audio track.")

uploaded_file = st.file_uploader("📤 Upload WAV or MP3", type=["wav", "mp3"])
temp_path = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        if uploaded_file.type == "audio/mpeg":
            audio = AudioSegment.from_file(uploaded_file, format="mp3")
            audio.export(tmp_file.name, format="wav")
        else:
            tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

class VoiceRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

st.subheader("🎤 Or Record Audio")
ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
    audio_processor_factory=VoiceRecorder,
)

if ctx.audio_processor:
    if st.button("🔄 Convert Recording to File"):
        audio_data = np.concatenate(ctx.audio_processor.frames, axis=1)
        out_path = os.path.join(tempfile.gettempdir(), "recorded.wav")
        sf.write(out_path, audio_data.T, 16000)
        temp_path = out_path
        st.audio(out_path, format="audio/wav")

if temp_path:
    st.write("Processing... please wait.")
    wav, sr = torchaudio.load(temp_path)
    est_sources = model.separate(wav)

    speaker1_path = os.path.join(tempfile.gettempdir(), "speaker1.wav")
    speaker2_path = os.path.join(tempfile.gettempdir(), "speaker2.wav")

    sf.write(speaker1_path, est_sources[0].detach().cpu().numpy().T, sr)
    sf.write(speaker2_path, est_sources[1].detach().cpu().numpy().T, sr)

    st.success("✅ Voices separated!")
    st.audio(speaker1_path, format="audio/wav")
    st.audio(speaker2_path, format="audio/wav")
    st.download_button("⬇️ Download Speaker 1", open(speaker1_path, "rb"), file_name="speaker1.wav")
    st.download_button("⬇️ Download Speaker 2", open(speaker2_path, "rb"), file_name="speaker2.wav")