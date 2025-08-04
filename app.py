import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random
import shutil

# --- Funzioni di analisi e manipolazione audio ---

@st.cache_data
def analyze_track_for_slicing(audio_file_object):
    """
    Analizza un brano e restituisce l'audio e i dati necessari per lo slicing.
    """
    audio_file_object.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        audio = AudioSegment.from_file(audio_file_object)
        audio.export(tmp_wav_file.name, format="wav")
        tmp_path = tmp_wav_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return y, sr, onset_frames, tempo
    finally:
        os.remove(tmp_path)

def get_beat_segments(y, sr, tempo, num_beats_per_segment):
    """
    Divide il brano in segmenti di N battute.
    """
    if tempo == 0:
        return []

    samples_per_beat = sr * 60 / tempo
    
    segments = []
    current_sample = 0
    while current_sample < len(y):
        end_sample = int(current_sample + samples_per_beat * num_beats_per_segment)
        if end_sample > len(y):
            end_sample = len(y)
        
        segment = y[current_sample:end_sample]
        if len(segment) > 0:
            segments.append(segment)
        
        current_sample = end_sample
    return segments

def combine_segments(segments, target_sr):
    """
    Combina una lista di segmenti in un unico brano, riscampionando se necessario.
    """
    if not segments:
        return np.array([])
    
    resampled_segments = []
    for seg, sr_orig in segments:
        if sr_orig != target_sr:
            resampled_segments.append(librosa.resample(y=seg, orig_sr=sr_orig, target_sr=target_sr))
        else:
            resampled_segments.append(seg)

    combined_audio = np.concatenate(resampled_segments)
    return combined_audio

def export_audio(y, sr):
    """
    Esporta l'audio processato in un buffer MP3.
    """
    if len(y) == 0 or sr is None:
        return None
        
    buffer = BytesIO()
    audio_segment = AudioSegment(
        (y * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# --- Interfaccia utente con Streamlit ---
st.title("Loop507 in the Mix: Decomposizione e Ricomposizione")
st.write("Carica due brani, decomponili in segmenti e ricomponili in un modo unico!")

if 'deck_a' not in st.session_state:
    st.session_state.deck_a = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_b' not in st.session_state:
    st.session_state.deck_b = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'combined_segments' not in st.session_state:
    st.session_state.combined_segments = []

col1, col2 = st.columns(2)

with col1:
    st.header("Deck A")
    uploaded_file_a = st.file_uploader("Carica Brano A", type=["mp3", "wav"], key="uploader_a")
    if uploaded_file_a:
        st.audio(uploaded_file_a, format='audio/mp3')
        if uploaded_file_a != st.session_state.deck_a['file']:
            with st.spinner('Analizzo il brano...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_a)
                st.session_state.deck_a = {'file': uploaded_file_a, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

with col2:
    st.header("Deck B")
    uploaded_file_b = st.file_uploader("Carica Brano B", type=["mp3", "wav"], key="uploader_b")
    if uploaded_file_b:
        st.audio(uploaded_file_b, format='audio/mp3')
        if uploaded_file_b != st.session_state.deck_b['file']:
            with st.spinner('Analizzo il brano...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_b)
                st.session_state.deck_b = {'file': uploaded_file_b, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

st.sidebar.header("Controlli Ricomposizione")

if st.session_state.deck_a['file'] and st.session_state.deck_b['file']:
    num_beats_per_segment = st.sidebar.selectbox("Segmenti da (battute):", [1, 2, 4, 8])
    if st.sidebar.button("Decomponi Brani"):
        st.session_state.combined_segments = []
        
        segments_a = get_beat_segments(
            st.session_state.deck_a['y'], st.session_state.deck_a['sr'], st.session_state.deck_a['tempo'], num_beats_per_segment)
        segments_b = get_beat_segments(
            st.session_state.deck_b['y'], st.session_state.deck_b['sr'], st.session_state.deck_b['tempo'], num_beats_per_segment)

        # Prepara una lista di tuple (segmento, sample_rate) per la combinazione
        segments_with_sr = [(seg, st.session_state.deck_a['sr']) for seg in segments_a] + \
                          [(seg, st.session_state.deck_b['sr']) for seg in segments_b]
        
        st.session_state.combined_segments = segments_with_sr
        st.sidebar.success(f"Brani decomposti in {len(st.session_state.combined_segments)} segmenti.")

if st.session_state.combined_segments:
    st.sidebar.subheader("Ricomponi il Brano")
    
    if st.sidebar.button("Shuffle (Riordina)"):
        random.shuffle(st.session_state.combined_segments)
        st.sidebar.success("Segmenti riordinati in modo casuale!")
    
    if st.sidebar.button("Crea Brano Ricomposto"):
        with st.spinner('Creazione del brano ricomposto...'):
            if st.session_state.deck_a['sr'] is not None:
                target_sr = st.session_state.deck_a['sr']
                combined_audio = combine_segments(st.session_state.combined_segments, target_sr)
                processed_audio_buffer = export_audio(combined_audio, target_sr)
            else:
                processed_audio_buffer = None

        if processed_audio_buffer:
            st.subheader("Brano Ricomposto")
            st.audio(processed_audio_buffer, format="audio/mp3")
            st.download_button("Scarica Brano Ricomposto", data=processed_audio_buffer, file_name="ricomposto.mp3", mime="audio/mp3")
        else:
            st.error("Impossibile creare il brano ricomposto. Carica prima i brani e decomponili.")
