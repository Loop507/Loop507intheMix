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
    Analizza un brano per trovare i punti di attacco (onset) e restituisce
    l'audio, il sample rate e gli indici dei punti di attacco.
    """
    audio_file_object.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        audio = AudioSegment.from_file(audio_file_object)
        audio.export(tmp_wav_file.name, format="wav")
        tmp_path = tmp_wav_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
        
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        
        # Stima un tempo per definire la griglia ritmica, ma non lo mostriamo
        # L'importante è che la griglia sia coerente con il brano
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return y, sr, onset_frames, tempo
    
    finally:
        os.remove(tmp_path)


def create_loop(y, sr, tempo, num_beats):
    """
    Crea un loop di N battute.
    """
    if tempo == 0:
        return y

    samples_per_beat = sr * 60 / tempo
    
    start_sample = 0
    end_sample = int(start_sample + samples_per_beat * num_beats)
    
    if end_sample > len(y):
        end_sample = len(y)
    
    looped_audio = y[start_sample:end_sample]
    
    return looped_audio


def randomize_track(y, sr, onset_frames, tempo):
    """
    Taglia il brano in segmenti e li ricompone in ordine casuale.
    """
    if tempo == 0:
        return y
        
    frames_per_beat = sr * 60 / tempo
    
    segments = []
    for i in range(len(onset_frames) - 1):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i+1]
        
        if end_frame - start_frame >= frames_per_beat / 4:
            segments.append(y[start_frame:end_frame])

    if not segments:
        return y
    
    random.shuffle(segments)
    
    randomized_audio = np.concatenate(segments)
    
    return randomized_audio

def randomize_two_decks(y1, sr1, onsets1, tempo1, y2, sr2, onsets2, tempo2):
    """
    Mescola segmenti di due brani in modo casuale.
    """
    if tempo1 == 0 or tempo2 == 0:
        return None, None
        
    frames_per_beat1 = sr1 * 60 / tempo1
    frames_per_beat2 = sr2 * 60 / tempo2
    
    segments1 = []
    for i in range(len(onsets1) - 1):
        start_frame = onsets1[i]
        end_frame = onsets1[i+1]
        if end_frame - start_frame >= frames_per_beat1 / 4:
            segments1.append(y1[start_frame:end_frame])

    segments2 = []
    for i in range(len(onsets2) - 1):
        start_frame = onsets2[i]
        end_frame = onsets2[i+1]
        if end_frame - start_frame >= frames_per_beat2 / 4:
            segments2.append(y2[start_frame:end_frame])

    if not segments1 or not segments2:
        return None, None
    
    all_segments = segments1 + segments2
    random.shuffle(all_segments)
    
    target_sr = sr1
    
    resampled_segments2 = [librosa.resample(y=seg, orig_sr=sr2, target_sr=target_sr) for seg in segments2]
    
    all_segments = segments1 + resampled_segments2
    random.shuffle(all_segments)
        
    randomized_mashup = np.concatenate(all_segments)
    
    return randomized_mashup, target_sr


def export_audio(y, sr):
    """
    Esporta l'audio processato in un buffer MP3.
    """
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
st.title("Loop507 in the Mix: Beat Slicer")
st.write("Carica un brano o due e ricomponili in modo creativo!")

if 'deck_a' not in st.session_state:
    st.session_state.deck_a = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_b' not in st.session_state:
    st.session_state.deck_b = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}

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

st.sidebar.header("Controlli Slicing e Ricomposizione")

# Controlli per il Deck A
if st.session_state.deck_a['file']:
    st.sidebar.subheader("Brano A")
    num_beats_a = st.sidebar.selectbox("Loop (battute)", [1, 2, 3, 4, 8], key="loop_a")
    
    if st.sidebar.button("Crea Loop A", key="create_loop_a"):
        with st.spinner('Creo loop...'):
            looped_audio = create_loop(st.session_state.deck_a['y'], st.session_state.deck_a['sr'], st.session_state.deck_a['tempo'], num_beats_a)
            processed_audio_buffer = export_audio(looped_audio, st.session_state.deck_a['sr'])
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button("Scarica Loop A", data=processed_audio_buffer, file_name=f"loop_A_{num_beats_a}_beats.mp3", mime="audio/mp3")

    if not st.session_state.deck_b['file'] and st.sidebar.button("Crea Random A", key="random_a_solo"):
        with st.spinner('Ricomponendo il brano...'):
            randomized_audio = randomize_track(st.session_state.deck_a['y'], st.session_state.deck_a['sr'], st.session_state.deck_a['onsets'], st.session_state.deck_a['tempo'])
            processed_audio_buffer = export_audio(randomized_audio, st.session_state.deck_a['sr'])
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button("Scarica Random A", data=processed_audio_buffer, file_name=f"random_A.mp3", mime="audio/mp3")

st.sidebar.write("---")

# Controlli per il Deck B
if st.session_state.deck_b['file']:
    st.sidebar.subheader("Brano B")
    num_beats_b = st.sidebar.selectbox("Loop (battute)", [1, 2, 3, 4, 8], key="loop_b")
    
    if st.sidebar.button("Crea Loop B", key="create_loop_b"):
        with st.spinner('Creo loop...'):
            looped_audio = create_loop(st.session_state.deck_b['y'], st.session_state.deck_b['sr'], st.session_state.deck_b['tempo'], num_beats_b)
            processed_audio_buffer = export_audio(looped_audio, st.session_state.deck_b['sr'])
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button("Scarica Loop B", data=processed_audio_buffer, file_name=f"loop_B_{num_beats_b}_beats.mp3", mime="audio/mp3")

    if not st.session_state.deck_a['file'] and st.sidebar.button("Crea Random B", key="random_b_solo"):
        with st.spinner('Ricomponendo il brano...'):
            randomized_audio = randomize_track(st.session_state.deck_b['y'], st.session_state.deck_b['sr'], st.session_state.deck_b['onsets'], st.session_state.deck_b['tempo'])
            processed_audio_buffer = export_audio(randomized_audio, st.session_state.deck_b['sr'])
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button("Scarica Random B", data=processed_audio_buffer, file_name=f"random_B.mp3", mime="audio/mp3")

# Funzione per mescolare due brani
if st.session_state.deck_a['file'] and st.session_state.deck_b['file']:
    st.sidebar.subheader("Mashup a Due Deck")
    if st.sidebar.button("Crea Random Mix", key="random_mashup"):
        with st.spinner('Ricomponendo il mix...'):
            mixed_audio, mixed_sr = randomize_two_decks(
                st.session_state.deck_a['y'], st.session_state.deck_a['sr'], st.session_state.deck_a['onsets'], st.session_state.deck_a['tempo'],
                st.session_state.deck_b['y'], st.session_state.deck_b['sr'], st.session_state.deck_b['onsets'], st.session_state.deck_b['tempo']
            )
            if mixed_audio is not None and mixed_sr is not None:
                processed_audio_buffer = export_audio(mixed_audio, mixed_sr)
                st.audio(processed_audio_buffer, format="audio/mp3")
                st.download_button("Scarica Mix Random", data=processed_audio_buffer, file_name="random_mashup.mp3", mime="audio/mp3")
            else:
                st.error("Impossibile creare il mashup. Assicurati che entrambi i brani abbiano abbastanza punti di attacco.")
