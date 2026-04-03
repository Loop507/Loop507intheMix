import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random
import shutil

# --- Configurazione FFmpeg per il server ---
# Questo blocco aiuta Pydub a trovare FFmpeg su Streamlit Cloud
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    AudioSegment.converter = ffmpeg_bin
else:
    st.error("ATTENZIONE: FFmpeg non trovato. Assicurati di avere 'ffmpeg' nel file packages.txt")

# --- Funzioni di analisi e manipolazione audio ---

@st.cache_data
def analyze_track_for_slicing(audio_file_object):
    """
    Analizza un brano e restituisce l'audio e i dati necessari per lo slicing.
    """
    audio_file_object.seek(0)
    
    # Usiamo un file temporaneo per permettere a librosa di leggere il file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        try:
            # Convertiamo l'input in un formato che librosa capisce sicuramente
            audio = AudioSegment.from_file(audio_file_object)
            audio.export(tmp_wav_file.name, format="wav")
            tmp_path = tmp_wav_file.name

            y, sr = librosa.load(tmp_path, sr=None)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Gestione caso in cui il tempo non venga rilevato (array o float)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 0
            
            duration = librosa.get_duration(y=y, sr=sr)
            return y, sr, onset_frames, tempo, duration
        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            return None, None, None, 0, 0
        finally:
            if os.path.exists(tmp_wav_file.name):
                os.remove(tmp_path)

def get_beat_segments(y, sr, tempo, num_beats_per_segment):
    """Divide il brano in segmenti basati sulle battute."""
    if tempo <= 0:
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

def combine_segments(segments):
    """Unisce i segmenti garantendo che abbiano lo stesso sample rate."""
    if not segments:
        return np.array([]), None
    
    target_sr = segments[0][1]
    resampled_segments = []
    for seg, sr_orig in segments:
        if sr_orig != target_sr:
            resampled_segments.append(librosa.resample(y=seg, orig_sr=sr_orig, target_sr=target_sr))
        else:
            resampled_segments.append(seg)

    combined_audio = np.concatenate(resampled_segments)
    return combined_audio, target_sr

def export_audio(y, sr):
    """Esporta l'audio in un buffer MP3 pronto per il download o l'ascolto."""
    if len(y) == 0 or sr is None:
        return None
        
    buffer = BytesIO()
    # Normalizzazione e conversione in 16-bit PCM
    y_norm = np.int16(y / np.max(np.abs(y)) * 32767) if np.max(np.abs(y)) > 0 else y.astype(np.int16)
    
    audio_segment = AudioSegment(
        y_norm.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# --- Interfaccia utente con Streamlit ---
st.set_page_config(page_title="Loop507 Mix", layout="wide")
st.title("Loop507 in the Mix: Decomposizione e Ricomposizione")

# Inizializzazione Deck
deck_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for key in deck_keys:
    if f'deck_{key}' not in st.session_state:
        st.session_state[f'deck_{key}'] = {'file_name': None, 'y': None, 'sr': None, 'tempo': None}

if 'decomposed_manual_segments' not in st.session_state:
    st.session_state.decomposed_manual_segments = []

# Layout a Griglia per i Deck
for row in [deck_keys[:4], deck_keys[4:]]:
    cols = st.columns(4)
    for i, key in enumerate(row):
        with cols[i]:
            st.subheader(f"Deck {key.upper()}")
            uploaded_file = st.file_uploader(f"Carica {key.upper()}", type=["mp3", "wav"], key=f"up_{key}")
            if uploaded_file:
                # Carichiamo l'analisi solo se il file è cambiato
                if st.session_state[f'deck_{key}']['file_name'] != uploaded_file.name:
                    with st.spinner('Analisi...'):
                        y, sr, onsets, tempo, duration = analyze_track_for_slicing(uploaded_file)
                        if y is not None:
                            st.session_state[f'deck_{key}'] = {
                                'file_name': uploaded_file.name, 
                                'y': y, 'sr': sr, 'tempo': tempo
                            }
                            st.success(f"{tempo:.0f} BPM rilevati")
                st.audio(uploaded_file)

# --- Controlli Ricomposizione ---
st.sidebar.header("🎛️ Mixer Control")
active_decks = [(k.upper(), st.session_state[f'deck_{k}']) for k in deck_keys if st.session_state[f'deck_{k}']['y'] is not None]

if active_decks:
    num_beats = st.sidebar.selectbox("Battute per segmento:", [1, 2, 4, 8], index=2)
    
    if st.sidebar.button("🔨 Decomponi Brani"):
        st.session_state.decomposed_manual_segments = []
        for name, data in active_decks:
            segs = get_beat_segments(data['y'], data['sr'], data['tempo'], num_beats)
            for i, s in enumerate(segs):
                st.session_state.decomposed_manual_segments.append({'source': name, 'segment': s, 'sr': data['sr']})
        st.sidebar.toast(f"Creati {len(st.session_state.decomposed_manual_segments)} segmenti!")

    mode = st.sidebar.radio("Modalità Mix:", ["Casuale", "Manuale"])

    if mode == "Casuale" and st.session_state.decomposed_manual_segments:
        dur = st.sidebar.slider("Durata Mix (secondi)", 10, 120, 30)
        if st.sidebar.button("Genera Mix Magico"):
            with st.spinner("Miscelando..."):
                # Calcolo approssimativo segmenti necessari
                first_sr = st.session_state.decomposed_manual_segments[0]['sr']
                total_samples_needed = dur * first_sr
                current_samples = 0
                chosen = []
                
                while current_samples < total_samples_needed:
                    s = random.choice(st.session_state.decomposed_manual_segments)
                    chosen.append((s['segment'], s['sr']))
                    current_samples += len(s['segment'])
                
                combined, final_sr = combine_segments(chosen)
                out = export_audio(combined, final_sr)
                if out:
                    st.divider()
                    st.subheader("✨ Il tuo nuovo Mix Casuale")
                    st.audio(out, format="audio/mp3")
                    st.download_button("Scarica Mix", out, "mix_loop507.mp3")

    elif mode == "Manuale" and active_decks:
        selected = st.sidebar.multiselect("Quali deck includere?", [d[0] for d in active_decks])
        if st.sidebar.button("Crea Mix Manuale") and selected:
            # Prende i segmenti solo dai deck selezionati
            manual_chosen = [(s['segment'], s['sr']) for s in st.session_state.decomposed_manual_segments if s['source'] in selected]
            if manual_chosen:
                combined, final_sr = combine_segments(manual_chosen)
                out = export_audio(combined, final_sr)
                st.audio(out)
else:
    st.info("Carica almeno un brano per iniziare a mixare!")
