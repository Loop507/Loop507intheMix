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
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    AudioSegment.converter = ffmpeg_bin
else:
    st.error("ATTENZIONE: FFmpeg non trovato. Verifica il file packages.txt")

# --- Funzioni di analisi e manipolazione audio ---

@st.cache_data
def analyze_track_for_slicing(audio_file_object):
    """Analizza il brano e calcola BPM e durata."""
    audio_file_object.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        try:
            audio = AudioSegment.from_file(audio_file_object)
            audio.export(tmp_wav_file.name, format="wav")
            tmp_path = tmp_wav_file.name

            y, sr = librosa.load(tmp_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Normalizzazione BPM
            if isinstance(tempo, np.ndarray):
                tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo_val = float(tempo)
            
            duration = librosa.get_duration(y=y, sr=sr)
            return y, sr, tempo_val, duration
        except Exception as e:
            st.error(f"Errore analisi: {e}")
            return None, None, 0.0, 0.0
        finally:
            if os.path.exists(tmp_wav_file.name):
                os.remove(tmp_wav_file.name)

def get_beat_segments(y, sr, tempo, num_beats_per_segment):
    """Divide l'audio in segmenti basati sul tempo musicale."""
    if tempo <= 0:
        return []
    samples_per_beat = sr * 60 / tempo
    segment_length = int(samples_per_beat * num_beats_per_segment)
    
    segments = []
    for i in range(0, len(y), segment_length):
        seg = y[i : i + segment_length]
        if len(seg) > 0:
            segments.append(seg)
    return segments

def combine_segments(segments_list):
    """Unisce i segmenti audio in un unico array."""
    if not segments_list:
        return np.array([]), None
    target_sr = segments_list[0][1]
    final_audio = []
    for seg, sr_orig in segments_list:
        if sr_orig != target_sr:
            seg = librosa.resample(y=seg, orig_sr=sr_orig, target_sr=target_sr)
        final_audio.append(seg)
    return np.concatenate(final_audio), target_sr

def export_audio(y, sr):
    """Trasforma l'array NumPy in un file MP3."""
    if len(y) == 0: return None
    max_val = np.max(np.abs(y))
    y_norm = (y / max_val) * 32767 if max_val > 0 else y
    buffer = BytesIO()
    audio_seg = AudioSegment(
        y_norm.astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    audio_seg.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# --- Interfaccia Utente ---
st.set_page_config(page_title="Loop507 Mixer", layout="wide")
st.title("🎧 Loop507 in the Mix")

# Inizializzazione Stato
deck_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
if 'decks' not in st.session_state:
    st.session_state.decks = {k: {'name': None, 'y': None, 'sr': None, 'tempo': 0.0} for k in deck_keys}
if 'segments' not in st.session_state:
    st.session_state.segments = []

# Griglia dei Deck (4x2)
for row_keys in [deck_keys[:4], deck_keys[4:]]:
    cols = st.columns(4)
    for i, k in enumerate(row_keys):
        with cols[i]:
            st.subheader(f"Deck {k.upper()}")
            up = st.file_uploader(f"Audio {k.upper()}", type=["mp3", "wav"], key=f"file_{k}")
            
            if up:
                # Se il file è nuovo, lo analizziamo
                if st.session_state.decks[k]['name'] != up.name:
                    with st.spinner("Analizzando..."):
                        y, sr, t_val, dur = analyze_track_for_slicing(up)
                        st.session_state.decks[k] = {'name': up.name, 'y': y, 'sr': sr, 'tempo': t_val}
                
                # Visualizziamo i dati recuperandoli dalla session_state (così non danno NameError)
                current_bpm = st.session_state.decks[k]['tempo']
                st.write(f"⏱️ {current_bpm:.1f} BPM")
                st.audio(up)

# Sidebar Comandi
st.sidebar.header("🎛️ Pannello di Controllo")
active_decks = {k: v for k, v in st.session_state.decks.items() if v['y'] is not None}

if active_decks:
    beats = st.sidebar.selectbox("Battute per segmento:", [1, 2, 4, 8], index=2)
    
    if st.sidebar.button("🔨 Decomponi Audio"):
        st.session_state.segments = []
        for k, data in active_decks.items():
            s_list = get_beat_segments(data['y'], data['sr'], data['tempo'], beats)
            for s in s_list:
                st.session_state.segments.append({'deck': k, 'audio': s, 'sr': data['sr']})
        st.toast(f"Creati {len(st.session_state.segments)} segmenti!")

    mode = st.sidebar.radio("Modalità Mix:", ["Casuale", "Selettiva"])

    if st.session_state.segments:
        if mode == "Casuale":
            sec = st.sidebar.number_input("Secondi di mix:", 10, 300, 30)
            if st.sidebar.button("Genera Mix Casuale"):
                with st.spinner("Mixing..."):
                    chosen = []
                    curr_len = 0
                    # Usiamo il sample rate del primo segmento come riferimento
                    ref_sr = st.session_state.segments[0]['sr']
                    target_samples = sec * ref_sr
                    
                    while curr_len < target_samples:
                        pick = random.choice(st.session_state.segments)
                        chosen.append((pick['audio'], pick['sr']))
                        curr_len += len(pick['audio'])
                    
                    mix, final_sr = combine_segments(chosen)
                    out = export_audio(mix, final_sr)
                    if out:
                        st.divider()
                        st.audio(out)
                        st.download_button("Scarica Mix", out, "mix_casuale.mp3")

        else: # Selettiva
            opts = [k.upper() for k in active_decks.keys()]
            to_include = st.sidebar.multiselect("Includi Deck:", opts)
            if st.sidebar.button("Crea Mix Selettivo") and to_include:
                chosen = [(s['audio'], s['sr']) for s in st.session_state.segments if s['deck'].upper() in to_include]
                if chosen:
                    mix, final_sr = combine_segments(chosen)
                    out = export_audio(mix, final_sr)
                    st.divider()
                    st.audio(out)
                    st.download_button("Scarica Mix Manuale", out, "mix_manuale.mp3")
else:
    st.sidebar.info("Carica brani nei Deck per iniziare.")
