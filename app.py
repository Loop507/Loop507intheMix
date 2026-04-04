import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random
import shutil

# --- Configurazione FFmpeg ---
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    AudioSegment.converter = ffmpeg_bin
else:
    st.error("ATTENZIONE: FFmpeg non trovato. Verifica l'installazione sul sistema.")

# --- Funzioni di Analisi e Taglio ---

@st.cache_data
def analyze_track(audio_file_object):
    """Analizza il brano per estrarre waveform, Sample Rate e BPM."""
    audio_file_object.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio = AudioSegment.from_file(audio_file_object)
        audio.export(tmp.name, format="wav")
        y, sr = librosa.load(tmp.name, sr=None)
        # Calcolo BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        return y, sr, tempo_val

def get_beat_segments(y, sr, tempo, num_beats):
    """Taglia l'audio seguendo il tempo musicale (BPM)."""
    if tempo <= 0: return []
    samples_per_beat = sr * 60 / tempo
    segment_length = int(samples_per_beat * num_beats)
    return [y[i : i + segment_length] for i in range(0, len(y), segment_length) if (i + segment_length) <= len(y)]

def get_random_segments(y, sr, min_dur, max_dur):
    """Taglia l'audio in pezzi casuali (Frenesia/Glitch)."""
    segments = []
    curr = 0
    while curr < len(y):
        dur = random.uniform(min_dur, max_dur)
        length = int(dur * sr)
        if curr + length > len(y): break
        segments.append(y[curr : curr + length])
        curr += length
    return segments

def export_audio(y, sr):
    """Esporta l'array numpy in un buffer MP3 pronto per il download."""
    if len(y) == 0: return None
    # Normalizzazione per evitare distorsioni
    max_val = np.max(np.abs(y))
    y_norm = (y / max_val * 32767).astype(np.int16) if max_val > 0 else y.astype(np.int16)
    buffer = BytesIO()
    audio_seg = AudioSegment(y_norm.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    audio_seg.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# --- Interfaccia Utente Streamlit ---
st.set_page_config(page_title="Loop507 Hyper-Mixer", layout="wide")
st.title("🎧 Loop507: Audio Shuffler & Glitcher")

# Stato dell'applicazione
if 'decks' not in st.session_state:
    st.session_state.decks = {k: {'y': None, 'sr': None, 'tempo': 0.0, 'name': None} for k in 'abcdefgh'}
if 'segments' not in st.session_state:
    st.session_state.segments = []

# Griglia dei Deck (Caricamento Audio)
deck_keys = list(st.session_state.decks.keys())
for row_idx in [0, 4]:
    cols = st.columns(4)
    for i, k in enumerate(deck_keys[row_idx : row_idx + 4]):
        with cols[i]:
            st.markdown(f"### Deck {k.upper()}")
            up = st.file_uploader(f"Carica {k.upper()}", type=["mp3", "wav"], key=f"up_{k}", label_visibility="collapsed")
            
            if up:
                if st.session_state.decks[k]['name'] != up.name:
                    with st.spinner(f"Analizzando {k.upper()}..."):
                        y, sr, t = analyze_track(up)
                        st.session_state.decks[k] = {'y': y, 'sr': sr, 'tempo': t, 'name': up.name}
                
                if st.session_state.decks[k]['y'] is not None:
                    st.success(f"{up.name}")
                    st.write(f"⏱️ **{st.session_state.decks[k]['tempo']:.1f} BPM**")
                    st.audio(up)

# Sidebar: Pannello di Controllo Doppia Modalità
st.sidebar.header("🎛️ Pannello di Controllo")
active_decks = {k: v for k, v in st.session_state.decks.items() if v['y'] is not None}

if active_decks:
    st.sidebar.subheader("1. Scegli Stile di Taglio")
    tipo_taglio = st.sidebar.radio(
        "Modalità:", 
        ["Ritmo Musicale (BPM)", "Frenesia Casuale (Secondi)"],
        help="BPM per remix coerenti, Secondi per effetti glitch estremi."
    )

    if tipo_taglio == "Ritmo Musicale (BPM)":
        beats = st.sidebar.selectbox("Battute per segmento:", [0.5, 1, 2, 4, 8], index=2)
        if st.sidebar.button("🔨 Decomponi a Tempo"):
            st.session_state.segments = []
            for k, d in active_decks.items():
                segs = get_beat_segments(d['y'], d['sr'], d['tempo'], beats)
                for s in segs: st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            st.sidebar.success(f"Creati {len(st.session_state.segments)} pezzi musicali!")

    else: # Frenesia Casuale (Secondi)
        st.sidebar.info("Imposta un range opposto al video per creare contrasto!")
        range_sec = st.sidebar.slider(
            "Range durata (sec):", 
            0.05, 2.0, (0.8, 1.2), 
            step=0.05,
            help="Tagli brevi (0.1-0.3) = Caos. Tagli lunghi (0.8-1.5) = Stabilità."
        )
        if st.sidebar.button("🌪️ Frulla Audio (Glitch)"):
            st.session_state.segments = []
            for k, d in active_decks.items():
                segs = get_random_segments(d['y'], d['sr'], range_sec[0], range_sec[1])
                for s in segs: st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            st.sidebar.warning(f"Creati {len(st.session_state.segments)} micro-pezzi!")

    # Sezione Generazione Mix
    if st.session_state.segments:
        st.sidebar.divider()
        st.sidebar.subheader("2. Esportazione")
        durata_mix = st.sidebar.number_input("Durata Mix Finale (sec):", 10, 600, 60)
        
        if st.sidebar.button("🚀 GENERA MIX FINALE"):
            with st.spinner("Rimescolando il mazzo..."):
                # Mescoliamo i segmenti per casualità totale
                all_segs = list(st.session_state.segments)
                random.shuffle(all_segs)
                
                chosen = []
                curr_samples = 0
                ref_sr = all_segs[0]['sr']
                target_samples = durata_mix * ref_sr
                
                while curr_samples < target_samples:
                    pick = random.choice(all_segs)
                    chosen.append(pick['audio'])
                    curr_samples += len(pick['audio'])
                
                final_y = np.concatenate(chosen)
                out = export_audio(final_y, ref_sr)
                
                st.divider()
                st.subheader("🎵 Risultato del Mix")
                st.audio(out)
                st.download_button("📥 Scarica Mix MP3", out, "loop507_custom_mix.mp3")
else:
    st.sidebar.info("Carica almeno un brano nei Deck per sbloccare i comandi.")

# Footer
st.markdown("---")
st.caption("Loop507 Hyper-Mixer | Modalità Glitch & BPM attiva")
