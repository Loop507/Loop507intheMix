import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random
import shutil
from datetime import datetime

# --- Configurazione FFmpeg ---
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    AudioSegment.converter = ffmpeg_bin
else:
    st.error("ATTENZIONE: FFmpeg non trovato. Verifica l'installazione sul sistema.")

# --- Funzioni di Analisi e Taglio ---

@st.cache_data
def analyze_track(audio_file_object):
    audio_file_object.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio = AudioSegment.from_file(audio_file_object)
        audio.export(tmp.name, format="wav")
        y, sr = librosa.load(tmp.name, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        return y, sr, tempo_val

def get_beat_segments(y, sr, tempo, num_beats):
    if tempo <= 0: return []
    samples_per_beat = sr * 60 / tempo
    segment_length = int(samples_per_beat * num_beats)
    return [y[i : i + segment_length] for i in range(0, len(y), segment_length) if (i + segment_length) <= len(y)]

def get_random_segments(y, sr, min_dur, max_dur):
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
    if len(y) == 0: return None
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

if 'decks' not in st.session_state:
    st.session_state.decks = {k: {'y': None, 'sr': None, 'tempo': 0.0, 'name': None} for k in 'abcdefgh'}
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'audio_report' not in st.session_state:
    st.session_state.audio_report = ""

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

st.sidebar.header("🎛️ Pannello di Controllo")
active_decks = {k: v for k, v in st.session_state.decks.items() if v['y'] is not None}

if active_decks:
    st.sidebar.subheader("1. Scegli Stile di Taglio")
    tipo_taglio = st.sidebar.radio("Modalità:", ["Ritmo Musicale (BPM)", "Frenesia Casuale (Secondi)"])

    if tipo_taglio == "Ritmo Musicale (BPM)":
        beats = st.sidebar.selectbox("Battute per segmento:", [0.5, 1, 2, 4, 8], index=2)
        if st.sidebar.button("🔨 Decomponi a Tempo"):
            st.session_state.segments = []
            for k, d in active_decks.items():
                segs = get_beat_segments(d['y'], d['sr'], d['tempo'], beats)
                for s in segs: st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            st.sidebar.success(f"Creati {len(st.session_state.segments)} pezzi musicali!")
    else:
        range_sec = st.sidebar.slider("Range durata (sec):", 0.05, 2.0, (0.8, 1.2), step=0.05)
        if st.sidebar.button("🌪️ Frulla Audio (Glitch)"):
            st.session_state.segments = []
            for k, d in active_decks.items():
                segs = get_random_segments(d['y'], d['sr'], range_sec[0], range_sec[1])
                for s in segs: st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            st.sidebar.warning(f"Creati {len(st.session_state.segments)} micro-pezzi!")

    if st.session_state.segments:
        st.sidebar.divider()
        st.sidebar.subheader("2. Esportazione")
        durata_mix = st.sidebar.number_input("Durata Mix Finale (sec):", 10, 600, 60)
        
        if st.sidebar.button("🚀 GENERA MIX FINALE"):
            with st.spinner("Rimescolando il mazzo..."):
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

                # --- GENERAZIONE REPORT BRANDIZZATO ---
                ts_audio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.audio_report = f"""
╔════════════════════════════════════════════════════════════════╗
  HYPER-MIXER v2.0 - AUDIO RECONSTRUCTION LOG
  Generated on: {ts_audio}
╚════════════════════════════════════════════════════════════════╝

[AUDIO_RECONSTRUCTION_LOG] // VOL_01 // MP3 // 320kbps

:: ENGINE: hyper_mixer_loop507 [v2.0]
:: ANALISI: Beat Tracking (Librosa) / RMS Envelope
:: STILE: Audio-Glitch / Granular Synthesis
:: PROCESSO: Shuffling Ricorsivo / Cross-Deck Fragmentation

"Audio-Data fragment: Il ritmo è solo una variabile manipolata dal caos."

---
> TECHNICAL LOG SHEET:
* Active Decks: {len(active_decks)} sorgenti caricate
* Segments Pool: {len(st.session_state.segments)} campioni estratti
* Modalità: {tipo_taglio}
* Campionamento: {ref_sr} Hz / Mono Downmix
* Output Duration: {durata_mix}s

> Regia e Algoritmo: Loop507

#Loop507 #AudioGlitch #SoundDesign #GranularSynthesis #ExperimentalMusic 
#AudioDecomposition #NoiseArt #SignalCorruption #RecursiveCollapse
"""
                st.session_state.mix_ready = out

if 'mix_ready' in st.session_state:
    st.divider()
    st.subheader("🎵 Risultato del Mix")
    st.audio(st.session_state.mix_ready)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("📥 Scarica Mix MP3", st.session_state.mix_ready, "loop507_custom_mix.mp3", use_container_width=True)
    with col_d2:
        st.download_button("📄 Scarica Report Audio", st.session_state.audio_report, "audio_report.txt", use_container_width=True)

st.markdown("---")
st.caption("Loop507 Hyper-Mixer | Modalità Glitch & BPM attiva")
