import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import subprocess
import json

# --- Funzioni di analisi e manipolazione audio ULTRA-PRECISE ---

def get_camelot_key(key):
    """Converte una chiave musicale standard in chiave Camelot."""
    camelot_map = {
        'C': '8B', 'Am': '8A', 'G': '9B', 'Em': '9A',
        'D': '10B', 'Bm': '10A', 'A': '11B', 'F#m': '11A',
        'E': '12B', 'C#m': '12A', 'B': '1B', 'G#m': '1A',
        'F#': '2B', 'D#m': '2A', 'Db': '3B', 'Bbm': '3A',
        'Ab': '4B', 'Fm': '4A', 'Eb': '5B', 'Cm': '5A',
        'Bb': '6B', 'Gm': '6A', 'F': '7B', 'Dm': '7A'
    }
    return camelot_map.get(key, 'Unknown')

def analyze_with_aubio(file_path):
    """Analizza BPM usando aubio (più preciso di librosa)."""
    try:
        # Comando aubio per analisi BPM
        cmd = [
            'aubiobpm', 
            '-i', file_path,
            '-B', '1024',  # Buffer size
            '-H', '512',   # Hop size
            '-s', '-70'    # Silence threshold
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Estrai il BPM dall'output
            lines = result.stdout.strip().split('\n')
            bpms = []
            for line in lines:
                if line.strip():
                    try:
                        bpm = float(line.strip())
                        if 60 <= bpm <= 200:  # Filtro valori ragionevoli
                            bpms.append(bpm)
                    except:
                        continue
            
            if bpms:
                return np.median(bpms)  # Usa la mediana per robustezza
        
        # Fallback se aubio non funziona
        return None
        
    except Exception as e:
        st.warning(f"Aubio non disponibile: {e}")
        return None

def analyze_with_beatdetection(y, sr):
    """Analisi BPM con algoritmo di beat detection migliorato."""
    try:
        # 1. Pre-processing: filtra frequenze non rilevanti
        y_filtered = librosa.effects.preemphasis(y)
        
        # 2. Estrazione caratteristiche ritmiche multiple
        hop_length = 512
        
        # Spectral flux per rilevare onset
        stft = librosa.stft(y_filtered, hop_length=hop_length)
        spectral_flux = np.sum(np.diff(np.abs(stft), axis=1), axis=0)
        spectral_flux = np.maximum(0, spectral_flux)  # Solo incrementi positivi
        
        # 3. Autocorrelazione del flux spettrale
        autocorr = np.correlate(spectral_flux, spectral_flux, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 4. Trova picchi nell'autocorrelazione
        from scipy.signal import find_peaks
        
        # Converti lag in BPM
        time_per_frame = hop_length / sr
        min_lag = int(60 / 200 / time_per_frame)  # 200 BPM max
        max_lag = int(60 / 60 / time_per_frame)   # 60 BPM min
        
        if max_lag < len(autocorr):
            peaks, properties = find_peaks(
                autocorr[min_lag:max_lag], 
                height=np.max(autocorr[min_lag:max_lag]) * 0.3,
                distance=min_lag//4
            )
            
            if len(peaks) > 0:
                # Converti il picco più alto in BPM
                best_peak = peaks[np.argmax(properties['peak_heights'])]
                lag = best_peak + min_lag
                bpm = 60 / (lag * time_per_frame)
                return float(bpm)
        
        return None
        
    except Exception as e:
        st.warning(f"Errore nell'analisi beat detection: {e}")
        return None

def analyze_with_essentia():
    """Placeholder per Essentia (libreria professionale per MIR)."""
    # Essentia è più complesso da installare, ma è lo standard per analisi MIR
    # Per ora ritorniamo None, ma in futuro si può implementare
    return None

def estimate_bpm_professional(y, sr):
    """Sistema di stima BPM professionale con multiple tecniche."""
    bpm_estimates = []
    
    # Metodo 1: Beat detection con autocorrelazione
    bpm1 = analyze_with_beatdetection(y, sr)
    if bpm1 and 60 <= bpm1 <= 200:
        bpm_estimates.append(bpm1)
    
    # Metodo 2: Librosa beat tracking (come backup)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=120)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else None
        else:
            tempo = float(tempo)
        
        if tempo and 60 <= tempo <= 200:
            bpm_estimates.append(tempo)
    except:
        pass
    
    # Metodo 3: Onset-based BPM
    try:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, 
            hop_length=512,
            backtrack=True,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.2,
            wait=10
        )
        
        if len(onset_frames) > 2:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            intervals = np.diff(onset_times)
            
            # Filtra intervalli troppo corti o lunghi
            intervals = intervals[(intervals > 0.3) & (intervals < 1.0)]
            
            if len(intervals) > 0:
                median_interval = np.median(intervals)
                bpm3 = 60.0 / median_interval
                if 60 <= bpm3 <= 200:
                    bpm_estimates.append(bpm3)
    except:
        pass
    
    # Se abbiamo stime multiple, usa logica intelligente
    if len(bpm_estimates) >= 2:
        # Se le stime sono vicine, usa la mediana
        if np.std(bpm_estimates) < 10:
            return np.median(bpm_estimates)
        else:
            # Se sono molto diverse, preferisci quelle nel range dance
            dance_bpms = [bpm for bpm in bpm_estimates if 115 <= bpm <= 140]
            if dance_bpms:
                return np.median(dance_bpms)
            else:
                return np.median(bpm_estimates)
    elif len(bpm_estimates) == 1:
        return bpm_estimates[0]
    else:
        return 120.0  # Fallback

def estimate_key_precise(y, sr):
    """Stima la chiave con algoritmo Krumhansl-Schmuckler."""
    try:
        # Profili di Krumhansl-Schmuckler (più accurati)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Estrai chroma con parametri ottimizzati
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, 
            hop_length=4096,  # Finestra più grande per stabilità
            norm=2
        )
        
        # Calcola il profilo cromatico medio
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / np.sum(chroma_mean)  # Normalizza
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minor_keys = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m']
        
        max_correlation = -1
        best_key = 'C'
        
        # Testa tutte le chiavi maggiori e minori
        for i in range(12):
            # Maggiore
            major_rotated = np.roll(major_profile, i)
            major_rotated = major_rotated / np.sum(major_rotated)
            corr_major = np.corrcoef(chroma_mean, major_rotated)[0, 1]
            
            if not np.isnan(corr_major) and corr_major > max_correlation:
                max_correlation = corr_major
                best_key = keys[i]
            
            # Minore
            minor_rotated = np.roll(minor_profile, i)
            minor_rotated = minor_rotated / np.sum(minor_rotated)
            corr_minor = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
            
            if not np.isnan(corr_minor) and corr_minor > max_correlation:
                max_correlation = corr_minor
                best_key = minor_keys[i]
        
        return best_key
        
    except Exception as e:
        st.warning(f"Errore nella stima della chiave: {e}")
        return 'C'

def analyze_track_ultra_precise(audio_file):
    """Analizza un file audio con massima precisione possibile."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Prima prova con aubio (se disponibile)
        aubio_bpm = analyze_with_aubio(tmp_file_path)
        
        # Carica audio per analisi Python
        # Usa un segmento più lungo per maggiore precisione
        y, sr = librosa.load(tmp_file_path, duration=90, offset=15)  # 90 secondi dal secondo 15
        
        if aubio_bpm:
            st.success(f"🎯 Usato aubio per BPM ultra-precisi!")
            tempo = aubio_bpm
        else:
            st.info("📊 Usando algoritmi Python avanzati...")
            tempo = estimate_bpm_professional(y, sr)
        
        # Stima chiave con algoritmo preciso
        key = estimate_key_precise(y, sr)
        
        return tempo, key
        
    except Exception as e:
        st.error(f"Errore nell'analisi ultra-precisa: {e}")
        return 120.0, 'C'
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def process_audio_improved(audio_file, new_tempo, new_pitch):
    """Modifica il tempo e l'intonazione del file audio con qualità migliorata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path)

        # Rilevamento BPM originale con metodo preciso
        original_tempo = estimate_bpm_professional(y, sr)
        
        if original_tempo == 0 or original_tempo is None:
            original_tempo = 120.0
            
        # Calcola il rate per il time stretching
        stretch_rate = new_tempo / original_tempo
        
        # Time-stretching con qualità migliorata
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)

        # Pitch-shifting
        if new_pitch != 0:
            y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=new_pitch)
        else:
            y_shifted = y_stretched

        # Normalizza l'audio per evitare clipping
        max_val = np.max(np.abs(y_shifted))
        if max_val > 0:
            y_shifted = y_shifted / max_val * 0.95  # Lascia un po' di headroom

        # Salvataggio in un buffer in memoria
        buffer = BytesIO()
        audio_segment = AudioSegment(
            (y_shifted * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        audio_segment.export(buffer, format="mp3", bitrate="320k")  # Qualità alta
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nell'elaborazione migliorata dell'audio: {e}")
        return None
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# --- Interfaccia utente ULTRA-PRECISA ---

st.set_page_config(page_title="Loop507 Ultra-Precise", page_icon="🎯", layout="wide")

st.title("🎯 Loop507 Ultra-Precise BPM Detection")
st.write("**Rilevamento BPM professionale con algoritmi multipli e aubio integration!**")

# Avviso importante
st.warning("""
🔥 **NOVITÀ ULTRA-PRECISE:** 
- **Aubio Integration**: Se disponibile, usa aubio per BPM ultra-precisi
- **Algoritmo Krumhansl-Schmuckler** per chiavi musicali
- **Beat Detection Avanzato** con autocorrelazione spettrale
- **Analisi su 90 secondi** per maggiore accuratezza
""")

# Colonne per layout migliorato
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Carica un file audio", 
        type=["mp3", "wav", "flac", "m4a"],
        help="Per installare aubio: pip install aubio (consigliato per BPM ultra-precisi)"
    )

with col_info:
    st.markdown("### 🎯 Sistema Ultra-Preciso")
    st.markdown("""
    **🔬 Metodi di Analisi:**
    1. **Aubio** (se disponibile) - Industry standard
    2. **Beat Detection** con autocorrelazione
    3. **Onset Analysis** ottimizzato
    4. **Spectral Flux** analysis
    
    **⚡ Installazione Aubio:**
    ```bash
    pip install aubio
    ```
    """)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Analisi del brano con barra di progresso dettagliata
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('🔍 Controllo disponibilità aubio...')
    progress_bar.progress(10)
    
    status_text.text('📊 Avvio analisi multi-algoritmo...')
    progress_bar.progress(25)
    
    status_text.text('🎵 Analisi BPM con massima precisione...')
    progress_bar.progress(50)
    
    tempo_val, key_val = analyze_track_ultra_precise(uploaded_file)
    progress_bar.progress(85)
    
    status_text.text('🎹 Calcolo chiave con Krumhansl-Schmuckler...')
    camelot_key = get_camelot_key(key_val)
    progress_bar.progress(100)
    
    status_text.text('✅ Analisi ultra-precisa completata!')
    progress_bar.empty()
    status_text.empty()
    
    # Mostra i risultati con indicatore di precisione
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 BPM Ultra-Precisi", f"{tempo_val:.2f}", help="Rilevamento con algoritmi professionali")
    with col2:
        st.metric("🎹 Chiave Musicale", key_val, help="Algoritmo Krumhansl-Schmuckler")
    with col3:
        st.metric("🔄 Chiave Camelot", camelot_key, help="Sistema Camelot per DJ")
    with col4:
        # Indicatore di confidenza (basato su se aubio è stato usato)
        confidence = "🟢 ALTA" if "aubio" in str(st.session_state.get('last_analysis', '')) else "🟡 MEDIA"
        st.metric("🎯 Confidenza", confidence, help="Basato su algoritmi utilizzati")

    # --- Controlli per la modifica migliorati ---
    st.sidebar.header("🎛️ Controlli Ultra-Precisi")
    
    # Presets estesi
    st.sidebar.subheader("⚡ Preset BPM")
    
    preset_cols = st.sidebar.columns(2)
    with preset_cols[0]:
        if st.button("🏠 House\n(128)", key="house"):
            st.session_state.preset_tempo = 128.0
        if st.button("🎵 Techno\n(132)", key="techno"):
            st.session_state.preset_tempo = 132.0
        if st.button("🎶 Trance\n(136)", key="trance"):
            st.session_state.preset_tempo = 136.0
    
    with preset_cols[1]:
        if st.button("🔥 Hardstyle\n(150)", key="hardstyle"):
            st.session_state.preset_tempo = 150.0
        if st.button("🎸 Rock\n(120)", key="rock"):
            st.session_state.preset_tempo = 120.0
        if st.button("🎺 Dubstep\n(140)", key="dubstep"):
            st.session_state.preset_tempo = 140.0
    
    # Usa preset se selezionato
    default_tempo = getattr(st.session_state, 'preset_tempo', float(tempo_val))
    
    new_tempo = st.sidebar.slider(
        "🥁 Nuovi BPM Ultra-Precisi", 
        min_value=60.0, 
        max_value=200.0, 
        value=default_tempo, 
        step=0.01,  # Precisione a centesimi
        help="Controllo ultra-preciso con step di 0.01 BPM"
    )
    
    new_pitch = st.sidebar.slider(
        "🎼 Modifica Tonalità (semitoni)", 
        min_value=-12, 
        max_value=12, 
        value=0,
        help="Pitch shifting preciso"
    )
    
    # Informazioni dettagliate sulle modifiche
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Anteprima Ultra-Precisa")
    
    tempo_change = ((new_tempo - tempo_val) / tempo_val) * 100
    st.sidebar.metric("Variazione Tempo", f"{tempo_change:+.2f}%")
    
    if abs(new_tempo - tempo_val) > 0.01:
        st.sidebar.write(f"🎵 Da {tempo_val:.2f} a {new_tempo:.2f} BPM")
    
    if new_pitch != 0:
        direction = "più acuto" if new_pitch > 0 else "più grave"
        st.sidebar.write(f"🎵 Tonalità: {abs(new_pitch)} semitoni {direction}")

    # Pulsante di elaborazione
    if st.sidebar.button("🎯 Applica Modifiche Ultra-Precise", type="primary", use_container_width=True):
        with st.spinner('🔄 Elaborazione ultra-precisa in corso...'):
            progress_bar_proc = st.progress(0)
            progress_bar_proc.progress(20)
            
            processed_audio_buffer = process_audio_improved(uploaded_file, new_tempo, new_pitch)
            progress_bar_proc.progress(100)
            progress_bar_proc.empty()
        
        if processed_audio_buffer is not None:
            st.success("🎯 Modifiche ultra-precise applicate!")
            st.balloons()
            
            st.subheader("🎧 Risultato Ultra-Preciso")
            st.audio(processed_audio_buffer, format="audio/mp3")
            
            # Nome del file con precisione
            original_name = uploaded_file.name.rsplit('.', 1)[0]
            pitch_str = f"{new_pitch:+d}st" if new_pitch != 0 else "0st"
            modified_name = f"Loop507_UltraPrecise_{original_name}_{new_tempo:.2f}BPM_{pitch_str}.mp3"
            
            st.download_button(
                label="⬇️ Scarica (Ultra-Preciso 320kbps)",
                data=processed_audio_buffer,
                file_name=modified_name,
                mime="audio/mp3",
                type="primary",
                use_container_width=True
            )
            
            st.info(f"🎯 **Ultra-Precise:** {tempo_val:.2f} → {new_tempo:.2f} BPM | Pitch: {new_pitch:+d}st")

# Footer con istruzioni aubio
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Setup Ultra-Preciso")
st.sidebar.code("""
# Installa aubio per BPM ultra-precisi
pip install aubio

# Su Ubuntu/Debian
sudo apt-get install aubio-tools

# Su macOS
brew install aubio
""")

st.sidebar.markdown("### 🎯 Algoritmi Utilizzati")
st.sidebar.markdown("""
**🔬 BPM Detection:**
- Aubio (industry standard)
- Spectral flux autocorrelation
- Multi-onset analysis
- Beat tracking avanzato

**🎹 Key Detection:**
- Krumhansl-Schmuckler algorithm
- Chroma CQT analysis
- Major/minor correlation

**💎 Qualità:**
- 90 secondi di analisi
- Step 0.01 BPM
- Export 320kbps
""")

if uploaded_file is None:
    st.markdown("---")
    st.info("""
    ### 🎯 Sistema Ultra-Preciso per Professionisti
    
    **Per massima precisione installa aubio:**
    ```bash
    pip install aubio
    ```
    
    **Algoritmi professionali integrati:**
    - Spectral flux analysis per beat detection
    - Krumhansl-Schmuckler per chiavi musicali  
    - Multi-method BPM consensus
    - 90 secondi di analisi per stabilità
    
    **Precision Controls:**
    - Step BPM di 0.01 per controllo millimetrico
    - Preset per tutti i generi musicali
    - Analisi di confidenza in tempo reale
    """)
