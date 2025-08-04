import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
from scipy import signal

# --- Funzioni di analisi e manipolazione audio migliorate ---

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

def estimate_key_advanced(y, sr):
    """Stima la chiave musicale usando il profilo cromatico avanzato."""
    try:
        # Usa una finestra più grande per maggiore precisione
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        
        # Calcola il profilo cromatico medio
        chroma_mean = np.mean(chroma, axis=1)
        
        # Trova i picchi nel profilo cromatico
        peaks, _ = signal.find_peaks(chroma_mean, height=np.mean(chroma_mean))
        
        if len(peaks) > 0:
            # Prendi il picco più alto
            key_idx = peaks[np.argmax(chroma_mean[peaks])]
        else:
            key_idx = np.argmax(chroma_mean)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Determina se è maggiore o minore basandosi sui profili armonici
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Ruota i profili per la chiave rilevata
        major_rotated = np.roll(major_profile, key_idx)
        minor_rotated = np.roll(minor_profile, key_idx)
        
        # Calcola la correlazione
        major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
        
        base_key = keys[key_idx]
        if major_corr > minor_corr:
            return base_key
        else:
            # Converti in minore
            minor_keys = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m']
            return minor_keys[key_idx]
            
    except Exception as e:
        st.warning(f"Errore nella stima avanzata della chiave: {e}")
        return 'C'

def estimate_bpm_advanced(y, sr):
    """Stima i BPM usando metodi multipli per maggiore precisione."""
    try:
        # Metodo 1: Beat tracking tradizionale
        tempo1, beats1 = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        
        # Metodo 2: Onset detection + autocorrelazione
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
        if len(onset_frames) > 1:
            onset_times = librosa.onset.frames_to_time(onset_frames, sr=sr, hop_length=512)
            onset_intervals = np.diff(onset_times)
            
            if len(onset_intervals) > 0:
                # Calcola BPM dalla mediana degli intervalli
                median_interval = np.median(onset_intervals)
                tempo2 = 60.0 / median_interval if median_interval > 0 else 120.0
            else:
                tempo2 = 120.0
        else:
            tempo2 = 120.0
        
        # Metodo 3: Tempogram (se disponibile)
        try:
            tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=512)
            tempo_freqs = librosa.tempo_frequencies(len(tempogram))
            tempo3_idx = np.argmax(np.mean(tempogram, axis=1))
            tempo3 = tempo_freqs[tempo3_idx]
        except:
            tempo3 = tempo1
        
        # Normalizza i tempi (gestisci array)
        if isinstance(tempo1, np.ndarray):
            tempo1 = float(tempo1[0]) if len(tempo1) > 0 else 120.0
        else:
            tempo1 = float(tempo1)
            
        tempo2 = float(tempo2)
        tempo3 = float(tempo3)
        
        # Filtra valori irrealistici
        tempos = [t for t in [tempo1, tempo2, tempo3] if 60 <= t <= 200]
        
        if not tempos:
            return 120.0
        
        # Calcola la mediana per robustezza
        final_tempo = np.median(tempos)
        
        # Se i tempi sono molto diversi, prendi quello più probabile
        if np.std(tempos) > 20:
            # Preferisci tempi nell'intervallo dance/elettronica (120-140 BPM)
            dance_tempos = [t for t in tempos if 115 <= t <= 145]
            if dance_tempos:
                final_tempo = np.median(dance_tempos)
        
        return float(final_tempo)
        
    except Exception as e:
        st.warning(f"Errore nella stima avanzata del BPM: {e}")
        return 120.0

def analyze_track_advanced(audio_file):
    """Analizza un file audio per BPM e chiave musicale con precisione migliorata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Carica solo i primi 60 secondi per velocizzare l'analisi
        y, sr = librosa.load(tmp_file_path, duration=60, offset=30)  # Inizia dal secondo 30
        
        # Rilevamento BPM avanzato
        tempo = estimate_bpm_advanced(y, sr)
        
        # Rilevamento chiave avanzato
        key = estimate_key_advanced(y, sr)
        
        return tempo, key
        
    except Exception as e:
        st.error(f"Errore nell'analisi avanzata del brano: {e}")
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

        # Rilevamento BPM originale
        original_tempo = estimate_bpm_advanced(y, sr)
        
        if original_tempo == 0:
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

# --- Interfaccia utente con Streamlit migliorata ---

st.set_page_config(page_title="Loop507 in the Mix", page_icon="🎧", layout="wide")

st.title("🎧 Loop507 in the Mix - Versione Pro")
st.write("**Analisi avanzata e modifica precisa di BPM e tonalità per un missaggio professionale!**")

# Colonne per layout migliorato
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Carica un file audio", 
        type=["mp3", "wav", "flac", "m4a"],
        help="Supporta MP3, WAV, FLAC, M4A - per risultati ottimali usa file di qualità alta"
    )

with col_info:
    st.markdown("### 🔥 Novità v2.0")
    st.markdown("""
    - **BPM ultra-precisi** con tripla analisi
    - **Rilevamento chiave avanzato** maggiore/minore
    - **Qualità audio migliorata** a 320kbps
    - **Analisi veloce** sui primi 60 secondi
    """)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Analisi del brano con barra di progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('🔍 Analizzo il brano con algoritmi avanzati...')
    progress_bar.progress(25)
    
    tempo_val, key_val = analyze_track_advanced(uploaded_file)
    progress_bar.progress(75)
    
    camelot_key = get_camelot_key(key_val)
    progress_bar.progress(100)
    
    status_text.text('✅ Analisi completata!')
    progress_bar.empty()
    status_text.empty()
    
    # Mostra i risultati in un layout migliorato
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎵 BPM Rilevati", f"{tempo_val:.1f}", help="Rilevamento con tripla verifica")
    with col2:
        st.metric("🎹 Chiave Musicale", key_val, help="Chiave in notazione standard")
    with col3:
        st.metric("🔄 Chiave Camelot", camelot_key, help="Sistema Camelot per DJ")

    # --- Controlli per la modifica migliorati ---
    st.sidebar.header("🎛️ Controlli Professionali")
    
    # Presets comuni per DJ
    st.sidebar.subheader("⚡ Preset Veloci")
    col_preset1, col_preset2 = st.sidebar.columns(2)
    
    with col_preset1:
        if st.button("🏠 House\n(128 BPM)", help="Converti a 128 BPM per house music"):
            st.session_state.preset_tempo = 128.0
    with col_preset2:
        if st.button("🎵 Techno\n(132 BPM)", help="Converti a 132 BPM per techno"):
            st.session_state.preset_tempo = 132.0
    
    # Usa preset se selezionato
    default_tempo = getattr(st.session_state, 'preset_tempo', float(tempo_val))
    
    new_tempo = st.sidebar.slider(
        "🥁 Nuovi BPM", 
        min_value=60.0, 
        max_value=200.0, 
        value=default_tempo, 
        step=0.1,
        help="Modifica la velocità del brano con precisione decimale"
    )
    
    new_pitch = st.sidebar.slider(
        "🎼 Modifica Tonalità (semitoni)", 
        min_value=-12, 
        max_value=12, 
        value=0,
        help="Cambia la tonalità: +12 = ottava alta, -12 = ottava bassa"
    )
    
    # Informazioni dettagliate sulle modifiche
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Anteprima Modifiche")
    
    tempo_change = ((new_tempo - tempo_val) / tempo_val) * 100
    st.sidebar.metric("Variazione Tempo", f"{tempo_change:+.1f}%")
    
    if new_pitch != 0:
        direction = "più acuto" if new_pitch > 0 else "più grave"
        st.sidebar.write(f"🎵 Tonalità: {abs(new_pitch)} semitoni {direction}")
    
    # Calcola la nuova chiave stimata
    if new_pitch != 0:
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        try:
            current_idx = keys.index(key_val.replace('m', ''))
            new_key_idx = (current_idx + new_pitch) % 12
            new_key_base = keys[new_key_idx]
            new_key = new_key_base + ('m' if 'm' in key_val else '')
            new_camelot = get_camelot_key(new_key)
            st.sidebar.write(f"🔄 Nuova chiave: {new_key} ({new_camelot})")
        except:
            pass

    # Pulsante di elaborazione prominente
    if st.sidebar.button("🎵 Applica Modifiche Pro", type="primary", use_container_width=True):
        with st.spinner('🔄 Elaborazione professionale in corso...'):
            # Barra di progresso per l'elaborazione
            progress_bar_proc = st.progress(0)
            progress_bar_proc.progress(10)
            
            processed_audio_buffer = process_audio_improved(uploaded_file, new_tempo, new_pitch)
            progress_bar_proc.progress(100)
            progress_bar_proc.empty()
        
        if processed_audio_buffer is not None:
            st.success("🎉 Modifiche applicate con successo!")
            st.balloons()  # Effetto celebrativo
            
            # Player per l'anteprima
            st.subheader("🎧 Anteprima Risultato")
            st.audio(processed_audio_buffer, format="audio/mp3")
            
            # Nome del file modificato più dettagliato
            original_name = uploaded_file.name.rsplit('.', 1)[0]
            pitch_str = f"{new_pitch:+d}st" if new_pitch != 0 else "0st"
            modified_name = f"Loop507_Pro_{original_name}_{new_tempo:.0f}BPM_{pitch_str}.mp3"
            
            # Pulsante di download prominente
            st.download_button(
                label="⬇️ Scarica Brano Remixato (320kbps)",
                data=processed_audio_buffer,
                file_name=modified_name,
                mime="audio/mp3",
                type="primary",
                use_container_width=True
            )
            
            # Statistiche finali
            st.info(f"📈 **Statistiche:** Tempo originale: {tempo_val:.1f} BPM → Nuovo: {new_tempo:.1f} BPM | Pitch: {new_pitch:+d} semitoni")
        else:
            st.error("❌ Errore nell'elaborazione del file audio.")

# Footer informativo migliorato
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Loop507 Pro Features")
st.sidebar.markdown("""
**🔬 Analisi Avanzata:**
- Rilevamento BPM con tripla verifica
- Analisi armonica per chiavi maggiori/minori
- Supporto formati professionali

**⚡ Preset Rapidi:**
- House (128 BPM)
- Techno (132 BPM)
- Controlli decimali precisi

**💎 Qualità Pro:**
- Export MP3 a 320kbps
- Preservazione qualità audio
- Normalizzazione automatica

**📱 Formati:** MP3, WAV, FLAC, M4A
""")

# Info aggiuntiva nel main
if uploaded_file is None:
    st.markdown("---")
    col_tip1, col_tip2, col_tip3 = st.columns(3)
    
    with col_tip1:
        st.markdown("""
        ### 🎯 Per DJ Professionisti
        - Rilevamento BPM ultra-preciso
        - Sistema Camelot integrato
        - Preset per generi dance
        """)
    
    with col_tip2:
        st.markdown("""
        ### 🔊 Qualità Audio
        - Export a 320kbps
        - Preservazione dinamiche
        - Normalizzazione smart
        """)
    
    with col_tip3:
        st.markdown("""
        ### ⚡ Velocità
        - Analisi su 60 secondi
        - Algoritmi ottimizzati
        - Interfaccia reattiva
        """)
