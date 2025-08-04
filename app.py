import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os

# --- Funzioni di analisi e manipolazione audio ---

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

def estimate_key(y, sr):
    """Stima la chiave musicale usando il croma."""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Per semplicità, restituiamo una chiave basata sul picco più alto nel croma
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return keys[key_idx]
    except Exception as e:
        st.warning(f"Errore nella stima della chiave: {e}")
        return 'C'

def analyze_track(audio_file):
    """Analizza un file audio per BPM e chiave musicale."""
    # Salva il file caricato in un file temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Carica l'audio usando il percorso del file temporaneo
        y, sr = librosa.load(tmp_file_path)

        # Rilevamento BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Assicurati che 'tempo' sia un numero singolo
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Rilevamento chiave
        key = estimate_key(y, sr)
        
        return tempo, key
        
    except Exception as e:
        st.error(f"Errore nell'analisi del brano: {e}")
        return 120.0, 'C'
    finally:
        # Rimuovi il file temporaneo
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def process_audio(audio_file, new_tempo, new_pitch):
    """Modifica il tempo e l'intonazione del file audio."""
    # Salva il file caricato in un file temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path)

        # Time-stretching
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Gestione del caso in cui il tempo originale è 0
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
            
        if tempo == 0:
            tempo = 120.0
            
        # Calcola il rate per il time stretching
        stretch_rate = new_tempo / tempo
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)

        # Pitch-shifting
        y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=new_pitch)

        # Normalizza l'audio per evitare clipping
        y_shifted = y_shifted / np.max(np.abs(y_shifted))

        # Salvataggio in un buffer in memoria
        buffer = BytesIO()
        audio_segment = AudioSegment(
            (y_shifted * 32767).astype(np.int16).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        audio_segment.export(buffer, format="mp3")
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nell'elaborazione dell'audio: {e}")
        return None
    finally:
        # Rimuovi il file temporaneo
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# --- Interfaccia utente con Streamlit ---

st.title("🎧 Loop507 in the Mix")
st.write("Carica un brano e modifica BPM e tonalità per un missaggio perfetto!")

uploaded_file = st.file_uploader("Carica un file audio", type=["mp3", "wav", "flac", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Analisi del brano
    with st.spinner('🔍 Analizzo il brano...'):
        tempo_val, key_val = analyze_track(uploaded_file)
        camelot_key = get_camelot_key(key_val)
    
    st.success("✅ Analisi completata!")
    
    # Mostra i risultati in colonne
    col1, col2 = st.columns(2)
    with col1:
        st.metric("BPM Originali", f"{tempo_val:.1f}")
    with col2:
        st.metric("Chiave Camelot", camelot_key)

    # --- Controlli per la modifica ---
    st.sidebar.header("🎛️ Modifica i Parametri")
    
    new_tempo = st.sidebar.slider(
        "Nuovi BPM", 
        min_value=50.0, 
        max_value=200.0, 
        value=float(tempo_val), 
        step=0.5,
        help="Modifica la velocità del brano"
    )
    
    new_pitch = st.sidebar.slider(
        "Modifica Tonalità (semitoni)", 
        min_value=-12, 
        max_value=12, 
        value=0,
        help="Cambia la tonalità del brano. +12 = un'ottava più acuta, -12 = un'ottava più grave"
    )
    
    # Mostra la differenza percentuale del tempo
    tempo_change = ((new_tempo - tempo_val) / tempo_val) * 100
    st.sidebar.write(f"Variazione tempo: {tempo_change:+.1f}%")

    if st.sidebar.button("🎵 Applica Modifiche", type="primary"):
        with st.spinner('🔄 Elaboro il brano...'):
            processed_audio_buffer = process_audio(uploaded_file, new_tempo, new_pitch)
        
        if processed_audio_buffer is not None:
            st.success("🎉 Modifiche applicate con successo!")
            st.audio(processed_audio_buffer, format="audio/mp3")
            
            # Nome del file modificato
            original_name = uploaded_file.name.rsplit('.', 1)[0]
            modified_name = f"loop507_mixed_{original_name}_BPM{new_tempo:.0f}_PITCH{new_pitch:+d}.mp3"
            
            st.download_button(
                label="⬇️ Scarica il brano modificato",
                data=processed_audio_buffer,
                file_name=modified_name,
                mime="audio/mp3"
            )
        else:
            st.error("❌ Errore nell'elaborazione del file audio.")

# Aggiungi informazioni nell'sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informazioni")
st.sidebar.markdown("""
**Come usare:**
1. Carica un file audio
2. Visualizza BPM e chiave originali
3. Modifica i parametri nella sidebar
4. Applica le modifiche e scarica il risultato

**Formati supportati:** MP3, WAV, FLAC, M4A
""")
