import streamlit as st
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from io import BytesIO

# --- Funzioni di analisi e manipolazione audio ---

def analyze_track(audio_file):
    """Analizza un file audio per BPM e chiave musicale."""
    y, sr = librosa.load(audio_file)

    # Rilevamento BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Rilevamento chiave
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    key = librosa.key_to_note(librosa.key_to_chroma(chroma=chroma), major=True) # Semplificato per l'esempio

    return tempo[0], key

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
    # Per una gestione più robusta, l'algoritmo di librosa andrebbe migliorato.
    return camelot_map.get(key.split(':')[0], 'Unknown')

def process_audio(audio_file, new_tempo, new_pitch):
    """Modifica il tempo e l'intonazione del file audio."""
    y, sr = librosa.load(audio_file)

    # Time-stretching (cambia il tempo senza cambiare l'intonazione)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    y_stretched = librosa.effects.time_stretch(y=y, rate=new_tempo/tempo)

    # Pitch-shifting (cambia l'intonazione)
    y_shifted = librosa.effects.pitch_shift(y=y_stretched, sr=sr, n_steps=new_pitch)

    # Salvataggio in un buffer in memoria (per evitare di salvare su disco)
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

# --- Interfaccia utente con Streamlit ---

st.title("Loop507 in the Mix")
st.write("Carica un brano e modifica BPM e tonalità per un missaggio perfetto!")

uploaded_file = st.file_uploader("Carica un file audio", type=["mp3", "wav"])

if uploaded_file is not None:
    # Mostra i controlli solo dopo aver caricato il file
    st.audio(uploaded_file, format='audio/mp3')

    # Analisi del brano
    with st.spinner('Analizzo il brano...'):
        tempo, key = analyze_track(uploaded_file)
        camelot_key = get_camelot_key(key)
    
    st.success("Analisi completata!")
    st.write(f"**BPM originali:** {tempo:.2f}")
    st.write(f"**Chiave di Camelot:** {camelot_key}")

    # --- Controlli per la modifica ---
    st.sidebar.header("Modifica i Parametri")
    
    new_tempo = st.sidebar.slider("Nuovi BPM", min_value=50.0, max_value=200.0, value=float(tempo), step=0.1)
    new_pitch = st.sidebar.slider("Modifica Tonalità (semitoni)", min_value=-12, max_value=12, value=0)

    if st.sidebar.button("Applica Modifiche"):
        with st.spinner('Elaboro il brano...'):
            processed_audio_buffer = process_audio(uploaded_file, new_tempo, new_pitch)
        
        st.success("Modifiche applicate!")
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button(
            label="Scarica il brano modificato",
            data=processed_audio_buffer,
            file_name=f"loop507_mixed_{uploaded_file.name}",
            mime="audio/mp3"
        )
