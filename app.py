import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO

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
    
    # Questo metodo funziona con chiavi come 'C', 'Am', 'G#m', ecc.
    return camelot_map.get(key, 'Unknown')

def estimate_key(y, sr):
    """Stima la chiave musicale usando il croma."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Calcola il profilo del croma medio
    chroma_mean = np.mean(chroma, axis=1)

    # Definisce i profili dei "modelli" per le chiavi maggiori e minori
    major_profiles = {
        'C': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'C#': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'D': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'Eb': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'E': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        'F': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'F#': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        'G': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'Ab': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'A': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        'Bb': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'B': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    }
    minor_profiles = {
        'Am': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'Bbm': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        'Bm': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'Cm': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        'C#m': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'Dm': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'D#m': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        'Em': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'Fm': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        'F#m': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'Gm': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'G#m': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    }

    # Calcola la correlazione tra il profilo del brano e i profili modello
    best_match_key = 'Unknown'
    best_score = 0
    all_profiles = {**major_profiles, **minor_profiles}
    for key, profile in all_profiles.items():
        score = np.dot(chroma_mean, profile)
        if score > best_score:
            best_score = score
            best_match_key = key
    
    return best_match_key


def analyze_track(audio_file):
    """Analizza un file audio per BPM e chiave musicale."""
    y, sr = librosa.load(audio_file)

    # Rilevamento BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Rilevamento chiave con il nostro algoritmo robusto
    key = estimate_key(y, sr)
    
    return tempo, key

def process_audio(audio_file, new_tempo, new_pitch):
    """Modifica il tempo e l'intonazione del file audio."""
    y, sr = librosa.load(audio_file)

    # Time-stretching
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    y_stretched = librosa.effects.time_stretch(y=y, rate=new_tempo / tempo)

    # Pitch-shifting
    y_shifted = librosa.effects.pitch_shift(y=y_stretched, sr=sr, n_steps=new_pitch)

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

# --- Interfaccia utente con Streamlit ---

st.title("Loop507 in the Mix")
st.write("Carica un brano e modifica BPM e tonalità per un missaggio perfetto!")

uploaded_file = st.file_uploader("Carica un file audio", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    # Analisi del brano
    with st.spinner('Analizzo il brano...'):
        tempo_val, key_val = analyze_track(uploaded_file)
        camelot_key = get_camelot_key(key_val)
    
    st.success("Analisi completata!")
    st.write(f"**BPM originali:** {tempo_val:.2f}")
    st.write(f"**Chiave di Camelot:** {camelot_key}")

    # --- Controlli per la modifica ---
    st.sidebar.header("Modifica i Parametri")
    
    new_tempo = st.sidebar.slider("Nuovi BPM", min_value=50.0, max_value=200.0, value=float(tempo_val), step=0.1)
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
