import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import aubio

# --- Funzioni di supporto per la chiave musicale ---

CAMELOT_MAP = {
    'C': '8B', 'Am': '8A', 'G': '9B', 'Em': '9A',
    'D': '10B', 'Bm': '10A', 'A': '11B', 'F#m': '11A',
    'E': '12B', 'C#m': '12A', 'B': '1B', 'G#m': '1A',
    'F#': '2B', 'D#m': '2A', 'Db': '3B', 'Bbm': '3A',
    'Ab': '4B', 'Fm': '4A', 'Eb': '5B', 'Cm': '5A',
    'Bb': '6B', 'Gm': '6A', 'F': '7B', 'Dm': '7A'
}

SEMITONES_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
    'A#': 10, 'Bb': 10, 'B': 11, 'Am': 9, 'Bm': 11, 'Em': 4, 'Fm': 5,
    'Gm': 7, 'C#m': 1, 'D#m': 3, 'F#m': 6, 'G#m': 8, 'Bbm': 10
}

def get_camelot_key(key):
    return CAMELOT_MAP.get(key, 'Unknown')

def get_standard_key(camelot_key):
    for standard_key, camelot_val in CAMELOT_MAP.items():
        if camelot_val == camelot_key:
            return standard_key
    return 'C'

def get_pitch_shift(original_key, new_key):
    if original_key in SEMITONES_MAP and new_key in SEMITONES_MAP:
        orig_semitones = SEMITONES_MAP[original_key]
        new_semitones = SEMITONES_MAP[new_key]
        shift = new_semitones - orig_semitones
        if shift > 6:
            shift -= 12
        elif shift < -6:
            shift += 12
        return shift
    return 0

# --- Funzioni di analisi e manipolazione audio ---

def analyze_track(audio_file_object):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file_object.getvalue())
        tmp_path = tmp_file.name

    # Rilevamento BPM con aubio (più preciso)
    samplerate = 0
    win_s = 512
    hop_s = win_s // 2
    s = aubio.source(tmp_path, samplerate, hop_s)
    samplerate = s.samplerate
    o = aubio.tempo("default", win_s, hop_s, samplerate)
    total_frames = 0
    beats = []
    while True:
        samples, read = s()
        if o(samples):
            beats.append(o.get_last_s())
        total_frames += read
        if read < hop_s:
            break
    
    if len(beats) > 1:
        tempo_val = 60. * (len(beats) - 1) / (beats[-1] - beats[0])
    else:
        tempo_val = 120.0
    
    # Rilevamento chiave con librosa
    y, sr = librosa.load(tmp_path, sr=None)
    key = estimate_key_simple(y, sr)
    
    return tempo_val, key

def estimate_key_simple(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_idx = np.argmax(chroma_mean)
    key_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return key_notes[key_idx]

def process_audio(audio_file_object, new_tempo, pitch_shift):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file_object.getvalue())
        tmp_path = tmp_file.name
    
    y, sr = librosa.load(tmp_path, sr=None)
    
    # Ricalcolo il tempo originale per la time-stretching
    onset_env = librosa.onset.onset_detect(y=y, sr=sr)
    tempo_originale, _ = librosa.beat.beat_track(onset_env=onset_env, sr=sr)

    if tempo_originale == 0: tempo_originale = 120.0
    
    y_stretched = librosa.effects.time_stretch(y=y, rate=new_tempo / tempo_originale)
    y_shifted = librosa.effects.pitch_shift(y=y_stretched, sr=sr, n_steps=pitch_shift)
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
st.write("Carica due brani, analizzali e sincronizzali per un missaggio perfetto!")
st.info("I brani vengono elaborati sul server, l'operazione potrebbe richiedere qualche secondo.")

if 'deck_a' not in st.session_state:
    st.session_state.deck_a = {'tempo': 0, 'key': 'C', 'file': None}
if 'deck_b' not in st.session_state:
    st.session_state.deck_b = {'tempo': 0, 'key': 'C', 'file': None}

col1, col2 = st.columns(2)

with col1:
    st.header("Deck A")
    uploaded_file_a = st.file_uploader("Carica Brano A", type=["mp3", "wav"], key="uploader_a")
    if uploaded_file_a:
        st.audio(uploaded_file_a, format='audio/mp3')
        if uploaded_file_a != st.session_state.deck_a['file']:
            with st.spinner('Analizzo Brano A...'):
                tempo_val, key_val = analyze_track(uploaded_file_a)
                st.session_state.deck_a['tempo'] = tempo_val
                st.session_state.deck_a['key'] = key_val
                st.session_state.deck_a['file'] = uploaded_file_a
        st.write(f"**BPM:** {st.session_state.deck_a['tempo']:.2f}")
        st.write(f"**Chiave Camelot:** {get_camelot_key(st.session_state.deck_a['key'])}")

with col2:
    st.header("Deck B")
    uploaded_file_b = st.file_uploader("Carica Brano B", type=["mp3", "wav"], key="uploader_b")
    if uploaded_file_b:
        st.audio(uploaded_file_b, format='audio/mp3')
        if uploaded_file_b != st.session_state.deck_b['file']:
            with st.spinner('Analizzo Brano B...'):
                tempo_val, key_val = analyze_track(uploaded_file_b)
                st.session_state.deck_b['tempo'] = tempo_val
                st.session_state.deck_b['key'] = key_val
                st.session_state.deck_b['file'] = uploaded_file_b
        st.write(f"**BPM:** {st.session_state.deck_b['tempo']:.2f}")
        st.write(f"**Chiave Camelot:** {get_camelot_key(st.session_state.deck_b['key'])}")

st.sidebar.header("Controlli Brano A")
if st.session_state.deck_a['file']:
    new_tempo_a = st.sidebar.slider("BPM (Brano A)", min_value=50.0, max_value=200.0, value=float(st.session_state.deck_a['tempo']), step=0.1, key="bpm_a")
    all_camelot_keys = sorted(list(CAMELOT_MAP.values()))
    current_key_a = get_camelot_key(st.session_state.deck_a['key'])
    new_camelot_key_a = st.sidebar.selectbox("Chiave (Brano A)", all_camelot_keys, index=all_camelot_keys.index(current_key_a) if current_key_a in all_camelot_keys else 0, key="key_a")
    if st.sidebar.button("Applica a Brano A", key="apply_a"):
        new_key_standard = get_standard_key(new_camelot_key_a)
        pitch_shift = get_pitch_shift(st.session_state.deck_a['key'], new_key_standard)
        with st.spinner('Elaboro Brano A...'):
            processed_audio_buffer = process_audio(st.session_state.deck_a['file'], new_tempo_a, pitch_shift)
        st.success("Modifiche applicate a Brano A!")
        st.audio(processed_audio_buffer, format="audio/mp3")
        st.download_button("Scarica Brano A", data=processed_audio_buffer, file_name=f"mixed_A.mp3", mime="audio/mp3", key="download_a")

st.sidebar.header("Controlli Brano B")
if st.session_state.deck_b['file']:
    new_tempo_b = st.sidebar.slider("BPM (Brano B)", min_value=50.0, max_value=200.0, value=float(st.session_state.deck_b['tempo']), step=0.1, key="bpm_b")
    all_camelot_keys = sorted(list(CAMELOT_MAP.values()))
    current_key_b = get_camelot_key(st.session_state.deck_b['key'])
    new_camelot_key_b = st.sidebar.selectbox("Chiave (Brano B)", all_camelot_keys, index=all_camelot_keys.index(current_key_b) if current_key_b in all_camelot_keys else 0, key="key_b")
    if st.sidebar.button("Sincronizza B su A", key="sync_b"):
        if st.session_state.deck_a['file']:
            new_tempo_b_sync = st.session_state.deck_a['tempo']
            new_key_standard = st.session_state.deck_a['key']
            pitch_shift = get_pitch_shift(st.session_state.deck_b['key'], new_key_standard)
            with st.spinner('Sincronizzo Brano B...'):
                processed_audio_buffer = process_audio(st.session_state.deck_b['file'], new_tempo_b_sync, pitch_shift)
            st.success("Brano B sincronizzato!")
            st.audio(processed_audio_buffer, format="audio/mp3")
            st.download_button("Scarica Brano B Sincronizzato", data=processed_audio_buffer, file_name=f"mixed_B_sync.mp3", mime="audio/mp3", key="download_sync_b")
            st.sidebar.write("BPM e chiave di Brano B sono stati allineati a Brano A.")
        else:
            st.sidebar.warning("Devi prima caricare il Brano A per la sincronizzazione!")
