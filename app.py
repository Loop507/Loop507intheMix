import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random

# --- Funzioni di analisi e manipolazione audio ---

@st.cache_data
def analyze_track_for_slicing(audio_file_object):
    """
    Analizza un brano e restituisce l'audio e i dati necessari per lo slicing.
    """
    audio_file_object.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        audio = AudioSegment.from_file(audio_file_object)
        audio.export(tmp_wav_file.name, format="wav")
        tmp_path = tmp_wav_file.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        duration = librosa.get_duration(y=y, sr=sr)
        
        return y, sr, onset_frames, tempo, duration
    finally:
        os.remove(tmp_path)

def get_beat_segments(y, sr, tempo, num_beats_per_segment):
    """
    Divide il brano in segmenti di N battute.
    """
    if tempo == 0:
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
    """
    Combina una lista di segmenti in un unico brano, riscampiando se necessario.
    Ogni segmento nella lista è una tupla (audio_data, sample_rate)
    """
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
    """
    Esporta l'audio processato in un buffer MP3.
    """
    if len(y) == 0 or sr is None:
        return None
        
    buffer = BytesIO()
    audio_segment = AudioSegment(
        (y * 32767).astype(np.int16).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# --- Interfaccia utente con Streamlit ---
st.title("Loop507 in the Mix: Decomposizione e Ricomposizione")
st.write("Carica i brani, decomponili in segmenti e ricomponili in un modo unico!")

# Inizializza gli 8 deck
deck_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for key in deck_keys:
    if f'deck_{key}' not in st.session_state:
        st.session_state[f'deck_{key}'] = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None, 'duration': None}

if 'decomposed_manual_segments' not in st.session_state:
    st.session_state.decomposed_manual_segments = []

col1, col2, col3, col4 = st.columns(4)
cols_1_4 = [col1, col2, col3, col4]
for i, key in enumerate(deck_keys[:4]):
    with cols_1_4[i]:
        st.header(f"Deck {key.upper()}")
        uploaded_file = st.file_uploader(f"Brano {key.upper()}", type=["mp3", "wav"], key=f"uploader_{key}")
        if uploaded_file:
            st.audio(uploaded_file, format='audio/mp3')
            if uploaded_file != st.session_state[f'deck_{key}']['file']:
                with st.spinner('Analisi in corso...'):
                    y, sr, onsets, tempo, duration = analyze_track_for_slicing(uploaded_file)
                    st.session_state[f'deck_{key}'] = {'file': uploaded_file, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo, 'duration': duration}
                st.success(f"Analisi completata! Durata: {duration:.2f} secondi")

col5, col6, col7, col8 = st.columns(4)
cols_5_8 = [col5, col6, col7, col8]
for i, key in enumerate(deck_keys[4:]):
    with cols_5_8[i]:
        st.header(f"Deck {key.upper()}")
        uploaded_file = st.file_uploader(f"Brano {key.upper()}", type=["mp3", "wav"], key=f"uploader_{key}")
        if uploaded_file:
            st.audio(uploaded_file, format='audio/mp3')
            if uploaded_file != st.session_state[f'deck_{key}']['file']:
                with st.spinner('Analisi in corso...'):
                    y, sr, onsets, tempo, duration = analyze_track_for_slicing(uploaded_file)
                    st.session_state[f'deck_{key}'] = {'file': uploaded_file, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo, 'duration': duration}
                st.success(f"Analisi completata! Durata: {duration:.2f} secondi")

st.sidebar.header("Controlli Ricomposizione")

active_decks_data = [
    (key.upper(), st.session_state[f'deck_{key}']) for key in deck_keys if st.session_state[f'deck_{key}']['file']
]

if active_decks_data:
    st.sidebar.subheader("Decomponi i Brani")
    num_beats_per_segment = st.sidebar.selectbox("Segmenti da (battute):", [1, 2, 4, 8])
    if st.sidebar.button("Decomponi Brani"):
        st.session_state.decomposed_manual_segments = []
        for deck_name, deck_data in active_decks_data:
            segments = get_beat_segments(
                deck_data['y'], deck_data['sr'], deck_data['tempo'], num_beats_per_segment)
            for i, seg in enumerate(segments):
                st.session_state.decomposed_manual_segments.append({'source': deck_name, 'index': i, 'segment': seg, 'sr': deck_data['sr']})
        
        st.sidebar.success(f"Brani decomposti in {len(st.session_state.decomposed_manual_segments)} segmenti.")

    st.sidebar.subheader("Opzioni di Ricomposizione")
    
    recomposition_mode = st.sidebar.radio(
        "Scegli una modalità:",
        ("Mix Casuale Completo", "Mix Manuale"),
    )

    if recomposition_mode == "Mix Manuale":
        if st.session_state.decomposed_manual_segments:
            st.sidebar.subheader("Seleziona e Ricomponi")
            selected_decks = st.sidebar.multiselect(
                "Scegli i brani da includere nel mix manuale",
                options=[deck[0] for deck in active_decks_data]
            )
            
            if st.sidebar.button("Crea Mix Manuale"):
                with st.spinner('Creazione del brano ricomposto...'):
                    selected_segments_list = []
                    for deck_name in selected_decks:
                        segments_from_deck = [
                            (seg['segment'], seg['sr']) for seg in st.session_state.decomposed_manual_segments if seg['source'] == deck_name
                        ]
                        selected_segments_list.extend(segments_from_deck)

                    if selected_segments_list:
                        combined_audio, target_sr = combine_segments(selected_segments_list)
                        processed_audio_buffer = export_audio(combined_audio, target_sr)
                        
                        if processed_audio_buffer:
                            st.subheader("Mix Manuale")
                            st.audio(processed_audio_buffer, format="audio/mp3")
                            duration_mix = librosa.get_duration(y=combined_audio, sr=target_sr)
                            st.success(f"Mix creato con successo! Durata: {duration_mix:.2f} secondi")
                            st.download_button("Scarica Mix Manuale", data=processed_audio_buffer, file_name="mix_manuale.mp3", mime="audio/mp3")
                        else:
                            st.error("Impossibile creare il mix selezionato.")
                    else:
                        st.error("Nessun brano selezionato per il mix manuale.")
        else:
            st.sidebar.warning("Devi decomporre i brani prima di usare questa opzione.")
    
    if recomposition_mode == "Mix Casuale Completo":
        st.sidebar.subheader("Controlli Avanzati")
        master_deck_options = ["Nessuno (default)"] + [deck[0] for deck in active_decks_data if deck[1]['tempo'] is not None and deck[1]['tempo'] > 0]
        master_deck_selection = st.sidebar.selectbox("Scegli il Deck Master per la sincronizzazione del tempo", options=master_deck_options)
        
        st.sidebar.subheader("Controllo Durata del Mix")
        st.sidebar.write("La durata standard suggerita per un mix è di 60 secondi.")
        desired_duration = st.sidebar.number_input("Durata desiderata del mix (in secondi)", min_value=1.0, value=60.0, step=1.0)
        
        if st.sidebar.button("Crea Mix Casuale Completo"):
            with st.spinner('Creazione del brano ricomposto...'):
                master_tempo = None
                if master_deck_selection != "Nessuno (default)":
                    for deck_name, deck_data in active_decks_data:
                        if deck_name == master_deck_selection and deck_data['tempo'] is not None and deck_data['tempo'] > 0:
                            master_tempo = deck_data['tempo']
                            break
                    if not master_tempo:
                        st.error("Il Deck Master selezionato non ha un tempo rilevabile. Riprova con un altro brano.")
                        st.stop()
                
                all_raw_segments = []
                for deck_name, deck_data in active_decks_data:
                    if deck_data['y'] is not None and isinstance(deck_data['y'], np.ndarray) and deck_data['y'].ndim > 0 and deck_data['tempo'] is not None and deck_data['tempo'] > 0:
                        y_to_process = deck_data['y']
                        sr_to_process = deck_data['sr']
                        current_tempo = deck_data['tempo']
                        
                        if master_tempo and current_tempo != master_tempo:
                            # Time-stretch per sincronizzare il tempo
                            stretch_factor = current_tempo / master_tempo
                            y_to_process = librosa.effects.time_stretch(y_to_process, rate=stretch_factor)
                            
                        tempo_for_segments = master_tempo if master_tempo else current_tempo
                        segments = get_beat_segments(y_to_process, sr_to_process, tempo_for_segments, 1)
                        all_raw_segments.extend([(seg, sr_to_process) for seg in segments])
                
                if not all_raw_segments:
                    st.error("Impossibile creare il mix. Carica almeno un brano con un ritmo rilevabile.")
                else:
                    tempo_for_duration = master_tempo if master_tempo else (active_decks_data[0][1]['tempo'] if active_decks_data[0][1]['tempo'] > 0 else None)
                    
                    if tempo_for_duration:
                        duration_per_beat = 60 / tempo_for_duration
                        num_segments_to_mix = int(desired_duration / duration_per_beat)
                        
                        if num_segments_to_mix > 0:
                            random_mix_segments = random.sample(all_raw_segments, min(num_segments_to_mix, len(all_raw_segments)))
                            
                            combined_audio, target_sr = combine_segments(random_mix_segments)
                            processed_audio_buffer = export_audio(combined_audio, target_sr)

                            if processed_audio_buffer:
                                st.subheader("Mix Casuale Completo")
                                st.audio(processed_audio_buffer, format="audio/mp3")
                                duration_mix = librosa.get_duration(y=combined_audio, sr=target_sr)
                                st.success(f"Mix creato con successo! Durata: {duration_mix:.2f} secondi")
                                st.download_button("Scarica Mix Casuale", data=processed_audio_buffer, file_name="mix_casuale.mp3", mime="audio/mp3")
                            else:
                                st.error("Impossibile creare il mix.")
                        else:
                            st.error("La durata desiderata è troppo breve per creare un mix significativo.")
                    else:
                        st.error("Impossibile calcolare la durata dei segmenti. Assicurati di caricare almeno un brano valido.")
