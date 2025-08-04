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
        
        return y, sr, onset_frames, tempo
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
if 'deck_a' not in st.session_state: st.session_state.deck_a = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_b' not in st.session_state: st.session_state.deck_b = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_c' not in st.session_state: st.session_state.deck_c = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_d' not in st.session_state: st.session_state.deck_d = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_e' not in st.session_state: st.session_state.deck_e = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_f' not in st.session_state: st.session_state.deck_f = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_g' not in st.session_state: st.session_state.deck_g = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'deck_h' not in st.session_state: st.session_state.deck_h = {'file': None, 'y': None, 'sr': None, 'onsets': None, 'tempo': None}
if 'decomposed_manual_segments' not in st.session_state:
    st.session_state.decomposed_manual_segments = []

# Deck A-D
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.header("Deck A")
    uploaded_file_a = st.file_uploader("Brano A", type=["mp3", "wav"], key="uploader_a")
    if uploaded_file_a:
        st.audio(uploaded_file_a, format='audio/mp3')
        if uploaded_file_a != st.session_state.deck_a['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_a)
                st.session_state.deck_a = {'file': uploaded_file_a, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

with col2:
    st.header("Deck B")
    uploaded_file_b = st.file_uploader("Brano B", type=["mp3", "wav"], key="uploader_b")
    if uploaded_file_b:
        st.audio(uploaded_file_b, format='audio/mp3')
        if uploaded_file_b != st.session_state.deck_b['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_b)
                st.session_state.deck_b = {'file': uploaded_file_b, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")
            
with col3:
    st.header("Deck C")
    uploaded_file_c = st.file_uploader("Brano C", type=["mp3", "wav"], key="uploader_c")
    if uploaded_file_c:
        st.audio(uploaded_file_c, format='audio/mp3')
        if uploaded_file_c != st.session_state.deck_c['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_c)
                st.session_state.deck_c = {'file': uploaded_file_c, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

with col4:
    st.header("Deck D")
    uploaded_file_d = st.file_uploader("Brano D", type=["mp3", "wav"], key="uploader_d")
    if uploaded_file_d:
        st.audio(uploaded_file_d, format='audio/mp3')
        if uploaded_file_d != st.session_state.deck_d['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_d)
                st.session_state.deck_d = {'file': uploaded_file_d, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

# Deck E-H
col5, col6, col7, col8 = st.columns(4)
with col5:
    st.header("Deck E")
    uploaded_file_e = st.file_uploader("Brano E", type=["mp3", "wav"], key="uploader_e")
    if uploaded_file_e:
        st.audio(uploaded_file_e, format='audio/mp3')
        if uploaded_file_e != st.session_state.deck_e['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_e)
                st.session_state.deck_e = {'file': uploaded_file_e, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

with col6:
    st.header("Deck F")
    uploaded_file_f = st.file_uploader("Brano F", type=["mp3", "wav"], key="uploader_f")
    if uploaded_file_f:
        st.audio(uploaded_file_f, format='audio/mp3')
        if uploaded_file_f != st.session_state.deck_f['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_f)
                st.session_state.deck_f = {'file': uploaded_file_f, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")
            
with col7:
    st.header("Deck G")
    uploaded_file_g = st.file_uploader("Brano G", type=["mp3", "wav"], key="uploader_g")
    if uploaded_file_g:
        st.audio(uploaded_file_g, format='audio/mp3')
        if uploaded_file_g != st.session_state.deck_g['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_g)
                st.session_state.deck_g = {'file': uploaded_file_g, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

with col8:
    st.header("Deck H")
    uploaded_file_h = st.file_uploader("Brano H", type=["mp3", "wav"], key="uploader_h")
    if uploaded_file_h:
        st.audio(uploaded_file_h, format='audio/mp3')
        if uploaded_file_h != st.session_state.deck_h['file']:
            with st.spinner('Analisi in corso...'):
                y, sr, onsets, tempo = analyze_track_for_slicing(uploaded_file_h)
                st.session_state.deck_h = {'file': uploaded_file_h, 'y': y, 'sr': sr, 'onsets': onsets, 'tempo': tempo}
            st.success("Analisi completata!")

st.sidebar.header("Controlli Ricomposizione")

active_decks_data = []
if st.session_state.deck_a['file']: active_decks_data.append(('A', st.session_state.deck_a))
if st.session_state.deck_b['file']: active_decks_data.append(('B', st.session_state.deck_b))
if st.session_state.deck_c['file']: active_decks_data.append(('C', st.session_state.deck_c))
if st.session_state.deck_d['file']: active_decks_data.append(('D', st.session_state.deck_d))
if st.session_state.deck_e['file']: active_decks_data.append(('E', st.session_state.deck_e))
if st.session_state.deck_f['file']: active_decks_data.append(('F', st.session_state.deck_f))
if st.session_state.deck_g['file']: active_decks_data.append(('G', st.session_state.deck_g))
if st.session_state.deck_h['file']: active_decks_data.append(('H', st.session_state.deck_h))


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
            st.sidebar.subheader("Selezione Segmenti")
            selected_segment_indices = st.sidebar.multiselect(
                "Scegli i segmenti per il tuo mix",
                options=[f"{seg['source']} - Segmento {seg['index'] + 1}" for seg in st.session_state.decomposed_manual_segments]
            )

            if st.sidebar.button("Crea Mix Manuale"):
                with st.spinner('Creazione del brano ricomposto...'):
                    selected_segments_list = []
                    for sel_index in selected_segment_indices:
                        parts = sel_index.split(' - Segmento ')
                        source = parts[0]
                        index = int(parts[1]) - 1
                        
                        for seg_info in st.session_state.decomposed_manual_segments:
                            if seg_info['source'] == source and seg_info['index'] == index:
                                selected_segments_list.append((seg_info['segment'], seg_info['sr']))
                                break
                    
                    if selected_segments_list:
                        combined_audio, target_sr = combine_segments(selected_segments_list)
                        processed_audio_buffer = export_audio(combined_audio, target_sr)
                        
                        if processed_audio_buffer:
                            st.subheader("Mix Manuale")
                            st.audio(processed_audio_buffer, format="audio/mp3")
                            st.download_button("Scarica Mix Manuale", data=processed_audio_buffer, file_name="mix_manuale.mp3", mime="audio/mp3")
                        else:
                            st.error("Impossibile creare il mix selezionato.")
                    else:
                        st.error("Nessun segmento selezionato.")
        else:
            st.sidebar.warning("Devi decomporre i brani prima di usare questa opzione.")
    
    if recomposition_mode == "Mix Casuale Completo":
        if st.sidebar.button("Crea Mix Casuale Completo"):
            with st.spinner('Creazione del brano ricomposto...'):
                all_raw_segments = []
                for deck_name, deck_data in active_decks_data:
                    if deck_data['y'] is not None and deck_data['tempo'] is not None and deck_data['tempo'] > 0:
                        all_raw_segments.extend([(seg, deck_data['sr']) for seg in get_beat_segments(deck_data['y'], deck_data['sr'], deck_data['tempo'], 1)])
                
                if not all_raw_segments:
                    st.error("Impossibile creare il mix. Carica almeno un brano con un ritmo rilevabile.")
                else:
                    random.shuffle(all_raw_segments)
                    
                    combined_audio, target_sr = combine_segments(all_raw_segments)
                    processed_audio_buffer = export_audio(combined_audio, target_sr)

                    if processed_audio_buffer:
                        st.subheader("Mix Casuale Completo")
                        st.audio(processed_audio_buffer, format="audio/mp3")
                        st.download_button("Scarica Mix Casuale", data=processed_audio_buffer, file_name="mix_casuale.mp3", mime="audio/mp3")
                    else:
                        st.error("Impossibile creare il mix.")
