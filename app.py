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

# Sample rate unico a cui tutti i deck vengono riportati (fix bug #1)
TARGET_SR = 44100

# Limiti di sicurezza per la lunghezza dei segmenti (fix bug #5)
MIN_SEGMENT_SAMPLES = 256          # ~5ms a 44.1kHz, evita segmenti a lunghezza 0
MAX_SEGMENT_SECONDS = 30           # tetto di sicurezza per singolo segmento

# --- Funzioni di Analisi e Taglio ---
# NOTA STEREO: da qui in poi ogni array audio 'y' ha shape (2, n_samples) -> [canale L, canale R].
# I brani mono vengono duplicati sui due canali in fase di analisi, cosi' tutta la pipeline
# a valle (taglio, shuffle, export) lavora sempre e solo su array stereo, senza casi speciali.

@st.cache_data
def analyze_track(audio_file_object):
    """Analizza un file audio: carica in stereo, converte in WAV temporaneo, rileva tempo.
    Il file temporaneo viene sempre ripulito (fix bug #3)."""
    audio_file_object.seek(0)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            audio = AudioSegment.from_file(audio_file_object)
            audio.export(tmp_path, format="wav")

        # mono=False per preservare i due canali; sr fissa a TARGET_SR per uniformare i deck (fix bug #1)
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=False)

        # Se la sorgente era mono, librosa restituisce un array 1D: lo duplichiamo su L/R
        # cosi' l'intera pipeline a valle puo' sempre assumere shape (2, n).
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)

        # Il beat tracking lavora su un segnale mono: usiamo la media dei due canali solo per l'analisi.
        y_mono_for_beat = np.mean(y, axis=0)
        tempo, _ = librosa.beat.beat_track(y=y_mono_for_beat, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        return y, sr, tempo_val
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_beat_segments(y, sr, tempo, num_beats):
    """Segmenta in base al BPM, con guardia su tempo invalido e lunghezza minima/massima (fix bug #2, #5).
    y ha shape (2, n): il taglio avviene lungo l'asse temporale (asse 1), mantenendo entrambi i canali."""
    if tempo is None or tempo <= 1e-3:
        return []

    samples_per_beat = sr * 60 / tempo
    segment_length = int(samples_per_beat * num_beats)

    # Clamp di sicurezza: mai 0, mai oltre il tetto massimo
    max_samples = int(MAX_SEGMENT_SECONDS * sr)
    segment_length = max(MIN_SEGMENT_SAMPLES, min(segment_length, max_samples))

    total_samples = y.shape[1]
    return [
        y[:, i:i + segment_length]
        for i in range(0, total_samples, segment_length)
        if (i + segment_length) <= total_samples
    ]


def get_random_segments(y, sr, min_dur, max_dur):
    """Segmentazione casuale, con guardia contro durate/lunghezze a zero (fix bug #2).
    y ha shape (2, n): il taglio avviene lungo l'asse temporale (asse 1)."""
    segments = []
    curr = 0
    total_samples = y.shape[1]
    min_dur = max(min_dur, MIN_SEGMENT_SAMPLES / sr)  # mai sotto la soglia minima
    while curr < total_samples:
        dur = random.uniform(min_dur, max_dur)
        length = int(dur * sr)
        if length <= 0:
            break
        if curr + length > total_samples:
            break
        segments.append(y[:, curr:curr + length])
        curr += length
    return segments


def export_audio(y, sr):
    """Esporta un array stereo (2, n) in MP3 stereo, con interleaving corretto dei canali."""
    if y.shape[1] == 0:
        return None
    max_val = np.max(np.abs(y))
    y_norm = (y / max_val * 32767).astype(np.int16) if max_val > 0 else y.astype(np.int16)

    # y_norm ha shape (2, n): trasponiamo a (n, 2) e appiattiamo per ottenere
    # il classico interleaving L,R,L,R,... richiesto da un file audio stereo.
    interleaved = y_norm.T.flatten()

    buffer = BytesIO()
    audio_seg = AudioSegment(interleaved.tobytes(), frame_rate=sr, sample_width=2, channels=2)
    audio_seg.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer


def time_stretch_stereo(y, rate):
    """Cambia davvero la velocità dell'audio (non solo la lunghezza dei tagli), preservando
    l'intonazione (phase vocoder via librosa.effects.time_stretch). rate > 1 = più veloce,
    rate < 1 = più lento. Applicato canale per canale per non mescolare L/R."""
    if rate is None or rate <= 0 or abs(rate - 1.0) < 1e-3:
        return y  # nulla da fare, o rate non valido: restituisco l'audio invariato
    stretched_channels = []
    for ch in range(y.shape[0]):
        stretched_channels.append(librosa.effects.time_stretch(y[ch], rate=rate))
    # Il phase vocoder puo' produrre canali di lunghezza leggermente diversa per arrotondamenti:
    # uniformo al più corto per poter poi fare np.stack senza errori di shape.
    min_len = min(ch.shape[0] for ch in stretched_channels)
    stretched_channels = [ch[:min_len] for ch in stretched_channels]
    return np.stack(stretched_channels, axis=0)


def invalidate_mix():
    """Invalida il mix precedente quando i segmenti cambiano (fix bug #4)."""
    st.session_state.pop("mix_ready", None)
    st.session_state.audio_report = ""


# --- Interfaccia Utente Streamlit ---
st.set_page_config(page_title="Loop507 Hyper-Mixer", layout="wide")
st.title("🎧 Loop507: Audio Shuffler & Glitcher")

if 'decks' not in st.session_state:
    st.session_state.decks = {
        k: {'y': None, 'sr': None, 'tempo': 0.0, 'tempo_detected': 0.0, 'name': None}
        for k in 'abcdefgh'
    }
else:
    # Migrazione difensiva: se una sessione già aperta prima di questo aggiornamento
    # ha un dict deck "vecchio" senza 'tempo_detected' (o altri campi futuri),
    # lo completiamo qui invece di andare in KeyError.
    for _k, _d in st.session_state.decks.items():
        _d.setdefault('tempo_detected', _d.get('tempo', 0.0))
        _d.setdefault('tempo', 0.0)
        _d.setdefault('y', None)
        _d.setdefault('sr', None)
        _d.setdefault('name', None)
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'audio_report' not in st.session_state:
    st.session_state.audio_report = ""

deck_keys = list(st.session_state.decks.keys())
for row_idx in [0, 4]:
    cols = st.columns(4)
    for i, k in enumerate(deck_keys[row_idx: row_idx + 4]):
        with cols[i]:
            st.markdown(f"### Deck {k.upper()}")
            up = st.file_uploader(f"Carica {k.upper()}", type=["mp3", "wav"], key=f"up_{k}", label_visibility="collapsed")
            if up:
                if st.session_state.decks[k]['name'] != up.name:
                    with st.spinner(f"Analizzando {k.upper()}..."):
                        y, sr, t = analyze_track(up)
                        st.session_state.decks[k] = {
                            'y': y, 'sr': sr, 'tempo': t, 'tempo_detected': t, 'name': up.name
                        }
                        # Cambiando un deck, i vecchi segmenti/mix non sono piu' validi (fix bug #4)
                        st.session_state.segments = []
                        invalidate_mix()
                        # Reset del widget BPM manuale: senza questo, Streamlit terrebbe il
                        # vecchio valore del brano precedente invece di quello nuovo rilevato
                        # (bug di persistenza key, stessa famiglia del bug #1 sample rate).
                        bpm_key = f"bpm_manual_{k}"
                        if bpm_key in st.session_state:
                            del st.session_state[bpm_key]
                if st.session_state.decks[k]['y'] is not None:
                    st.success(f"{up.name}")
                    detected = st.session_state.decks[k].get('tempo_detected', 0.0)
                    if detected <= 1.0:
                        # Beat non marcato (es. classica, ambient, brani senza percussione):
                        # non e' un errore, semplicemente Librosa non trova un battito regolare.
                        st.info("🎻 Nessun BPM chiaro rilevato (brano senza beat marcato?). Imposta un BPM manuale se vuoi usare il taglio a ritmo, oppure usa la modalità 'Frenesia Casuale'.")
                    else:
                        st.write(f"⏱️ **{detected:.1f} BPM rilevati**")
                    manual_bpm = st.number_input(
                        f"BPM manuale (Deck {k.upper()})",
                        min_value=0.0, max_value=300.0,
                        value=float(detected if detected > 1.0 else 120.0),
                        step=1.0, key=f"bpm_manual_{k}",
                        help="Correggi qui se il BPM auto-rilevato è sbagliato o assente (es. musica classica)."
                    )
                    st.session_state.decks[k]['tempo'] = manual_bpm
                    st.audio(up)

st.sidebar.header("🎛️ Pannello di Controllo")
active_decks = {k: v for k, v in st.session_state.decks.items() if v['y'] is not None}

if active_decks:
    st.sidebar.subheader("1. Scegli Stile di Taglio")
    tipo_taglio = st.sidebar.radio("Modalità:", ["Ritmo Musicale (BPM)", "Frenesia Casuale (Secondi)"])

    if tipo_taglio == "Ritmo Musicale (BPM)":
        beats = st.sidebar.selectbox("Battute per segmento:", [0.5, 1, 2, 4, 8], index=2)
        apply_stretch = st.sidebar.checkbox(
            "🎚️ Correggi davvero il tempo (time-stretch)",
            value=False,
            help="Se il BPM manuale è diverso da quello rilevato, l'audio viene realmente "
                 "accelerato/rallentato (intonazione preservata) invece di limitarsi a cambiare "
                 "la lunghezza dei tagli. Più lento da calcolare e introduce lievi artefatti."
        )
        if st.sidebar.button("🔨 Decomponi a Tempo"):
            st.session_state.segments = []
            stretched_decks = 0
            for k, d in active_decks.items():
                y_for_cut = d['y']
                # Applico il time-stretch solo se: l'utente lo ha attivato, esiste un BPM
                # rilevato affidabile come riferimento, e il BPM manuale è effettivamente diverso.
                if apply_stretch and d['tempo_detected'] > 1.0 and d['tempo'] > 0 \
                        and abs(d['tempo'] - d['tempo_detected']) > 0.5:
                    rate = d['tempo'] / d['tempo_detected']
                    with st.spinner(f"Time-stretch Deck {k.upper()} (rate {rate:.3f}x)..."):
                        y_for_cut = time_stretch_stereo(d['y'], rate)
                    stretched_decks += 1
                segs = get_beat_segments(y_for_cut, d['sr'], d['tempo'], beats)
                for s in segs:
                    st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            invalidate_mix()
            if st.session_state.segments:
                msg = f"Creati {len(st.session_state.segments)} pezzi musicali!"
                if stretched_decks:
                    msg += f" (tempo corretto realmente su {stretched_decks} deck)"
                st.sidebar.success(msg)
            else:
                st.sidebar.info(
                    "Nessun segmento creato con il BPM impostato: aumenta il valore di 'Battute per segmento' "
                    "oppure prova la modalità 'Frenesia Casuale' — utile per brani senza ritmo marcato (classica, ambient)."
                )
    else:
        range_sec = st.sidebar.slider("Range durata (sec):", 0.05, 2.0, (0.8, 1.2), step=0.05)
        if st.sidebar.button("🌪️ Frulla Audio (Glitch)"):
            st.session_state.segments = []
            for k, d in active_decks.items():
                segs = get_random_segments(d['y'], d['sr'], range_sec[0], range_sec[1])
                for s in segs:
                    st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            invalidate_mix()
            if st.session_state.segments:
                st.sidebar.warning(f"Creati {len(st.session_state.segments)} micro-pezzi!")
            else:
                st.sidebar.info("Nessun segmento creato: prova un range di durata più ampio o più corto.")

    if st.session_state.segments:
        st.sidebar.divider()
        st.sidebar.subheader("2. Esportazione")
        durata_mix = st.sidebar.number_input("Durata Mix Finale (sec):", 10, 600, 60)

        if st.sidebar.button("🚀 GENERA MIX FINALE"):
            with st.spinner("Rimescolando il mazzo..."):
                all_segs = list(st.session_state.segments)
                random.shuffle(all_segs)

                # Tutti i deck sono già a TARGET_SR grazie al fix in analyze_track,
                # ma manteniamo un controllo esplicito e un fallback difensivo (fix bug #1).
                ref_sr = TARGET_SR
                mismatched = [s for s in all_segs if s['sr'] != ref_sr]
                if mismatched:
                    st.sidebar.warning(
                        f"{len(mismatched)} segmenti con sample rate diversa da {ref_sr}Hz: "
                        "verranno ricampionati per evitare artefatti di pitch/velocità."
                    )
                    for s in mismatched:
                        # librosa.resample ricampiona lungo l'ultimo asse: funziona correttamente
                        # anche su array stereo (2, n) senza mescolare i canali.
                        s['audio'] = librosa.resample(s['audio'], orig_sr=s['sr'], target_sr=ref_sr)
                        s['sr'] = ref_sr

                chosen = []
                curr_samples = 0
                target_samples = durata_mix * ref_sr

                # Guardia anti-loop-infinito (fix bug #2): esce comunque dopo un numero
                # ragionevole di tentativi anche se, per qualche motivo, i segmenti
                # scelti non facessero avanzare il conteggio.
                max_attempts = max(1000, len(all_segs) * 50)
                attempts = 0
                while curr_samples < target_samples and attempts < max_attempts:
                    pick = random.choice(all_segs)
                    # shape[1] = lunghezza temporale del segmento stereo (2, n): NON usare len(),
                    # che su un array (2, n) restituirebbe 2 (il numero di canali) invece della durata.
                    seg_len = pick['audio'].shape[1]
                    if seg_len > 0:
                        chosen.append(pick['audio'])
                        curr_samples += seg_len
                    attempts += 1

                if not chosen:
                    st.sidebar.error("Impossibile generare il mix: nessun segmento valido disponibile.")
                else:
                    # Concatenazione lungo l'asse temporale (axis=1), non lungo i canali.
                    final_y = np.concatenate(chosen, axis=1)
                    out = export_audio(final_y, ref_sr)

                    # --- GENERAZIONE REPORT BRANDIZZATO BILINGUE ---
                    ts_audio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.audio_report = f"""
╔════════════════════════════════════════════════════════════════╗
  HYPER-MIXER v3.3 - AUDIO RECONSTRUCTION LOG (STEREO + TIME-STRETCH)
  Generated on: {ts_audio}
╚════════════════════════════════════════════════════════════════╝

[AUDIO_RECONSTRUCTION_LOG] // VOL_01 // MP3 // 320kbps // STEREO

═══════════════════ ITALIANO ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v3.3]
:: ANALISI: Beat Tracking (Librosa) / RMS Envelope
:: STILE: Audio-Glitch / Granular Synthesis
:: PROCESSO: Shuffling Ricorsivo / Cross-Deck Fragmentation / Sample Rate Uniformato / Stereo Preservato

"Audio-Data fragment: Il ritmo è solo una variabile manipolata dal caos."

> SCHEDA TECNICA:
* Deck Attivi: {len(active_decks)} sorgenti caricate
* Pool Segmenti: {len(st.session_state.segments)} campioni estratti
* Modalità: {tipo_taglio}
* Campionamento: {ref_sr} Hz / Stereo (L/R preservati, mono duplicato se sorgente mono)
* Durata Output: {durata_mix}s

═══════════════════ ENGLISH ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v3.3]
:: ANALYSIS: Beat Tracking (Librosa) / RMS Envelope
:: STYLE: Audio-Glitch / Granular Synthesis
:: PROCESS: Recursive Shuffling / Cross-Deck Fragmentation / Uniform Sample Rate / Stereo Preserved

"Audio-Data fragment: Rhythm is just a variable manipulated by chaos."

> TECHNICAL LOG SHEET:
* Active Decks: {len(active_decks)} loaded sources
* Segments Pool: {len(st.session_state.segments)} extracted samples
* Mode: {tipo_taglio}
* Sampling: {ref_sr} Hz / Stereo (L/R preserved, mono duplicated if source was mono)
* Output Duration: {durata_mix}s

> Regia e Algoritmo / Direction & Algorithm: Loop507

#Loop507 #AudioGlitch #SoundDesign #GranularSynthesis #ExperimentalMusic
#AudioDecomposition #NoiseArt #SignalCorruption #RecursiveCollapse
"""
                    st.session_state.mix_ready = out

if st.session_state.get('mix_ready'):
    st.divider()
    st.subheader("🎵 Risultato del Mix")
    st.audio(st.session_state.mix_ready)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("📥 Scarica Mix MP3", st.session_state.mix_ready, "loop507_custom_mix.mp3", use_container_width=True)
    with col_d2:
        st.download_button("📄 Scarica Report Audio", st.session_state.audio_report, "audio_report.txt", use_container_width=True)

st.markdown("---")
st.caption("Loop507 Hyper-Mixer | Modalità Glitch & BPM attiva | Stereo + Time-Stretch v3.3")
