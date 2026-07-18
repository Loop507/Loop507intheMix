import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import tempfile
import os
import random
import shutil
import json
from datetime import datetime

try:
    import mido
    MIDI_DISPONIBILE = True
except ImportError:
    # mido non e' nei requirements: l'export MIDI si disattiva da sola invece di far
    # crashare tutta l'app. Se vedi questo messaggio, aggiungi 'mido' a requirements.txt.
    MIDI_DISPONIBILE = False

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
    # .copy() e' essenziale: senza, ogni segmento sarebbe solo una "vista" sulla memoria
    # dell'array originale, quindi anche liberando d['y'] la RAM non verrebbe mai rilasciata.
    return [
        y[:, i:i + segment_length].copy()
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
        segments.append(y[:, curr:curr + length].copy())
        curr += length
    return segments


def get_beat_segments_range(y, sr, tempo, min_beats, max_beats):
    """Come get_beat_segments, ma invece di un numero fisso di battute per taglio, ogni
    segmento usa una lunghezza casuale (in battute) pescata nel range [min_beats, max_beats].
    Utile per non avere sempre tagli tutti identici anche in modalità 'a ritmo'."""
    if tempo is None or tempo <= 1e-3:
        return []
    samples_per_beat = sr * 60 / tempo
    max_samples = int(MAX_SEGMENT_SECONDS * sr)
    total_samples = y.shape[1]
    lo, hi = (min_beats, max_beats) if min_beats <= max_beats else (max_beats, min_beats)

    segments = []
    curr = 0
    while curr < total_samples:
        beats = random.uniform(lo, hi)
        length = int(samples_per_beat * beats)
        length = max(MIN_SEGMENT_SAMPLES, min(length, max_samples))
        if curr + length > total_samples:
            break
        segments.append(y[:, curr:curr + length].copy())
        curr += length
    return segments


def get_onset_segments(y, sr, min_dur_sec=None, max_dur_sec=None):
    """Taglia ai transienti reali dell'audio (attacchi di nota, percussioni, cambi bruschi
    di energia) invece che su un grid rigido o su durate casuali. Stesso principio del
    backtrack sugli onset già usato in R13, qui applicato alla detection mono e poi
    proiettato sul taglio stereo. y ha shape (2, n).
    min_dur_sec/max_dur_sec (opzionali) scartano i segmenti troppo corti/lunghi risultanti
    dalla detection, per dare comunque un minimo di controllo sulla durata dei tagli."""
    total_samples = y.shape[1]
    if total_samples < MIN_SEGMENT_SAMPLES * 2:
        return []

    y_mono = np.mean(y, axis=0)
    onset_samples = librosa.onset.onset_detect(y=y_mono, sr=sr, backtrack=True, units="samples")

    # Servono almeno due punti (inizio e fine) per formare un segmento; se la detection
    # non trova transienti utili (es. drone, rumore bianco costante), niente di grave:
    # semplicemente questa modalità non produce nulla per questo deck.
    points = sorted(set([0, *onset_samples.tolist(), total_samples]))
    if len(points) < 2:
        return []

    max_samples = int(MAX_SEGMENT_SECONDS * sr)
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        end = min(end, start + max_samples)  # clamp di sicurezza anche qui (fix bug #5)
        length = end - start
        if length < MIN_SEGMENT_SAMPLES:
            continue
        dur = length / sr
        if min_dur_sec is not None and dur < min_dur_sec:
            continue
        if max_dur_sec is not None and dur > max_dur_sec:
            continue
        segments.append(y[:, start:end].copy())
    return segments


def get_leader_cut_lengths(y_leader, sr, leader_mode, tempo=None, num_beats=None,
                            beats_range=None, onset_min_dur=None, onset_max_dur=None):
    """Ricava dal deck 'leader' solo le LUNGHEZZE dei tagli (non l'audio), da imporre
    poi a tutti gli altri deck: e' la 'struttura' che i follower devono rispettare."""
    if leader_mode == "onset":
        segs = get_onset_segments(y_leader, sr, onset_min_dur, onset_max_dur)
    else:  # "bpm"
        if beats_range is not None:
            segs = get_beat_segments_range(y_leader, sr, tempo, beats_range[0], beats_range[1])
        else:
            segs = get_beat_segments(y_leader, sr, tempo, num_beats)
    return [s.shape[1] for s in segs]


def apply_cut_lengths(y, lengths):
    """Applica in sequenza una lista di lunghezze (derivata dal leader) a un deck qualsiasi,
    scorrendo il suo audio dall'inizio. Se il deck 'follower' finisce prima di esaurire
    tutte le lunghezze richieste, si ferma semplicemente li' — nessun loop forzato."""
    segments = []
    curr = 0
    total_samples = y.shape[1]
    for length in lengths:
        if length <= 0:
            continue
        if curr + length > total_samples:
            break
        segments.append(y[:, curr:curr + length].copy())
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


def concat_with_crossfade(segments, crossfade_samples):
    """Concatena i segmenti con un breve dissolvenza incrociata (overlap-add) tra ognuno,
    per evitare i click/pop tipici della concatenazione a taglio netto. Se crossfade_samples
    e' 0 o c'e' un solo segmento, equivale a una concatenazione normale."""
    if crossfade_samples <= 0 or len(segments) <= 1:
        return np.concatenate(segments, axis=1)

    result = segments[0].copy()
    for seg in segments[1:]:
        fade = min(crossfade_samples, result.shape[1], seg.shape[1])
        if fade <= 0:
            result = np.concatenate([result, seg], axis=1)
            continue
        fade_out = np.linspace(1.0, 0.0, fade)
        fade_in = np.linspace(0.0, 1.0, fade)
        overlap = result[:, -fade:] * fade_out + seg[:, :fade] * fade_in
        result = np.concatenate([result[:, :-fade], overlap, seg[:, fade:]], axis=1)
    return result


def get_beat_grid_samples(y, sr, tempo=None):
    """Rileva la griglia dei beat del deck (in campioni): e' la 'griglia magnetica' su cui
    agganciare gli overlay in DJ Remix, cosi' i picchi ritmici dei due audio combaciano
    davvero, come farebbe un DJ ad orecchio."""
    y_mono = np.mean(y, axis=0)
    try:
        _, beat_samples = librosa.beat.beat_track(y=y_mono, sr=sr, units="samples")
    except Exception:
        return []
    return sorted(beat_samples.tolist()) if len(beat_samples) else []


def find_strongest_onset_offset(seg, sr):
    """Trova, dentro un singolo frammento, la posizione (in campioni) del suo attacco più
    forte — il suo 'colpo' principale. E' il punto che vogliamo far combaciare esattamente
    con un beat del leader, non l'inizio arbitrario del frammento."""
    seg_mono = np.mean(seg, axis=0)
    try:
        onsets = librosa.onset.onset_detect(y=seg_mono, sr=sr, backtrack=True, units="samples")
    except Exception:
        return 0
    if len(onsets) == 0:
        return 0
    return int(onsets[0])  # il primo transiente e' tipicamente il "punch" principale


def build_dj_remix_overlay(leader_y, leader_sr, overlay_segments, overlay_gain, num_events,
                            rng, deck_weights=None, beatmatch=True):
    """Costruisce un mix 'letto + overlay': leader_y resta INTATTO come base (nessun taglio),
    e sopra vengono sparsi num_events frammenti presi da overlay_segments, sommati (non
    concatenati), moltiplicati per overlay_gain.
    Se beatmatch e' attivo: ogni frammento non va a un punto casuale qualsiasi, ma il suo
    attacco più forte viene agganciato esattamente a un beat della griglia del leader —
    il vero 'aggancio ai picchi' stile DJ, non solo lo stesso tempo.
    Ritorna anche la lista degli eventi piazzati (per l'export MIDI della struttura)."""
    buffer = leader_y.astype(np.float64).copy()
    total = buffer.shape[1]
    if not overlay_segments or total == 0:
        return buffer, 0, []

    beat_grid = get_beat_grid_samples(leader_y, leader_sr) if beatmatch else []
    if beatmatch and not beat_grid:
        beatmatch = False  # il leader non ha un beat rilevabile: fallback a piazzamento casuale

    weights_list = None
    if deck_weights:
        counts = {}
        for s in overlay_segments:
            counts[s['deck']] = counts.get(s['deck'], 0) + 1
        weights_list = [deck_weights.get(s['deck'], 5) / max(1, counts.get(s['deck'], 1)) for s in overlay_segments]

    placed = 0
    attempts = 0
    max_attempts = max(500, num_events * 20)  # guardia anti-loop-infinito, stessa logica del bug #2
    events = []  # per il report MIDI: (start_sample, deck)
    while placed < num_events and attempts < max_attempts:
        attempts += 1
        pick = rng.choices(overlay_segments, weights=weights_list, k=1)[0] if weights_list else rng.choice(overlay_segments)
        seg = pick['audio']
        seg_len = seg.shape[1]
        if seg_len <= 0 or seg_len > total:
            continue

        if beatmatch:
            beat_pos = rng.choice(beat_grid)
            # onset_offset e' pre-calcolato su ogni segmento (vedi sotto) per non rifare
            # la detection ad ogni singolo piazzamento, anche se lo stesso frammento
            # viene ripescato più volte.
            onset_offset = pick.get('onset_offset', 0)
            start = beat_pos - onset_offset
            if start < 0:
                start = 0
            if start + seg_len > total:
                start = total - seg_len
                if start < 0:
                    continue
        else:
            start = rng.randint(0, total - seg_len)

        buffer[:, start:start + seg_len] += seg.astype(np.float64) * overlay_gain
        events.append((start, pick['deck']))
        placed += 1
    return buffer, placed, events


def build_structure_midi(events, sr, tempo_bpm=120.0, base_note=36):
    """Genera un MIDI 'mappa strutturale' del mix: ogni evento (taglio o overlay) diventa
    una nota. L'altezza della nota varia per deck (A->36, B->37, ... come un mini drum-kit
    GM), cosi' aprendo il file in un DAW si vede a colpo d'occhio quale deck ha contribuito
    a quale istante della struttura. events: lista di (start_sample, deck_letter)."""
    if not MIDI_DISPONIBILE or not events:
        return None

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo_us = mido.bpm2tempo(max(20.0, tempo_bpm))
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))

    deck_to_note = {letter: base_note + i for i, letter in enumerate('abcdefgh')}
    note_duration_ticks = int(mido.second2tick(0.08, mid.ticks_per_beat, tempo_us))  # colpo breve

    # Ordino cronologicamente e converto in eventi assoluti su un'unica timeline (note_on / note_off)
    timeline = []
    for start_sample, deck in sorted(events, key=lambda e: e[0]):
        t_sec = start_sample / sr
        note = deck_to_note.get(deck, base_note)
        tick_on = int(mido.second2tick(t_sec, mid.ticks_per_beat, tempo_us))
        timeline.append((tick_on, 'note_on', note))
        timeline.append((tick_on + note_duration_ticks, 'note_off', note))
    timeline.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

    prev_tick = 0
    for tick, kind, note in timeline:
        delta = max(0, tick - prev_tick)
        vel = 100 if kind == 'note_on' else 0
        track.append(mido.Message(kind, note=note, velocity=vel, time=delta))
        prev_tick = tick

    buffer = BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    return buffer


def estimate_memory_mb(y):
    """Stima approssimativa della RAM occupata da un array audio (utile per avvisare
    l'utente prima che Streamlit Cloud vada in affanno con troppi deck lunghi)."""
    if y is None:
        return 0.0
    return y.nbytes / (1024 * 1024)


def invalidate_mix():
    """Invalida il mix precedente quando i segmenti cambiano (fix bug #4)."""
    st.session_state.pop("mix_ready", None)
    st.session_state.pop("mix_preset", None)
    st.session_state.pop("mix_events", None)
    st.session_state.pop("dj_rendered_by_deck", None)
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
                        st.session_state.pop(f"exported_rend_{k}", None)  # export renderizzato del vecchio file non più valido
                if st.session_state.decks[k]['y'] is not None:
                    st.success(f"{up.name}")
                    mem_mb = estimate_memory_mb(st.session_state.decks[k]['y'])
                    st.caption(f"💾 ~{mem_mb:.1f} MB in RAM")
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

                    if st.button(f"🎛️ Esporta Renderizzato {k.upper()}", key=f"export_rend_{k}", use_container_width=True,
                                 help="Il deck così com'è DOPO l'ultima elaborazione (taglio, time-stretch, allineamento DJ Remix)."):
                        with st.spinner(f"Esporto Deck {k.upper()} (renderizzato)..."):
                            dj_rendered = st.session_state.get('dj_rendered_by_deck')
                            if dj_rendered and k in dj_rendered.get('audio', {}):
                                # Priorità al materiale DJ Remix: e' il più "processato"
                                # (stretch al BPM del leader), altrimenti andrebbe perso.
                                pieces = dj_rendered['audio'][k]
                                render_sr = dj_rendered['sr']
                                render_y = concat_with_crossfade(pieces, 0)
                                st.session_state[f"exported_rend_{k}"] = export_audio(render_y, render_sr)
                            else:
                                # Fallback: concateno i segmenti di questo deck così come
                                # sono nel pool corrente (riflette l'ultimo taglio fatto,
                                # incluso l'eventuale time-stretch della modalità BPM).
                                pieces = [s['audio'] for s in st.session_state.segments if s['deck'] == k]
                                if pieces:
                                    render_sr = st.session_state.decks[k]['sr']
                                    render_y = concat_with_crossfade(pieces, 0)
                                    st.session_state[f"exported_rend_{k}"] = export_audio(render_y, render_sr)
                                else:
                                    st.session_state[f"exported_rend_{k}"] = None
                                    st.warning(f"Nessun materiale renderizzato per Deck {k.upper()}: taglialo prima con una delle modalità.")

                    if st.session_state.get(f"exported_rend_{k}"):
                        st.download_button(
                            f"📥 Scarica Renderizzato {k.upper()}.mp3", st.session_state[f"exported_rend_{k}"],
                            f"loop507_deck_{k}_renderizzato.mp3", key=f"dl_rend_{k}", use_container_width=True
                        )

                    if st.session_state.segments:
                        # Ha senso liberare la RAM solo dopo aver già estratto i segmenti:
                        # da questo momento i segmenti sono copie indipendenti (fix .copy()),
                        # quindi l'audio originale del deck non serve più finché non vuoi
                        # ri-tagliarlo con parametri diversi.
                        if st.button(f"🧹 Libera RAM Deck {k.upper()}", key=f"free_{k}"):
                            st.session_state.decks[k]['y'] = None
                            st.rerun()
                elif st.session_state.decks[k]['name'] == up.name:
                    # Il deck e' stato "liberato" volontariamente: i segmenti già estratti
                    # restano validi e utilizzabili nel mix, ma per ri-tagliare questo deck
                    # con parametri diversi serve ricaricare il file.
                    st.success(f"{up.name}")
                    st.caption("💾 RAM liberata — i segmenti già estratti restano disponibili per il mix.")
                    if st.button(f"🔄 Ricarica in RAM Deck {k.upper()}", key=f"reload_{k}"):
                        # analyze_track e' decorata con @st.cache_data: se il file non e' cambiato,
                        # questa chiamata NON rianalizza da zero, recupera solo il risultato già
                        # calcolato in precedenza — quindi è praticamente istantanea.
                        y, sr, t = analyze_track(up)
                        st.session_state.decks[k]['y'] = y
                        st.session_state.decks[k]['sr'] = sr
                        st.rerun()

st.sidebar.header("🎛️ Pannello di Controllo")
active_decks = {k: v for k, v in st.session_state.decks.items() if v['y'] is not None}

if active_decks:
    st.sidebar.subheader("1. Scegli Stile di Taglio")
    tipo_taglio = st.sidebar.radio(
        "Modalità:",
        [
            "Ritmo Musicale (BPM)",
            "Frenesia Casuale (Secondi)",
            "🎯 Slice per Transienti (Onset)",
            "👑 Deck Leader / Followers",
        ]
    )

    taglio_meta = {}  # metadati del taglio corrente, valorizzati nel ramo attivo qui sotto;
                      # evita di dover ri-derivare i parametri più avanti (rischio NameError
                      # ora che le modalità sono 4 e non più solo BPM/Casuale)

    if tipo_taglio == "Ritmo Musicale (BPM)":
        usa_range_battute = st.sidebar.checkbox(
            "🎲 Usa un range di battute invece di un valore fisso",
            value=False,
            help="Invece di tagliare sempre alla stessa lunghezza (es. sempre 2 battute), "
                 "ogni taglio pesca una lunghezza casuale nel range che imposti (es. tra 1 e 2 battute)."
        )
        if usa_range_battute:
            beats_range = st.sidebar.slider(
                "Battute per segmento (min-max):", 0.25, 8.0, (1.0, 2.0), step=0.25
            )
            beats = None
        else:
            beats = st.sidebar.selectbox("Battute per segmento:", [0.5, 1, 2, 4, 8], index=2)
            beats_range = None
        taglio_meta = {"modalita": "bpm", "beats": beats, "beats_range": list(beats_range) if beats_range else None}
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
                if beats_range is not None:
                    segs = get_beat_segments_range(y_for_cut, d['sr'], d['tempo'], beats_range[0], beats_range[1])
                else:
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
    elif tipo_taglio == "Frenesia Casuale (Secondi)":
        range_sec = st.sidebar.slider("Range durata (sec):", 0.05, 2.0, (0.8, 1.2), step=0.05)
        taglio_meta = {"modalita": "casuale", "range_sec": list(range_sec)}
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

    elif tipo_taglio == "🎯 Slice per Transienti (Onset)":
        st.sidebar.caption(
            "Taglia ogni deck nei suoi punti di attacco reali (percussioni, note, cambi di energia) "
            "invece che su un grid fisso o su durate casuali. Ogni deck produce segmenti di lunghezza "
            "naturale, dettata dal contenuto stesso — qui puoi solo scartare quelli troppo corti o lunghi."
        )
        onset_range = st.sidebar.slider(
            "Durata segmenti accettata (sec):", 0.02, 5.0, (0.08, 2.0), step=0.02,
            help="I transienti rilevati fuori da questo range vengono scartati. Non forza tutti i "
                 "segmenti alla stessa durata: filtra solo gli estremi."
        )
        taglio_meta = {"modalita": "onset", "durata_range_sec": list(onset_range)}
        if st.sidebar.button("⚡ Slice ai Transienti"):
            st.session_state.segments = []
            decks_senza_transienti = []
            for k, d in active_decks.items():
                with st.spinner(f"Rilevo transienti Deck {k.upper()}..."):
                    segs = get_onset_segments(d['y'], d['sr'], onset_range[0], onset_range[1])
                if not segs:
                    decks_senza_transienti.append(k.upper())
                for s in segs:
                    st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
            invalidate_mix()
            if st.session_state.segments:
                st.sidebar.success(f"Creati {len(st.session_state.segments)} frammenti dai transienti!")
                if decks_senza_transienti:
                    st.sidebar.info(
                        f"Deck {', '.join(decks_senza_transienti)}: nessun transiente nel range di durata "
                        "impostato (prova ad allargarlo) — nessun segmento da lì, non è un errore."
                    )
            else:
                st.sidebar.info(
                    "Nessun transiente rilevato su nessun deck: prova 'Ritmo Musicale' o "
                    "'Frenesia Casuale' per questo materiale."
                )

    else:  # "👑 Deck Leader / Followers"
        st.sidebar.caption(
            "Un deck 'conduttore' impone la propria griglia di tagli a tutti gli altri: ogni "
            "follower viene tagliato con le STESSE lunghezze del leader, applicate al proprio "
            "audio in sequenza dall'inizio. Un follower più corto del leader si ferma prima, "
            "senza loop forzati — è la sua materia che finisce, non un errore."
        )
        leader_key = st.sidebar.selectbox(
            "Deck Leader (conduttore):",
            options=list(active_decks.keys()),
            format_func=lambda k: f"Deck {k.upper()}"
        )
        leader_submode = st.sidebar.radio(
            "Il leader detta i tagli tramite:",
            ["BPM", "Transienti (Onset)"],
            horizontal=True
        )
        leader_beats = None
        leader_beats_range = None
        leader_onset_range = None
        apply_bpm_align = False
        if leader_submode == "BPM":
            leader_usa_range = st.sidebar.checkbox(
                "🎲 Range di battute per il leader (invece di un valore fisso)", value=False,
                key="leader_range_toggle"
            )
            if leader_usa_range:
                leader_beats_range = st.sidebar.slider(
                    "Battute per segmento leader (min-max):", 0.25, 8.0, (1.0, 2.0), step=0.25,
                    key="leader_beats_range"
                )
            else:
                leader_beats = st.sidebar.selectbox("Battute per segmento (leader):", [0.5, 1, 2, 4, 8], index=2)
            apply_bpm_align = st.sidebar.checkbox(
                "🎚️ Allinea davvero il BPM dei follower al leader (time-stretch)",
                value=False,
                help="Senza questa opzione, i follower vengono tagliati alle stesse LUNGHEZZE del "
                     "leader ma continuano a suonare alla loro velocità originale — la struttura è "
                     "sincronizzata, il tempo musicale no. Con questa opzione attiva, ogni follower "
                     "viene prima accelerato/rallentato (intonazione preservata) per portarlo davvero "
                     "al BPM del leader, poi tagliato. Più lento da calcolare, lievi artefatti."
            )
        else:
            leader_onset_range = st.sidebar.slider(
                "Durata tagli leader accettata (sec):", 0.02, 5.0, (0.08, 2.0), step=0.02,
                key="leader_onset_range"
            )

        taglio_meta = {
            "modalita": "leader_followers",
            "leader": leader_key,
            "leader_submode": leader_submode,
            "leader_beats": leader_beats,
            "leader_beats_range": list(leader_beats_range) if leader_beats_range else None,
            "leader_onset_range": list(leader_onset_range) if leader_onset_range else None,
            "bpm_align": apply_bpm_align,
        }

        if st.sidebar.button("👑 Applica Struttura del Leader"):
            leader_d = active_decks[leader_key]
            leader_mode = "bpm" if leader_submode == "BPM" else "onset"
            with st.spinner(f"Calcolo la struttura dal Deck {leader_key.upper()}..."):
                lengths = get_leader_cut_lengths(
                    leader_d['y'], leader_d['sr'], leader_mode,
                    tempo=leader_d['tempo'], num_beats=leader_beats,
                    beats_range=leader_beats_range,
                    onset_min_dur=leader_onset_range[0] if leader_onset_range else None,
                    onset_max_dur=leader_onset_range[1] if leader_onset_range else None,
                )
            if not lengths:
                st.sidebar.info(
                    f"Il Deck {leader_key.upper()} non produce una struttura utilizzabile con questo metodo "
                    "(BPM non impostato o nessun transiente rilevato). Prova l'altro sotto-metodo."
                )
            else:
                st.session_state.segments = []
                follower_vuoti = []
                allineati = 0
                for k, d in active_decks.items():
                    y_for_cut = d['y']
                    # Allineo il BPM reale del follower a quello del leader solo se: l'utente lo ha
                    # attivato, siamo in sotto-modo BPM (ha senso solo con un riferimento numerico),
                    # non è il leader stesso, e il follower ha un BPM effettivo valido da cui partire.
                    if apply_bpm_align and leader_mode == "bpm" and k != leader_key and d['tempo'] > 0:
                        rate = leader_d['tempo'] / d['tempo']
                        with st.spinner(f"Allineo BPM Deck {k.upper()} al leader (rate {rate:.3f}x)..."):
                            y_for_cut = time_stretch_stereo(d['y'], rate)
                        allineati += 1
                    segs = apply_cut_lengths(y_for_cut, lengths)
                    if not segs:
                        follower_vuoti.append(k.upper())
                    for s in segs:
                        st.session_state.segments.append({'audio': s, 'sr': d['sr'], 'deck': k})
                invalidate_mix()
                if st.session_state.segments:
                    msg = (
                        f"Struttura del Deck {leader_key.upper()} ({len(lengths)} tagli) imposta a "
                        f"{len(active_decks)} deck — {len(st.session_state.segments)} segmenti totali."
                    )
                    if allineati:
                        msg += f" BPM realmente allineato su {allineati} follower."
                    st.sidebar.success(msg)
                    if follower_vuoti:
                        st.sidebar.info(
                            f"Deck {', '.join(follower_vuoti)}: troppo corti per seguire questa struttura, "
                            "nessun segmento prodotto da lì."
                        )
                else:
                    st.sidebar.error("Nessun deck è riuscito a seguire questa struttura: prova un leader diverso.")

    if st.session_state.segments:
        st.sidebar.divider()
        st.sidebar.subheader("2. Esportazione")

        # Pesi per deck: un deck con pochi segmenti (es. un loop corto) non deve sparire
        # nel mix solo perché un altro deck ne ha prodotti molti di più.
        contributing_decks = sorted({s['deck'] for s in st.session_state.segments})
        deck_weights = {}
        with st.sidebar.expander("⚖️ Peso dei deck nel mix"):
            st.caption("Peso più alto = quel deck viene pescato più spesso, a prescindere da quanti segmenti ha prodotto.")
            for dk in contributing_decks:
                deck_weights[dk] = st.slider(f"Deck {dk.upper()}", 1, 10, 5, key=f"weight_{dk}")

        apply_dj_remix = st.sidebar.checkbox(
            "🎙️ Modalità DJ Remix (leader intatto + overlay)",
            value=False,
            help="Un deck resta la traccia intera, non tagliata: fa da base. Sopra ci vengono "
                 "sparsi, sovrapposti (non incollati in sequenza), i frammenti degli ALTRI deck. "
                 "La durata del mix diventa quella del leader, non un valore a piacere."
        )

        dj_leader_key = None
        overlay_gain = 0.6
        num_overlay_events = 60
        dj_bpm_align = True
        dj_beatmatch = True
        if apply_dj_remix:
            dj_leader_key = st.sidebar.selectbox(
                "Deck da tenere intatto (base):",
                options=list(active_decks.keys()),
                format_func=lambda k: f"Deck {k.upper()}",
                key="dj_leader_key"
            )
            overlay_gain = st.sidebar.slider(
                "🔊 Volume overlay (rispetto al leader):", 0.1, 1.5, 0.6, step=0.05,
                help="Sotto 1.0 = gli overlay stanno 'sotto' il leader nel mix. Sopra 1.0 = lo sovrastano."
            )
            num_overlay_events = st.sidebar.slider(
                "🎯 Densità overlay (numero di frammenti sparsi):", 5, 400, 60
            )
            dj_bpm_align = st.sidebar.checkbox(
                "🎚️ Allinea automaticamente il BPM di tutti i follower al leader", value=True,
                help="Attivo di default: ogni deck (tranne il leader) viene accelerato/rallentato "
                     "col time-stretch per suonare davvero allo stesso tempo del leader, prima "
                     "di essere sparso come overlay."
            )
            dj_beatmatch = st.sidebar.checkbox(
                "🎯 Aggancia i picchi ritmici (beatmatching)", value=True,
                help="Attivo di default: ogni frammento overlay non viene piazzato a un punto "
                     "casuale, ma il suo attacco più forte viene fatto combaciare esattamente "
                     "con un beat del leader — proprio come farebbe un DJ mixando ad orecchio."
            )
            st.sidebar.caption(
                f"I frammenti verranno presi da tutti i deck TRANNE {dj_leader_key.upper() if dj_leader_key else '—'} "
                "(quello resta la base intatta)."
            )

        apply_crossfade = st.sidebar.checkbox(
            "🎛️ Crossfade tra i tagli",
            value=False,
            disabled=apply_dj_remix,
            help="Breve dissolvenza incrociata tra un segmento e il successivo, per evitare "
                 "click/pop secchi. Disattivalo se vuoi l'effetto glitch più tagliente e crudo. "
                 "Non si applica in Modalità DJ Remix (qui si sovrappone, non si concatena)."
        )
        crossfade_ms = st.sidebar.slider("Durata crossfade (ms)", 1, 100, 15, disabled=not apply_crossfade or apply_dj_remix) if apply_crossfade and not apply_dj_remix else 0

        seed_input = st.sidebar.number_input(
            "🎲 Seed (0 = casuale ogni volta)", min_value=0, value=0, step=1,
            help="Imposta un numero per rendere il mix riproducibile. Il seed usato viene "
                 "salvato nel preset scaricabile, così puoi ritrovare un mix riuscito per caso."
        )

        if not apply_dj_remix:
            durata_mix = st.sidebar.number_input("Durata Mix Finale (sec):", 10, 600, 60)
        else:
            durata_mix = None  # in DJ Remix la durata è quella del leader, non impostabile

        if st.sidebar.button("🚀 GENERA MIX FINALE"):
            with st.spinner("Rimescolando il mazzo..."):
                seed_used = seed_input if seed_input > 0 else random.randint(1, 2**31 - 1)
                rng = random.Random(seed_used)  # RNG locale e seedato: rende il mix riproducibile
                ref_sr = TARGET_SR

                if apply_dj_remix:
                    # --- RAMO DJ REMIX: leader intatto come base, overlay sopra ---
                    dj_leader_d = active_decks[dj_leader_key]
                    overlay_pool_raw = [s for s in st.session_state.segments if s['deck'] != dj_leader_key]

                    if not overlay_pool_raw:
                        st.sidebar.error(
                            f"Nessun segmento disponibile dagli altri deck per l'overlay: servono "
                            f"segmenti già tagliati da almeno un deck diverso da {dj_leader_key.upper()}."
                        )
                        chosen = []
                    else:
                        ref_sr = dj_leader_d['sr']
                        # Ricampiono eventuali segmenti overlay con sr diverso da quello del leader.
                        mismatched = [s for s in overlay_pool_raw if s['sr'] != ref_sr]
                        if mismatched:
                            for s in mismatched:
                                s['audio'] = librosa.resample(s['audio'], orig_sr=s['sr'], target_sr=ref_sr)
                                s['sr'] = ref_sr

                        # --- Auto-allineamento BPM: ogni follower viene stretchato al tempo del
                        # leader PRIMA di essere sparso come overlay, cosi' suona davvero alla
                        # stessa velocità (non solo tagliato alla stessa lunghezza). ---
                        overlay_pool = []
                        decks_allineati = set()
                        decks_non_allineabili = set()
                        for s in overlay_pool_raw:
                            seg_audio = s['audio']
                            if dj_bpm_align and dj_leader_d['tempo'] > 0:
                                follower_tempo = active_decks.get(s['deck'], {}).get('tempo', 0)
                                if follower_tempo > 0 and abs(dj_leader_d['tempo'] - follower_tempo) > 0.5:
                                    rate = dj_leader_d['tempo'] / follower_tempo
                                    seg_audio = time_stretch_stereo(seg_audio, rate)
                                    decks_allineati.add(s['deck'])
                                elif follower_tempo <= 0:
                                    decks_non_allineabili.add(s['deck'])
                            overlay_pool.append({'audio': seg_audio, 'sr': ref_sr, 'deck': s['deck']})

                        # --- Pre-calcolo dell'attacco più forte di ogni segmento, una volta sola,
                        # cosi' il beatmatching non deve rifare la detection ad ogni piazzamento
                        # anche se lo stesso frammento viene ripescato più volte. ---
                        if dj_beatmatch:
                            with st.spinner("Rilevo i picchi ritmici per l'aggancio..."):
                                for s in overlay_pool:
                                    s['onset_offset'] = find_strongest_onset_offset(s['audio'], ref_sr)

                        with st.spinner("Sovrappongo i frammenti (DJ Remix)..."):
                            final_y, eventi_piazzati, dj_events = build_dj_remix_overlay(
                                dj_leader_d['y'], ref_sr, overlay_pool, overlay_gain, num_overlay_events,
                                rng, deck_weights, beatmatch=dj_beatmatch
                            )
                        chosen = [True]  # solo per riusare il check "if not chosen" sotto

                        # Salvo il materiale POST-allineamento (stretch al BPM del leader, se
                        # applicato) raggruppato per deck: e' quello che "Esporta Renderizzato"
                        # userà, invece dell'audio originale mai toccato.
                        rendered_by_deck = {}
                        for s in overlay_pool:
                            rendered_by_deck.setdefault(s['deck'], []).append(s['audio'])
                        st.session_state.dj_rendered_by_deck = {'sr': ref_sr, 'audio': rendered_by_deck}
                else:
                    # --- RAMO CLASSICO: shuffle + concatenazione (+ crossfade opzionale) ---
                    all_segs = list(st.session_state.segments)
                    rng.shuffle(all_segs)

                    # Tutti i deck sono già a TARGET_SR grazie al fix in analyze_track,
                    # ma manteniamo un controllo esplicito e un fallback difensivo (fix bug #1).
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
                    dj_events = []  # riuso lo stesso nome del ramo DJ Remix per il MIDI

                    # Peso effettivo per segmento: peso_deck diviso per quanti segmenti quel deck
                    # ha prodotto, cosi' il peso impostato dall'utente vale per il deck nel suo
                    # complesso e non viene "diluito" se il deck ha tanti segmenti piccoli.
                    deck_counts = {}
                    for s in all_segs:
                        deck_counts[s['deck']] = deck_counts.get(s['deck'], 0) + 1
                    weights_list = [
                        deck_weights.get(s['deck'], 5) / max(1, deck_counts.get(s['deck'], 1))
                        for s in all_segs
                    ]

                    # Guardia anti-loop-infinito (fix bug #2): esce comunque dopo un numero
                    # ragionevole di tentativi anche se, per qualche motivo, i segmenti
                    # scelti non facessero avanzare il conteggio.
                    max_attempts = max(1000, len(all_segs) * 50)
                    attempts = 0
                    while curr_samples < target_samples and attempts < max_attempts:
                        pick = rng.choices(all_segs, weights=weights_list, k=1)[0]
                        # shape[1] = lunghezza temporale del segmento stereo (2, n): NON usare len(),
                        # che su un array (2, n) restituirebbe 2 (il numero di canali) invece della durata.
                        seg_len = pick['audio'].shape[1]
                        if seg_len > 0:
                            dj_events.append((curr_samples, pick['deck']))  # posizione nel mix finale
                            chosen.append(pick['audio'])
                            curr_samples += seg_len
                        attempts += 1

                    if chosen:
                        # Concatenazione lungo l'asse temporale, con crossfade opzionale per
                        # evitare i click secchi tra un segmento e il successivo.
                        crossfade_samples = int(ref_sr * crossfade_ms / 1000) if apply_crossfade else 0
                        final_y = concat_with_crossfade(chosen, crossfade_samples)
                        eventi_piazzati = len(chosen)

                if not chosen:
                    st.sidebar.error("Impossibile generare il mix: nessun segmento valido disponibile.")
                else:
                    out = export_audio(final_y, ref_sr)
                    durata_effettiva = final_y.shape[1] / ref_sr

                    # Salvati per l'eventuale export MIDI della struttura (vedi sezione risultato).
                    st.session_state.mix_events = dj_events
                    st.session_state.mix_events_sr = ref_sr
                    st.session_state.mix_events_tempo = (
                        dj_leader_d['tempo'] if apply_dj_remix and dj_leader_d['tempo'] > 0 else 120.0
                    )

                    # --- PRESET RIPRODUCIBILE (seed + parametri) ---
                    preset = {
                        "loop507_hyper_mixer_preset": True,
                        "versione_app": "5.1",
                        "seed": seed_used,
                        "modalita_taglio": tipo_taglio,
                        "parametro_taglio": taglio_meta,
                        "dj_remix": {
                            "attivo": apply_dj_remix,
                            "leader": dj_leader_key,
                            "overlay_gain": overlay_gain,
                            "num_eventi": num_overlay_events,
                            "bpm_align": dj_bpm_align,
                            "beatmatch": dj_beatmatch,
                        } if apply_dj_remix else None,
                        "crossfade_ms": crossfade_ms if (apply_crossfade and not apply_dj_remix) else 0,
                        "pesi_deck": deck_weights,
                        "durata_mix_sec": round(durata_effettiva, 2),
                        "deck": [
                            {"deck": k, "file": d['name'], "bpm_usato": d['tempo']}
                            for k, d in active_decks.items()
                        ],
                        "nota": (
                            "Per riprovare a riottenere questo stesso mix: ricarica gli stessi file "
                            "negli stessi deck, imposta la stessa modalità/parametro di taglio e gli "
                            "stessi pesi, poi inserisci questo seed prima di generare."
                        )
                    }
                    st.session_state.mix_preset = json.dumps(preset, ensure_ascii=False, indent=2)

                    # --- GENERAZIONE REPORT BRANDIZZATO BILINGUE ---
                    ts_audio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if apply_dj_remix:
                        allineamento_info = ""
                        if dj_bpm_align:
                            n_align = len(decks_allineati)
                            allineamento_info = f" / BPM allineato su {n_align} follower" if n_align else " / nessun follower necessitava allineamento"
                        beatmatch_info = " / Beatmatching attivo (picchi agganciati)" if dj_beatmatch else " / Beatmatching disattivato (piazzamento casuale)"
                        processo_it = f"DJ Remix: Deck {dj_leader_key.upper()} intatto come base / {eventi_piazzati} overlay sparsi (seed {seed_used}){allineamento_info}{beatmatch_info} / Stereo Preservato"
                        processo_en = f"DJ Remix: Deck {dj_leader_key.upper()} kept intact as base / {eventi_piazzati} scattered overlays (seed {seed_used}) / Stereo Preserved"
                        extra_it = f"* Deck Base (intatto): {dj_leader_key.upper()}\n* Overlay Piazzati: {eventi_piazzati} (volume {overlay_gain:.2f}x)\n* Allineamento BPM: {'attivo' if dj_bpm_align else 'disattivo'}\n* Beatmatching: {'attivo' if dj_beatmatch else 'disattivo'}"
                        extra_en = f"* Base Deck (intact): {dj_leader_key.upper()}\n* Overlays Placed: {eventi_piazzati} (gain {overlay_gain:.2f}x)\n* BPM Alignment: {'on' if dj_bpm_align else 'off'}\n* Beatmatching: {'on' if dj_beatmatch else 'off'}"
                    else:
                        processo_it = f"Shuffling Ricorsivo (seed {seed_used}) / Cross-Deck Fragmentation / Sample Rate Uniformato / Stereo Preservato / Pesi Deck Personalizzati"
                        processo_en = f"Recursive Shuffling (seed {seed_used}) / Cross-Deck Fragmentation / Uniform Sample Rate / Stereo Preserved / Custom Deck Weights"
                        extra_it = f"* Crossfade: {f'{crossfade_ms}ms' if apply_crossfade else 'disattivato'}"
                        extra_en = f"* Crossfade: {f'{crossfade_ms}ms' if apply_crossfade else 'disabled'}"

                    st.session_state.audio_report = f"""
╔════════════════════════════════════════════════════════════════╗
  HYPER-MIXER v5.1 - AUDIO RECONSTRUCTION LOG (STEREO + DJ REMIX + MIDI)
  Generated on: {ts_audio}
╚════════════════════════════════════════════════════════════════╝

[AUDIO_RECONSTRUCTION_LOG] // VOL_01 // MP3 // 320kbps // STEREO

═══════════════════ ITALIANO ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v5.1]
:: ANALISI: Beat Tracking (Librosa) / RMS Envelope / Onset Detection
:: STILE: Audio-Glitch / Granular Synthesis
:: PROCESSO: {processo_it}

"Audio-Data fragment: Il ritmo è solo una variabile manipolata dal caos."

> SCHEDA TECNICA:
* Deck Attivi: {len(active_decks)} sorgenti caricate
* Pool Segmenti: {len(st.session_state.segments)} campioni estratti
* Modalità Taglio: {tipo_taglio}
* Campionamento: {ref_sr} Hz / Stereo (L/R preservati, mono duplicato se sorgente mono)
{extra_it}
* Seed: {seed_used}
* Durata Output: {durata_effettiva:.1f}s

═══════════════════ ENGLISH ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v5.1]
:: ANALYSIS: Beat Tracking (Librosa) / RMS Envelope / Onset Detection
:: STYLE: Audio-Glitch / Granular Synthesis
:: PROCESS: {processo_en}

"Audio-Data fragment: Rhythm is just a variable manipulated by chaos."

> TECHNICAL LOG SHEET:
* Active Decks: {len(active_decks)} loaded sources
* Segments Pool: {len(st.session_state.segments)} extracted samples
* Cut Mode: {tipo_taglio}
* Sampling: {ref_sr} Hz / Stereo (L/R preserved, mono duplicated if source was mono)
{extra_en}
* Seed: {seed_used}
* Output Duration: {durata_effettiva:.1f}s

> Regia e Algoritmo / Direction & Algorithm: Loop507

#Loop507 #AudioGlitch #SoundDesign #GranularSynthesis #ExperimentalMusic
#AudioDecomposition #NoiseArt #SignalCorruption #RecursiveCollapse
"""
                    st.session_state.mix_ready = out

if st.session_state.get('mix_ready'):
    st.divider()
    st.subheader("🎵 Risultato del Mix")
    st.audio(st.session_state.mix_ready)
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.download_button("📥 Scarica Mix MP3", st.session_state.mix_ready, "loop507_custom_mix.mp3", use_container_width=True)
    with col_d2:
        st.download_button("📄 Scarica Report Audio", st.session_state.audio_report, "audio_report.txt", use_container_width=True)
    with col_d3:
        st.download_button(
            "🎲 Scarica Preset (seed)", st.session_state.get('mix_preset', '{}'),
            "loop507_preset.json", use_container_width=True,
            help="Contiene il seed e i parametri usati: ricaricalo per riprovare a ritrovare questo mix."
        )
    with col_d4:
        if MIDI_DISPONIBILE:
            midi_buffer = build_structure_midi(
                st.session_state.get('mix_events', []),
                st.session_state.get('mix_events_sr', TARGET_SR),
                st.session_state.get('mix_events_tempo', 120.0),
            )
            if midi_buffer:
                st.download_button(
                    "🎹 Scarica Struttura MIDI", midi_buffer, "loop507_struttura.mid",
                    use_container_width=True,
                    help="Ogni taglio/overlay diventa una nota (l'altezza varia per deck: A=36, B=37, ...). "
                         "Utile per importare la struttura ritmica del mix in un DAW."
                )
            else:
                st.caption("🎹 Nessuna struttura MIDI disponibile per questo mix.")
        else:
            st.caption("🎹 Export MIDI non disponibile: aggiungi 'mido' a requirements.txt.")

st.markdown("---")
st.caption("Loop507 Hyper-Mixer | Modalità Glitch & BPM attiva | Stereo + DJ Remix + MIDI v5.1")
