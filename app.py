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
import zipfile
from datetime import datetime

try:
    from audiocomplib import AudioCompressor
    COMPRESSOR_DISPONIBILE = True
except ImportError:
    # audiocomplib non e' nei requirements: la compressione finale si disattiva da sola.
    # Se vedi questo messaggio, aggiungi 'audiocomplib' a requirements.txt.
    COMPRESSOR_DISPONIBILE = False

try:
    from scipy.signal import butter, sosfiltfilt
    SCIPY_DISPONIBILE = True
except ImportError:
    # scipy e' quasi sempre installato insieme a librosa, ma per sicurezza l'EQ Bass Swap
    # si disattiva da sola invece di far crashare tutta l'app se manca.
    SCIPY_DISPONIBILE = False

try:
    from mutagen import File as MutagenFile
    MUTAGEN_DISPONIBILE = True
except ImportError:
    # mutagen non e' nei requirements: l'artista nel report resta '—' invece di crashare.
    # Se vedi questo messaggio, aggiungi 'mutagen' a requirements.txt.
    MUTAGEN_DISPONIBILE = False

try:
    import mido
    MIDI_DISPONIBILE = True
except ImportError:
    # mido non e' nei requirements: l'export MIDI si disattiva da sola invece di far
    # crashare tutta l'app. Se vedi questo messaggio, aggiungi 'mido' a requirements.txt.
    MIDI_DISPONIBILE = False

try:
    import pyloudnorm as pyln
    LOUDNORM_DISPONIBILE = True
except ImportError:
    # pyloudnorm non e' nei requirements: la normalizzazione LUFS si disattiva da sola.
    # Se vedi questo messaggio, aggiungi 'pyloudnorm' a requirements.txt.
    LOUDNORM_DISPONIBILE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # backend headless: niente display, necessario su server tipo Streamlit Cloud
    import matplotlib.pyplot as plt
    MATPLOTLIB_DISPONIBILE = True
except ImportError:
    # matplotlib non e' nei requirements: l'export dell'immagine onda si disattiva da sola.
    # Se vedi questo messaggio, aggiungi 'matplotlib' a requirements.txt.
    MATPLOTLIB_DISPONIBILE = False

try:
    import cv2
    CV2_DISPONIBILE = True
except ImportError:
    # opencv non e' nei requirements: l'export del video onda si disattiva da sola.
    # Se vedi questo messaggio, aggiungi 'opencv-python-headless' a requirements.txt.
    CV2_DISPONIBILE = False

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

# Versione unica dell'app: usata in preset, report e footer (prima era sparsa in 3 punti con valori diversi)
APP_VERSION = "5.11"
NOME_BASE_OUT = "loop507_custom_mix"  # unico nome base per MP3, video e report (cambialo solo qui)
DOWNBEAT_MIN_CONFIDENCE = 0.15  # sotto questa confidenza la stima del downbeat viene scartata

# --- Funzioni di Analisi e Taglio ---
# NOTA STEREO: da qui in poi ogni array audio 'y' ha shape (2, n_samples) -> [canale L, canale R].
# I brani mono vengono duplicati sui due canali in fase di analisi, cosi' tutta la pipeline
# a valle (taglio, shuffle, export) lavora sempre e solo su array stereo, senza casi speciali.

def refine_tempo_from_beats(beat_samples, sr, fallback_tempo):
    """Il BPM restituito da beat_track è una stima algoritmica globale, spesso imprecisa.
    Le posizioni dei singoli beat rilevati sono generalmente più affidabili: calcoliamo la
    MEDIANA (non la media, per non farci rovinare da un singolo colpo fuori tempo) degli
    intervalli reali tra un beat e il successivo, e ne ricaviamo un BPM più preciso."""
    if beat_samples is None or len(beat_samples) < 4:
        return fallback_tempo  # troppo pochi beat per una stima affidabile: meglio il fallback
    beat_samples_sorted = np.sort(np.asarray(beat_samples, dtype=np.float64))
    intervals_sec = np.diff(beat_samples_sorted) / sr
    intervals_sec = intervals_sec[intervals_sec > 0]
    if len(intervals_sec) == 0:
        return fallback_tempo
    median_interval = np.median(intervals_sec)
    if median_interval <= 0:
        return fallback_tempo
    return 60.0 / median_interval


def correggi_errore_ottava(tempo_val, bpm_min, bpm_max):
    """I beat-tracker confondono spesso il doppio o la metà del BPM reale (es. rilevano 70
    invece di 140, o viceversa) — è l'errore più comune di questi algoritmi. Se il BPM
    rilevato cade fuori dal range plausibile impostato, lo dimezza/raddoppia finché non
    rientra, invece di lasciarlo palesemente sbagliato."""
    if tempo_val is None or tempo_val <= 0:
        return tempo_val
    t = tempo_val
    guard = 0  # sicurezza anti-loop, non dovrebbe mai servire con range ragionevoli
    while t > bpm_max and t / 2 >= bpm_min and guard < 10:
        t /= 2
        guard += 1
    while t < bpm_min and t * 2 <= bpm_max and guard < 10:
        t *= 2
        guard += 1
    return t


def check_tempo_ambiguity(y, sr, tempo_val):
    """Verifica se esistono BPM alternativi plausibili con forza ritmica comparabile a quello
    rilevato — non solo il classico raddoppio/dimezzamento (2:1), ma anche i rapporti 3:2 e 2:3,
    tipici di ritmi sincopati/shuffle che possono ingannare il beat-tracker facendogli agganciare
    una sottodivisione sbagliata (es. rileva 92 invece di 138 BPM, rapporto 3:2, non 2:1 — quindi
    la correzione ottava classica non lo intercetta). Confronta l'autocorrelazione dell'onset
    envelope al periodo del BPM rilevato contro quella ai periodi candidati alternativi: se un
    alternativo ha una forza comparabile o superiore, lo segnala come possibile ambiguità, senza
    correggerlo automaticamente — qui la decisione finale spetta a un orecchio umano (o ad una
    lettura del BPM dal tag ID3 del brano), non a un'euristica cieca."""
    if tempo_val is None or tempo_val <= 0:
        return []
    try:
        y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        if len(onset_env) < 16:
            return []
        # Tolgo la media PRIMA di autocorrelare: onset_strength e' sempre >= 0, quindi
        # l'autocorrelazione grezza ha un grosso offset costante (DC) presente a OGNI lag. Quel
        # piedistallo avvicina a 1 il rapporto tra forza alternativa e primaria e faceva scattare
        # molti falsi "possibile ambiguità". Normalizzo anche per ac[0] (energia), cosi' i valori
        # sono confrontabili (1.0 = perfettamente periodico a quel lag).
        oenv = onset_env - np.mean(onset_env)
        ac = librosa.autocorrelate(oenv, max_size=len(oenv))
        if ac[0] <= 1e-9:
            return []
        ac = ac / ac[0]
        hop_length = 512  # default di librosa.onset.onset_strength
        fps = sr / hop_length

        def forza_a_bpm(bpm):
            if bpm <= 0:
                return 0.0
            # Picco entro ±3% del periodo teorico: arrotondare a UN solo frame (circa 2% di errore
            # a 120 BPM) rischiava di campionare di lato al picco vero e sottostimarne la forza.
            periodo = fps * 60.0 / bpm
            lo = max(1, int(np.floor(periodo * 0.97)))
            hi = min(len(ac) - 1, int(np.ceil(periodo * 1.03)))
            if lo > hi:
                return 0.0
            return float(np.max(ac[lo:hi + 1]))

        forza_primaria = forza_a_bpm(tempo_val)
        if forza_primaria < 0.1:
            return []

        candidati = []
        for ratio, nome in [(0.5, "metà"), (2.0, "doppio"), (1.5, "3/2"), (2 / 3, "2/3")]:
            bpm_alt = tempo_val * ratio
            forza_alt = forza_a_bpm(bpm_alt)
            if forza_alt >= forza_primaria * 0.85:  # comparabile o più forte del rilevato
                candidati.append((round(bpm_alt, 1), nome, round(forza_alt / forza_primaria, 2)))
        candidati.sort(key=lambda c: -c[2])
        return candidati
    except Exception:
        return []


def classifica_rapporto(a, b, tolleranza=0.03):
    """Classifica il rapporto b/a in una categoria nota: 'uguale', 'metà', 'doppio', '3/2',
    '2/3', o 'diverso' (nessun rapporto noto)."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    rapporto = b / a
    if abs(rapporto - 1.0) < tolleranza:
        return "uguale"
    for r, nome in [(0.5, "metà"), (2.0, "doppio"), (1.5, "3/2"), (2 / 3, "2/3")]:
        if abs(rapporto - r) < tolleranza:
            return nome
    return "diverso"


IOI_MIN_PROMINENZA = 1.8  # picco migliore / mediana dei candidati: sotto, nessuna pulsazione chiara


def ioi_tempo_estimate(y, sr, bpm_min=60.0, bpm_max=200.0, _debug=False):
    """Terzo parere sul BPM, solo NumPy/SciPy/librosa (sostituisce aubio, che non si compila su
    Python 3.13). Tecnica nel DOMINIO DEL TEMPO, volutamente diversa dagli altri due metodi:
    il beat-tracking dinamico e librosa.feature.tempo lavorano sulla PERIODICITA' dell'inviluppo
    (autocorrelazione), questo invece conta gli ATTACCHI:
      1) inviluppo HFC (High Frequency Content: energia pesata per la frequenza, che esalta i
         transienti di kick/rullante/hi-hat — lo stesso principio di rilevamento di aubio) e flusso
         positivo di quell'inviluppo;
      2) picchi = onset discreti, con la loro forza;
      3) istogramma degli intervalli tra TUTTE le coppie di onset entro 2.5s (non solo i
         consecutivi: Dixon, 2001), pesato dalla forza degli attacchi e levigato;
      4) per ogni BPM candidato si somma l'istogramma ai multipli del periodo (1x,2x,3x,4x) e,
         con peso minore, alla sua metà (le suddivisioni): un tempo vero ha picchi a tutti i multipli,
         una sua ottava sbagliata no, quindi il confronto risolve gli errori d'ottava;
      5) prior log-normale centrata su 120 BPM (larghezza 1 ottava) per i casi residui.
    Ritorna il BPM o None se non ci sono abbastanza attacchi o nessun candidato emerge."""
    try:
        y_mono = np.mean(y, axis=0) if getattr(y, 'ndim', 1) == 2 else y
        max_n = int(120 * sr)  # estratto centrale: oltre 2 minuti non servono, costa solo tempo
        if len(y_mono) > max_n:
            a = (len(y_mono) - max_n) // 2
            y_mono = y_mono[a:a + max_n]
        n_fft, hop = 1024, 512
        if len(y_mono) < n_fft * 8:
            return None
        S = np.abs(librosa.stft(y_mono.astype(np.float32), n_fft=n_fft, hop_length=hop))
        k = np.arange(S.shape[0], dtype=np.float32)[:, None]
        hfc = np.sum((S ** 2) * k, axis=0)
        hfc = np.log1p(1000.0 * hfc / (np.max(hfc) + 1e-12))     # compressione: attacchi deboli contano
        flusso = np.maximum(0.0, np.diff(hfc, prepend=hfc[0]))    # solo gli AUMENTI di energia
        env = np.convolve(flusso, np.ones(3) / 3.0, mode='same')
        fps = sr / hop
        from scipy.signal import find_peaks
        picchi, _ = find_peaks(env, height=float(np.mean(env) + 0.5 * np.std(env)),
                               distance=max(1, int(0.05 * fps)))
        if len(picchi) < 12:
            return None
        t = picchi / fps
        a_on = env[picchi]
        # istogramma IOI (bin da 10ms) fino a 2.5s
        bin_s = 0.01
        n_bins = int(2.5 / bin_s) + 1
        H = np.zeros(n_bins)
        for i in range(len(t) - 1):
            j = i + 1
            while j < len(t) and t[j] - t[i] <= 2.5:
                b = int(round((t[j] - t[i]) / bin_s))
                if b < n_bins:
                    H[b] += np.sqrt(a_on[i] * a_on[j])
                j += 1
        if H.sum() <= 0:
            return None
        sig = 2  # ~20ms: tollera il leggero jitter degli onset
        ker = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
        H = np.convolve(H, ker / ker.sum(), mode='same')
        H = H / (H.sum() + 1e-12)
        asse = np.arange(n_bins) * bin_s

        def h(sec):
            return float(np.interp(sec, asse, H)) if 0.0 < sec <= 2.5 else 0.0

        bpms = np.arange(bpm_min, bpm_max + 0.01, 0.25)
        punteggi = np.zeros(len(bpms))
        for idx, bpm in enumerate(bpms):
            P = 60.0 / bpm
            s = h(P) + 0.7 * h(2 * P) + 0.5 * h(3 * P) + 0.4 * h(4 * P) + 0.3 * h(P / 2.0)
            prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.0) ** 2)
            punteggi[idx] = s * prior
        best = int(np.argmax(punteggi))
        if punteggi[best] <= 0:
            return None
        # Affidabilita': il picco deve staccarsi dalla mediana dei candidati. Con rumore o materiale
        # senza pulsazione l'istogramma e' piatto e il "massimo" sarebbe casuale: meglio dire None.
        if punteggi[best] < IOI_MIN_PROMINENZA * (float(np.median(punteggi)) + 1e-12):
            return None
        return (float(bpms[best]), float(punteggi[best] / (np.median(punteggi) + 1e-12))) if _debug else float(bpms[best])
    except Exception:
        return None


def cross_check_tempo(y, sr, tempo_principale):
    """Confronta il BPM principale (beat-tracking dinamico + mediana intervalli, il nostro
    metodo di riferimento) con DUE metodi indipendenti aggiuntivi che usano principi di
    rilevamento diversi tra loro: librosa.feature.tempo (autocorrelazione del tempogram) e
    IOI (istogramma degli intervalli tra onset High-Frequency-Content, nel dominio del tempo).
    - Se entrambi concordano col principale: alta fiducia, nessun avviso necessario.
    - Se ENTRAMBI concordano TRA LORO ma disaccordano dal principale per lo stesso rapporto
      noto (es. 3:2): è un segnale molto più forte — due metodi indipendenti contro uno solo
      — che il principale sia sbagliato, rispetto al vecchio confronto a due soli metodi.
    - Se concordano parzialmente, o nessuno conferma: fiducia intermedia/bassa, segnalata
      esplicitamente invece di far finta che tutto vada bene.
    Ritorna un dict con tutti i dettagli (utile sia per la UI che per il report)."""
    if tempo_principale is None or tempo_principale <= 0:
        return {"esito": "n/d", "feature_tempo": None, "ioi_tempo": None,
                "feature_rapporto": None, "ioi_rapporto": None}

    try:
        y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
        tempo_feature_raw = librosa.feature.tempo(y=y_mono, sr=sr)
        tempo_feature = float(tempo_feature_raw[0]) if hasattr(tempo_feature_raw, '__len__') else float(tempo_feature_raw)
        if tempo_feature <= 0:
            tempo_feature = None
    except Exception:
        tempo_feature = None

    tempo_ioi = ioi_tempo_estimate(y, sr)

    feature_rapporto = classifica_rapporto(tempo_principale, tempo_feature) if tempo_feature else None
    ioi_rapporto = classifica_rapporto(tempo_principale, tempo_ioi) if tempo_ioi else None

    concordi_col_principale = sum(1 for r in [feature_rapporto, ioi_rapporto] if r == "uguale")
    stesso_disaccordo = (
        feature_rapporto and ioi_rapporto
        and feature_rapporto == ioi_rapporto
        and feature_rapporto != "uguale"
    )

    if concordi_col_principale == 2:
        esito = "confermato (2/2 metodi indipendenti concordano)"
    elif stesso_disaccordo:
        esito = f"probabile errore — 2 metodi indipendenti concordano tra loro su un rapporto '{feature_rapporto}' diverso dal principale"
    elif concordi_col_principale == 1:
        esito = "parzialmente confermato (1/2 metodi concorda)"
    else:
        esito = "nessuna conferma indipendente disponibile"

    return {
        "esito": esito,
        "feature_tempo": round(tempo_feature, 1) if tempo_feature else None,
        "ioi_tempo": round(tempo_ioi, 1) if tempo_ioi else None,
        "feature_rapporto": feature_rapporto,
        "ioi_rapporto": ioi_rapporto,
    }


# --- Rilevamento tonalità (Krumhansl-Schmuckler) + notazione Camelot ---------------------
# Profili statistici standard: quanto ogni grado della scala cromatica "suona stabile"
# in un contesto maggiore/minore (Krumhansl & Kessler, 1982). Sono la base di praticamente
# ogni key-detector open source basato su librosa (music-key-finder, pyKeyFinder, ecc.).
_KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Profili tarati su musica popolare/elettronica (Shaath) e ricavati da un corpus di EDM
# (Faraldo et al., "Key Estimation in Electronic Dance Music", ECIR 2016; "A Multi-Profile Method
# for Key Estimation in EDM", AES 2017). I valori sono quelli pubblicati nel sorgente di Essentia
# (MTG/essentia, algoritmo Key, profili 'shaath' e 'edma'). I profili di Krumhansl derivano da
# esperimenti di percezione su musica classica occidentale: su house/techno/pop sbagliano piu'
# spesso, soprattutto sui brani in minore.
_SHAATH_MAJOR_PROFILE = np.array([6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4])
_SHAATH_MINOR_PROFILE = np.array([6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2])
_EDMA_MAJOR_PROFILE = np.array([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39])
_EDMA_MINOR_PROFILE = np.array([1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32])
# Il punteggio finale di ogni tonalità e' la MEDIA delle correlazioni sui tre set di profili:
# un solo set tende a sbagliare in modo sistematico, tre set indipendenti si compensano.
_KEY_PROFILE_SETS = (
    (_KS_MAJOR_PROFILE, _KS_MINOR_PROFILE),
    (_SHAATH_MAJOR_PROFILE, _SHAATH_MINOR_PROFILE),
    (_EDMA_MAJOR_PROFILE, _EDMA_MINOR_PROFILE),
)
KEY_ANALYSIS_SECONDS = 90  # il key detection usa un estratto centrale: la tonalità non cambia e costa meno
_PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Notazione Camelot: numero+lettera per ogni tonalità maggiore (B) e minore (A).
# Le tonalità "compatibili" sono: stesso numero, ±1, o stessa cifra A<->B (relativa magg/min).
_CAMELOT_MAJOR = {'B': '1B', 'F#': '2B', 'Db': '3B', 'Ab': '4B', 'Eb': '5B', 'Bb': '6B',
                  'F': '7B', 'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B'}
_CAMELOT_MINOR = {'Ab': '1A', 'Eb': '2A', 'Bb': '3A', 'F': '4A', 'C': '5A', 'G': '6A',
                  'D': '7A', 'A': '8A', 'E': '9A', 'B': '10A', 'F#': '11A', 'Db': '12A'}
_ENHARMONIC = {'C#': 'Db', 'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb'}  # per pescare dalle tabelle sopra


def _correlazioni_tonalita(chroma_media):
    """Correlazione di Pearson tra il chroma medio e i profili di ogni tonalità, mediata sui
    set di _KEY_PROFILE_SETS. Ritorna un array (12, 2): [semitono_tonica, 0=maggiore/1=minore]."""
    x = chroma_media - np.mean(chroma_media)
    nx = np.linalg.norm(x)
    out = np.zeros((12, 2))
    if nx <= 1e-12:
        return out
    for profili in _KEY_PROFILE_SETS:
        for modo, profilo in enumerate(profili):
            for shift in range(12):
                p = np.roll(profilo, shift)
                p = p - np.mean(p)
                out[shift, modo] += float(np.dot(x, p) / (nx * np.linalg.norm(p) + 1e-12))
    return out / len(_KEY_PROFILE_SETS)


def detect_key_camelot(y, sr):
    """Rileva la tonalità (es. 'A minor') e il codice Camelot corrispondente (es. '8A').
    Pipeline (ispirata ai pre-processing di Faraldo et al. per l'EDM, solo DSP, niente ML):
    1) estratto centrale di KEY_ANALYSIS_SECONDS secondi, 2) componente ARMONICA (HPSS): kick,
    hi-hat e rullanti sporcano il chroma e non portano informazione tonale, 3) stima
    dell'intonazione del brano (tuning) e chroma CQT corretto di conseguenza: un brano
    accordato a 432Hz o con pitch-shift sposta l'energia tra due semitoni e inganna i profili,
    4) correlazione con tre set di profili (Krumhansl, Shaath, EDMA) e media.
    Ritorna (nome_tonalita, codice_camelot, confidenza) — confidenza bassa (< ~0.15 di scarto
    tra 1° e 2° risultato) segnala un brano tonalmente ambiguo."""
    y_mono = np.mean(y, axis=0)
    max_n = int(KEY_ANALYSIS_SECONDS * sr)
    if len(y_mono) > max_n:
        a = (len(y_mono) - max_n) // 2
        y_mono = y_mono[a:a + max_n]
    try:
        y_arm = librosa.effects.harmonic(y_mono)
        # Se la componente armonica e' praticamente vuota (brano solo percussivo) uso l'originale
        if float(np.sum(np.abs(y_arm))) < 1e-3 * float(np.sum(np.abs(y_mono)) + 1e-12):
            y_arm = y_mono
        try:
            tuning = float(librosa.estimate_tuning(y=y_arm, sr=sr))
        except Exception:
            tuning = 0.0
        chroma = librosa.feature.chroma_cqt(y=y_arm, sr=sr, tuning=tuning)
    except Exception:
        return None, None, 0.0
    mean_chroma = np.mean(chroma, axis=1)
    if np.sum(mean_chroma) <= 0:
        return None, None, 0.0

    corr = _correlazioni_tonalita(mean_chroma)
    scores = []
    for shift in range(12):
        scores.append((corr[shift, 0], _PITCH_NAMES[shift], 'major'))
        scores.append((corr[shift, 1], _PITCH_NAMES[shift], 'minor'))

    scores.sort(key=lambda s: (s[0] if not np.isnan(s[0]) else -1), reverse=True)
    best_score, best_pitch, best_mode = scores[0]
    second_score = scores[1][0]
    confidenza = float(best_score - second_score) if not np.isnan(best_score) else 0.0

    nome_tonalita = f"{best_pitch} {'maggiore' if best_mode == 'major' else 'minore'}"
    lookup_pitch = _ENHARMONIC.get(best_pitch, best_pitch)
    tabella = _CAMELOT_MAJOR if best_mode == 'major' else _CAMELOT_MINOR
    camelot = tabella.get(lookup_pitch, '?')
    return nome_tonalita, camelot, confidenza


def camelot_compatibility(camelot_a, camelot_b):
    """Confronta due codici Camelot (es. '8A' e '9A') e dice se sono in relazione armonica
    compatibile secondo le regole standard: stesso numero, ±1, o stesso numero A<->B."""
    if not camelot_a or not camelot_b or camelot_a == '?' or camelot_b == '?':
        return None  # tonalità non rilevabile, non possiamo dire nulla
    try:
        num_a, letter_a = int(camelot_a[:-1]), camelot_a[-1]
        num_b, letter_b = int(camelot_b[:-1]), camelot_b[-1]
    except (ValueError, IndexError):
        return None
    if camelot_a == camelot_b:
        return "identica"
    if letter_a == letter_b and (abs(num_a - num_b) == 1 or abs(num_a - num_b) == 11):
        return "adiacente"
    if num_a == num_b and letter_a != letter_b:
        return "relativa"
    return "dissonante"


_COMPAT_SCORE = {"identica": 3, "adiacente": 3, "relativa": 2, "dissonante": 0, None: 1}
_COMPAT_EMOJI = {"identica": "🟢", "adiacente": "🟢", "relativa": "🟡", "dissonante": "🔴", None: "⚪"}


def extract_artist(file_obj):
    """Legge il campo 'artist' dai tag ID3/metadata del file (via mutagen), se presenti.
    Ritorna '—' se mutagen non è disponibile, il file non ha tag, o qualunque altro problema —
    non deve mai far fallire l'analisi solo per un tag mancante."""
    if not MUTAGEN_DISPONIBILE:
        return "—"
    try:
        file_obj.seek(0)
        audio = MutagenFile(file_obj, easy=True)
        file_obj.seek(0)  # riavvolgo per non interferire con letture successive dello stesso file
        if audio is None:
            return "—"
        artist = audio.get('artist', [None])[0]
        return artist if artist else "—"
    except Exception:
        return "—"


def build_ascii_table(headers, rows):
    """Costruisce una tabella in puro testo (larghezza colonne automatica, separatori |),
    leggibile in qualunque editor senza bisogno di HTML/colori."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def format_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [separator, format_row(headers), separator]
    for row in rows:
        lines.append(format_row(row))
    lines.append(separator)
    return "\n".join(lines)


def build_batch_report(results, order, target_bpm, saltate):
    """Report bilingue IT/EN in stile Loop507: tabella riassuntiva (titolo/artista/BPM/tonalità),
    tabella scaletta ideale per compatibilità Camelot, poi il dettaglio completo per traccia
    (originale vs modificato) — tutto in puro testo, nessun HTML."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Tabella 1: riepilogo tracce ---
    righe_tabella1 = []
    for r in results:
        bpm_label = f"{r['tempo']:.2f}" if r['tempo'] > 0 else "n/d"
        righe_tabella1.append([r['name'], r.get('artist', '—'), bpm_label, r['camelot']])
    tabella1 = build_ascii_table(["Titolo / Title", "Artista / Artist", "BPM", "Camelot"], righe_tabella1)

    # --- Tabella 2: scaletta ideale ---
    righe_tabella2 = []
    for pos, idx in enumerate(order):
        r = results[idx]
        if pos == 0:
            compat_label = "—"
        else:
            prev = results[order[pos - 1]]
            compat = camelot_compatibility(prev['camelot'], r['camelot'])
            compat_label = f"{_COMPAT_EMOJI[compat]} {compat or 'non rilevabile'}"
        righe_tabella2.append([str(pos + 1), r['name'], r['camelot'], compat_label])
    tabella2 = build_ascii_table(["#", "Titolo / Title", "Camelot", "Compatibilità / Compatibility"], righe_tabella2)

    # --- Dettaglio per traccia (originale -> modificato) ---
    righe_it = []
    righe_en = []
    for r in results:
        if r['tempo'] <= 0:
            righe_it.append(f"* {r['name']}: BPM non rilevato — traccia esclusa dall'allineamento")
            righe_en.append(f"* {r['name']}: no BPM detected — track excluded from alignment")
            continue
        rate = target_bpm / r['tempo']
        cross_info_it = ""
        cross_info_en = ""
        cc = r.get('crosscheck', {})
        esito = cc.get('esito', 'n/d')
        if esito not in ('n/d', 'confermato (2/2 metodi indipendenti concordano)'):
            cross_info_it = (f"\n    Verifica incrociata: {esito} "
                              f"(feature.tempo: {cc.get('feature_tempo')} BPM, IOI: {cc.get('ioi_tempo')} BPM)")
            cross_info_en = (f"\n    Cross-check: {esito} "
                              f"(feature.tempo: {cc.get('feature_tempo')} BPM, IOI: {cc.get('ioi_tempo')} BPM)")
        righe_it.append(
            f"* {r['name']}\n"
            f"    Originale: {r['tempo']:.2f} BPM — {r['key_nome']} ({r['camelot']})\n"
            f"    Modificato: portato a {target_bpm:.2f} BPM (rate {rate:.4f}x) — "
            f"tonalità invariata, intonazione preservata (time-stretch phase vocoder)"
            f"{cross_info_it}"
        )
        righe_en.append(
            f"* {r['name']}\n"
            f"    Original: {r['tempo']:.2f} BPM — {r['key_nome']} ({r['camelot']})\n"
            f"    Modified: brought to {target_bpm:.2f} BPM (rate {rate:.4f}x) — "
            f"key unchanged, pitch preserved (phase vocoder time-stretch)"
            f"{cross_info_en}"
        )

    saltate_it = f"\n> ESCLUSE (nessun BPM rilevato): {', '.join(saltate)}" if saltate else ""
    saltate_en = f"\n> EXCLUDED (no BPM detected): {', '.join(saltate)}" if saltate else ""

    return f"""
╔════════════════════════════════════════════════════════════════╗
  LOOP507 :: BATCH BPM MATCHER REPORT
  Generated on: {ts}
╚════════════════════════════════════════════════════════════════╝

> RIEPILOGO TRACCE / TRACKS SUMMARY:

{tabella1}

> SCALETTA IDEALE / SUGGESTED PLAYLIST ORDER (compatibilità armonica Camelot):

{tabella2}
{saltate_it}

═══════════════════ ITALIANO — DETTAGLIO ═══════════════════

:: ENGINE: batch_bpm_matcher_loop507
:: ANALISI: BPM raffinato (mediana intervalli beat) + correzione ottava + verifica incrociata
   (metodo indipendente) + tonalità Camelot (Krumhansl-Schmuckler)
:: BPM TARGET: {target_bpm:.2f}
:: EXPORT: MP3 320kbps

> TRACCE:
{chr(10).join(righe_it)}

═══════════════════ ENGLISH — DETAIL ═══════════════════

:: ENGINE: batch_bpm_matcher_loop507
:: ANALYSIS: refined BPM (beat interval median) + octave correction + cross-check
   (independent method) + Camelot key (Krumhansl-Schmuckler)
:: TARGET BPM: {target_bpm:.2f}
:: EXPORT: MP3 320kbps

> TRACKS:
{chr(10).join(righe_en)}
{saltate_en}

> Regia e Algoritmo / Direction & Algorithm: Loop507
#Loop507 #BatchBPMMatcher #HarmonicMixing #CamelotWheel
"""


def suggest_camelot_order(results):
    """Calcola una scaletta suggerita che massimizza la compatibilità armonica tra brani
    consecutivi: euristica greedy (nearest-neighbor) — parte dal primo brano caricato, poi ad
    ogni passo sceglie tra quelli rimasti il più compatibile col brano appena piazzato.
    Non è garantito l'ordine matematicamente ottimo (richiederebbe un solver completo, eccessivo
    per 5-10 tracce), ma è lo stesso approccio pratico che userebbe un DJ scorrendo la Camelot
    Wheel a mano. Ritorna la lista di indici in 'results' nell'ordine suggerito."""
    n = len(results)
    if n <= 1:
        return list(range(n))
    remaining = set(range(n))
    order = [0]
    remaining.discard(0)
    while remaining:
        last_camelot = results[order[-1]]['camelot']
        best_idx, best_score = None, -1
        for idx in remaining:
            score = _COMPAT_SCORE[camelot_compatibility(last_camelot, results[idx]['camelot'])]
            if score > best_score:
                best_idx, best_score = idx, score
        order.append(best_idx)
        remaining.discard(best_idx)
    return order


# --- Aggancio a frasi musicali (8/16 battute), non solo al singolo beat -------------------
def group_beats_into_phrases(beat_grid_samples, beats_per_phrase=8, phase=0):
    """I DJ non allineano solo i beat, allineano le FRASI (blocchi di 8/16 battute: intro,
    verse, drop...). Raggruppiamo la griglia beat già rilevata in blocchi di N beat e
    restituiamo solo l'inizio di ogni blocco: sono i punti 'strutturalmente giusti' dove
    piazzare un overlay, non un beat qualsiasi in mezzo a una frase."""
    if not beat_grid_samples or beats_per_phrase <= 0:
        return beat_grid_samples
    # `phase` = indice del primo downbeat: le frasi partono da li' e non dal primo beat qualunque
    return beat_grid_samples[max(0, int(phase or 0)) % beats_per_phrase::beats_per_phrase]


# --- EQ Bass Swap: taglia i bassi del frammento overlay per non "impastare" col leader ----
def apply_bass_cut(seg, sr, cutoff_hz=150, order=4):
    """Filtro passa-alto con risposta in ampiezza Linkwitz-Riley (LR4): -6dB esatti alla frequenza
    di taglio, pendenza 24dB/ottava, zero-phase (nessun click da sfasamento).
    Trucco: un Butterworth del 2° ordine applicato con sosfiltfilt (avanti + indietro) ha
    risposta |H|^2, che e' ESATTAMENTE la risposta in ampiezza di un LR4 (= due Butterworth
    2° ordine in cascata). Prima la cascata di due stage passava ANCHE per sosfiltfilt, quindi
    il filtro veniva applicato 4 volte: -12dB al taglio e pendenza 48dB/ottava, cioe' molto piu'
    aggressivo di quanto dichiarato (verificato numericamente: -12.04dB vs -6.02dB).
    `order` e' l'ordine della risposta finale (4 = LR4): lo stage singolo usa order // 2.
    E' la versione 'leggera' dell'EQ bass-swap dei DJ: pulisce i bassi dei frammenti overlay
    cosi' non litigano con la linea di basso del leader, che resta intatta.
    Sotto una lunghezza minima (padlen di sosfiltfilt) o in caso di qualunque errore,
    restituisce il frammento invariato invece di rompere il mix."""
    if not SCIPY_DISPONIBILE or seg.shape[1] < max(30, order * 6):
        return seg
    try:
        sos_stage = butter(order // 2, cutoff_hz, btype='highpass', fs=sr, output='sos')
        filtered = np.stack([sosfiltfilt(sos_stage, seg[ch]) for ch in range(seg.shape[0])], axis=0)
        return filtered
    except Exception:
        return seg  # in caso di segmento troppo corto o altro problema, non rompo il mix


# --- Tecniche leggere ispirate alle console DJ top di gamma (Opus Quad e simili) ----------
def apply_hpss_isolation(seg, sr, mode="percussive"):
    """Separazione armonica/percussiva (HPSS): la versione 'leggera' della Track Separation
    delle console top (niente modelli ML pesanti, solo matematica via librosa — stessa
    tecnica già usata in VIDEODECOMPOSER per isolare melodia/voce). mode='percussive' isola
    batteria/transienti, mode='harmonic' isola basso/melodia. Richiede un frammento con
    abbastanza campioni per una STFT sensata: sotto soglia, o in caso di qualunque problema,
    restituisce il frammento originale invariato invece di rischiare un crash."""
    if seg.shape[1] < 2048:
        return seg
    try:
        out_channels = []
        for ch in range(seg.shape[0]):
            harmonic, percussive = librosa.effects.hpss(seg[ch])
            out_channels.append(percussive if mode == "percussive" else harmonic)
        return np.stack(out_channels, axis=0)
    except Exception:
        return seg


def reverse_segment(seg):
    """'Backspin' semplificato: inverte il frammento nel tempo. Operazione puramente
    numerica (nessuna libreria esterna coinvolta), quindi non può fallire in pratica —
    il try/except resta comunque per uniformità con le altre funzioni FX."""
    try:
        return np.ascontiguousarray(seg[:, ::-1])
    except Exception:
        return seg


def build_loop_roll_segment(seg, repeats, max_total_samples=None):
    """'Loop Roll' in stile console DJ: ripete un microframmento più volte di seguito,
    creando lo stutter tipico di questi controlli. Se il risultato sarebbe più lungo di
    max_total_samples (es. il buffer disponibile), riduce automaticamente le ripetizioni
    invece di generare qualcosa che non potrà mai essere piazzato."""
    if seg.shape[1] <= 0 or repeats <= 1:
        return seg
    try:
        if max_total_samples:
            repeats = max(1, min(repeats, max_total_samples // max(1, seg.shape[1])))
        return np.tile(seg, (1, repeats))
    except Exception:
        return seg


def apply_echo_fx(seg, sr, delay_ms=125, feedback=0.35, num_repeats=3, mix=0.5):
    """Digital delay/echo semplice, stile 'Beat FX' dei mixer da DJ: somma copie ritardate
    e via via più attenuate del frammento a sé stesso (feedback < 1, quindi decade sempre,
    mai un loop che si autoalimenta all'infinito). Il buffer risultante è più lungo
    dell'originale per ospitare le code dell'eco."""
    if seg.shape[1] <= 0 or num_repeats <= 0:
        return seg
    try:
        delay_samples = max(1, int(sr * delay_ms / 1000))
        total_len = seg.shape[1] + delay_samples * num_repeats
        out = np.zeros((seg.shape[0], total_len), dtype=np.float64)
        out[:, :seg.shape[1]] += seg
        gain = feedback
        for i in range(1, num_repeats + 1):
            start = delay_samples * i
            out[:, start:start + seg.shape[1]] += seg * gain * mix
            gain *= feedback
        return out
    except Exception:
        return seg


def carica_audio_stereo(audio_file_object):
    """Decodifica un file audio in array stereo (2, n) a TARGET_SR, senza analisi.
    Il file temporaneo viene sempre ripulito (fix bug #3). Funzione separata da
    analyze_track perche' il Batch BPM Matcher la riusa per RILEGGERE il brano al momento
    dell'allineamento, invece di tenerlo decodificato in RAM per tutta la sessione."""
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
        return y, sr
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def analyze_track_uncached(audio_file_object):
    """Carica in stereo e rileva il tempo. Ritorna (y, sr, tempo, beats, fase_downbeat): `beats` e' la griglia
    dei beat rilevati (lista di campioni interi), conservata nel deck per due scopi: far partire
    i tagli 'a tempo' in fase col primo beat reale, e riusarla come griglia magnetica del
    beatmatching in DJ Remix invece di rifare beat_track ad ogni generazione."""
    y, sr = carica_audio_stereo(audio_file_object)

    # Il beat tracking lavora su un segnale mono: usiamo la media dei due canali solo per l'analisi.
    y_mono_for_beat = np.mean(y, axis=0)
    tempo_raw, beat_samples = librosa.beat.beat_track(y=y_mono_for_beat, sr=sr, units="samples")
    tempo_raw_val = float(tempo_raw[0]) if isinstance(tempo_raw, np.ndarray) else float(tempo_raw)
    # Raffino il BPM globale con la mediana degli intervalli reali tra i beat rilevati:
    # più preciso della singola stima algoritmica di beat_track (fix precisione BPM).
    tempo_val = refine_tempo_from_beats(beat_samples, sr, tempo_raw_val)
    beats = [int(b) for b in np.sort(np.asarray(beat_samples))] if len(beat_samples) else []
    fase_db, conf_db = estimate_downbeat_phase(y_mono_for_beat, sr, beats)
    # Sotto soglia la stima non e' affidabile: meglio nessun downbeat che uno sbagliato.
    if conf_db < DOWNBEAT_MIN_CONFIDENCE:
        fase_db = None
    return y, sr, tempo_val, beats, fase_db


# max_entries: la cache tiene una copia (pickle) di ogni brano decodificato; senza limite
# cresceva all'infinito a ogni file diverso caricato. 6 copre i deck realmente in uso.
@st.cache_data(max_entries=6)
def analyze_track(audio_file_object):
    """Versione cached di analyze_track_uncached, usata dai deck (il 'Ricarica in RAM' di un
    deck liberato e' quasi istantaneo se il file e' ancora in cache)."""
    return analyze_track_uncached(audio_file_object)


def fase_primo_beat(first_beat, tempo, sr, rate=1.0, beats_per_unit=1):
    """Offset (in campioni) da cui far partire una griglia di tagli a tempo, cosi' che i tagli
    cadano SUI beat reali del brano e non a metà battuta. Prima i tagli partivano sempre dal
    campione 0, ignorando dove cade davvero il primo beat. Si prende il primo beat modulo la
    lunghezza di un beat: si scarta al massimo il materiale prima del primo punto di griglia (meno di un beat; fino a una misura coi downbeat).
    `rate` e' il time-stretch eventualmente applicato (le posizioni scalano di 1/rate).
    `beats_per_unit`: 1 = fase sul beat; 4 = fase sulla BATTUTA (si passa il primo downbeat,
    cosi' i tagli da 4/8 battute partono davvero dall'inizio di una misura)."""
    if first_beat is None or tempo is None or tempo <= 1e-3 or not rate or rate <= 0:
        return 0
    samples_per_beat = sr * 60.0 / tempo * max(1, int(beats_per_unit))
    if samples_per_beat <= 0:
        return 0
    return int((first_beat / rate) % samples_per_beat)


def primo_downbeat_deck(d):
    """Primo downbeat stimato di un deck (campioni, gia' corretto per l'eventuale trim), o None
    se la stima non c'e' o non era abbastanza affidabile."""
    fase = d.get('downbeat_phase')
    beats = d.get('beats')
    if fase is None or not beats or fase >= len(beats):
        return None
    return beats[fase]


def fase_deck(d, tempo, sr, rate=1.0, usa_downbeat=False):
    """Offset di partenza dei tagli a tempo per un deck: sulla battuta (downbeat) se richiesto
    e disponibile, altrimenti sul primo beat rilevato."""
    if usa_downbeat:
        fb = primo_downbeat_deck(d)
        if fb is not None:
            return fase_primo_beat(fb, tempo, sr, rate, beats_per_unit=4)
    return fase_primo_beat(primo_beat_deck(d), tempo, sr, rate)


def estimate_downbeat_phase(y_mono, sr, beat_samples, beats_per_bar=4, fmax_hz=2000.0):
    """Stima quale beat (modulo beats_per_bar) e' l'inizio di una misura, con la differenza
    spettrale tra frame sincroni ai beat (idea di Davies & Plumbley, EUSIPCO 2006: i downbeat
    sono i beat dove il cambiamento spettrale tra il prima e il dopo e' massimo — rientra il
    basso, cambia l'accordo, parte la frase). Qui: spettro band-limited (< fmax_hz) in scala
    log, media su ogni intervallo di beat, aumento positivo di energia dal beat precedente al
    successivo, poi si sceglie la fase (0..beats_per_bar-1) con la media piu' alta.
    Assume tempo in 4/4 (la quasi totalita' della musica da club). E' una STIMA euristica,
    senza reti neurali: ritorna (fase, confidenza) con confidenza = quanto la fase migliore
    supera la media delle altre (0..1). Ritorna (None, 0.0) se i beat sono troppo pochi."""
    beats = np.asarray(beat_samples, dtype=np.int64)
    if len(beats) < beats_per_bar * 4 or len(y_mono) < 4096:
        return None, 0.0
    try:
        n_fft, hop = 2048, 512
        n_bins = max(8, int(fmax_hz * n_fft / sr) + 1)
        chunk_frames = 2048
        cols = []
        # STFT a blocchi, tenendo solo le colonne band-limited: su un brano intero la STFT
        # completa pesa centinaia di MB, cosi' invece resta in pochi MB.
        for f0 in range(0, 1 + max(0, (len(y_mono) - n_fft) // hop), chunk_frames):
            a = f0 * hop
            seg = y_mono[a:a + (chunk_frames - 1) * hop + n_fft]
            if len(seg) < n_fft:
                break
            S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop, center=False))[:n_bins]
            cols.append(np.log1p(10.0 * S))
        if not cols:
            return None, 0.0
        L = np.concatenate(cols, axis=1)                       # (bins, frames)
        csum = np.concatenate([np.zeros((L.shape[0], 1)), np.cumsum(L, axis=1)], axis=1)
        F = L.shape[1]
        # indice di frame il cui centro cade sul beat
        fb = np.clip(np.round((beats - n_fft // 2) / hop).astype(int), 0, F)

        def media(a, b):
            return (csum[:, b] - csum[:, a]) / max(1, b - a)

        diff = np.full(len(beats), np.nan)
        for k in range(1, len(beats) - 1):
            a0, a1, a2 = fb[k - 1], fb[k], fb[k + 1]
            if a1 - a0 < 1 or a2 - a1 < 1:
                continue
            d = media(a1, a2) - media(a0, a1)                  # dopo - prima
            diff[k] = float(np.sum(np.maximum(d, 0.0)))        # solo l'AUMENTO di energia
        scores = []
        for p in range(beats_per_bar):
            v = diff[p::beats_per_bar]
            v = v[~np.isnan(v)]
            scores.append(float(np.mean(v)) if len(v) else 0.0)
        scores = np.asarray(scores)
        best = int(np.argmax(scores))
        altri = np.delete(scores, best)
        if scores[best] <= 1e-9:
            return None, 0.0
        confidenza = float((scores[best] - np.mean(altri)) / scores[best])
        return best, max(0.0, min(1.0, confidenza))
    except Exception:
        return None, 0.0


def primo_beat_deck(d):
    """Primo beat rilevato di un deck (campioni, gia' corretto per l'eventuale trim), o None."""
    beats = d.get('beats')
    return beats[0] if beats else None


def get_beat_segments(y, sr, tempo, num_beats, start_offset=0):
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
    start_offset = max(0, int(start_offset or 0))  # fase della griglia (vedi fase_primo_beat)
    # .copy() e' essenziale: senza, ogni segmento sarebbe solo una "vista" sulla memoria
    # dell'array originale, quindi anche liberando d['y'] la RAM non verrebbe mai rilasciata.
    return [
        y[:, i:i + segment_length].copy()
        for i in range(start_offset, total_samples, segment_length)
        if (i + segment_length) <= total_samples
    ]


def get_random_segments(y, sr, min_dur, max_dur, rng=None):
    """Segmentazione casuale, con guardia contro durate/lunghezze a zero (fix bug #2).
    y ha shape (2, n): il taglio avviene lungo l'asse temporale (asse 1)."""
    rng = rng if rng is not None else random  # RNG seedato iniettato: taglio riproducibile
    segments = []
    curr = 0
    total_samples = y.shape[1]
    min_dur = max(min_dur, MIN_SEGMENT_SAMPLES / sr)  # mai sotto la soglia minima
    while curr < total_samples:
        dur = rng.uniform(min_dur, max_dur)
        length = int(dur * sr)
        if length <= 0:
            break
        if curr + length > total_samples:
            break
        segments.append(y[:, curr:curr + length].copy())
        curr += length
    return segments


def get_beat_segments_range(y, sr, tempo, min_beats, max_beats, start_offset=0, rng=None):
    """Come get_beat_segments, ma invece di un numero fisso di battute per taglio, ogni
    segmento usa una lunghezza casuale (in battute) pescata nel range [min_beats, max_beats].
    Utile per non avere sempre tagli tutti identici anche in modalità 'a ritmo'."""
    if tempo is None or tempo <= 1e-3:
        return []
    samples_per_beat = sr * 60 / tempo
    max_samples = int(MAX_SEGMENT_SECONDS * sr)
    total_samples = y.shape[1]
    lo, hi = (min_beats, max_beats) if min_beats <= max_beats else (max_beats, min_beats)

    rng = rng if rng is not None else random  # RNG seedato iniettato: taglio riproducibile
    segments = []
    curr = max(0, int(start_offset or 0))  # fase della griglia (vedi fase_primo_beat)
    while curr < total_samples:
        beats = rng.uniform(lo, hi)
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
                            beats_range=None, onset_min_dur=None, onset_max_dur=None,
                            start_offset=0, rng=None):
    """Ricava dal deck 'leader' solo le LUNGHEZZE dei tagli (non l'audio), da imporre
    poi a tutti gli altri deck: e' la 'struttura' che i follower devono rispettare."""
    if leader_mode == "onset":
        segs = get_onset_segments(y_leader, sr, onset_min_dur, onset_max_dur)
    else:  # "bpm"
        if beats_range is not None:
            segs = get_beat_segments_range(y_leader, sr, tempo, beats_range[0], beats_range[1],
                                           start_offset=start_offset, rng=rng)
        else:
            segs = get_beat_segments(y_leader, sr, tempo, num_beats, start_offset=start_offset)
    return [s.shape[1] for s in segs]


def apply_cut_lengths(y, lengths, start_offset=0):
    """Applica in sequenza una lista di lunghezze (derivata dal leader) a un deck qualsiasi,
    scorrendo il suo audio dall'inizio. Se il deck 'follower' finisce prima di esaurire
    tutte le lunghezze richieste, si ferma semplicemente li' — nessun loop forzato."""
    segments = []
    curr = max(0, int(start_offset or 0))  # ogni deck parte dal proprio primo beat (fase), non da 0
    total_samples = y.shape[1]
    for length in lengths:
        if length <= 0:
            continue
        if curr + length > total_samples:
            break
        segments.append(y[:, curr:curr + length].copy())
        curr += length
    return segments


def export_audio(y, sr, ceiling_db=-1.0, boost=True):
    """Esporta un array stereo (2, n) in MP3 stereo a 320kbps, con interleaving corretto
    dei canali. NOTA: prima di questo fix, il bitrate non era mai specificato esplicitamente,
    quindi pydub/ffmpeg usava un default molto più basso (~130kbps verificato), nonostante
    tutti i report dell'app dichiarassero 320kbps — un bug reale, non solo percezione."""
    if y.shape[1] == 0:
        return None
    # Headroom: prima il picco veniva SEMPRE portato a 0 dBFS, annullando il lavoro del
    # compressore e rischiando clipping inter-sample nella codifica MP3. Ora il tetto e' a
    # `ceiling_db` (default -1 dBFS).
    # boost=True  -> il picco va esattamente al tetto (mix finale: massimo livello utile).
    # boost=False -> si ABBASSA solo se si supera il tetto, senza mai amplificare: serve al
    #               Batch BPM Matcher, dove la normalizzazione per-traccia annullerebbe
    #               il bilanciamento LUFS tra le tracce dello ZIP.
    ceiling = 10 ** (ceiling_db / 20.0)
    max_val = float(np.max(np.abs(y)))
    if max_val > 0 and (boost or max_val > ceiling):
        y = y * (ceiling / max_val)
    y_norm = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)

    # y_norm ha shape (2, n): trasponiamo a (n, 2) e appiattiamo per ottenere
    # il classico interleaving L,R,L,R,... richiesto da un file audio stereo.
    interleaved = y_norm.T.flatten()

    buffer = BytesIO()
    audio_seg = AudioSegment(interleaved.tobytes(), frame_rate=sr, sample_width=2, channels=2)
    audio_seg.export(buffer, format="mp3", bitrate="320k")
    buffer.seek(0)
    return buffer


def time_stretch_stereo(y, rate):
    """Cambia davvero la velocità dell'audio (non solo la lunghezza dei tagli), preservando
    l'intonazione (phase vocoder via librosa.effects.time_stretch). rate > 1 = più veloce,
    rate < 1 = più lento. Applicato canale per canale per non mescolare L/R.
    Su frammenti molto corti (es. quelli usati per l'allineamento BPM in DJ Remix, che
    possono scendere sotto i 300 campioni) l'n_fft di default (2048) è più largo del
    segmento stesso: librosa non si rompe, ma la risoluzione della STFT degrada visibilmente
    (fix qualità: n_fft adattivo, mai più grande del segmento)."""
    if rate is None or rate <= 0 or abs(rate - 1.0) < 1e-3:
        return y  # nulla da fare, o rate non valido: restituisco l'audio invariato
    seg_len = y.shape[1]
    n_fft = 2048
    if seg_len < n_fft:
        n_fft = max(32, 2 ** int(np.floor(np.log2(max(seg_len, 1)))))
    hop_length = max(1, n_fft // 4)
    stretched_channels = []
    for ch in range(y.shape[0]):
        stretched_channels.append(
            librosa.effects.time_stretch(y[ch], rate=rate, n_fft=n_fft, hop_length=hop_length)
        )
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
                            rng, deck_weights=None, beatmatch=True, phrase_beats=None,
                            bass_cutoff_hz=None, hpss_mode=None, reverse_probability=0.0,
                            loop_roll_probability=0.0, loop_roll_repeats=4, echo_probability=0.0,
                            precomputed_beat_grid=None, downbeat_phase=None,
                            bass_sweep=False, gate_probability=0.0, tape_stop_probability=0.0,
                            bitcrush_probability=0.0, fx_tempo=None, section_idx=None,
                            placements_out=None):
    """Costruisce un mix 'letto + overlay': leader_y resta INTATTO come base (nessun taglio),
    e sopra vengono sparsi num_events frammenti presi da overlay_segments, sommati (non
    concatenati), moltiplicati per overlay_gain.
    Se beatmatch e' attivo: ogni frammento non va a un punto casuale qualsiasi, ma il suo
    attacco più forte viene agganciato esattamente a un beat della griglia del leader —
    il vero 'aggancio ai picchi' stile DJ, non solo lo stesso tempo.
    Se phrase_beats e' impostato (richiede beatmatch): invece di agganciare a QUALSIASI beat,
    aggancia solo all'inizio di una frase (blocco di phrase_beats battute) — allineamento
    strutturale, non solo ritmico.
    Se bass_cutoff_hz e' impostato: ogni frammento overlay viene filtrato passa-alto prima di
    essere sommato, per non 'impastare' con la linea di basso del leader (EQ bass swap).
    hpss_mode ('percussive'/'harmonic'/None): isola solo la componente percussiva o armonica
    del frammento prima di sommarlo (stem separation 'leggera', precalcolata una sola volta
    per frammento — vedi hpss_cache sotto — per non rifare un'operazione costosa ad ogni
    piazzamento se lo stesso frammento viene ripescato più volte).
    reverse_probability / loop_roll_probability / echo_probability: probabilità indipendenti
    (0-1) che, ad ogni piazzamento, il frammento venga invertito (backspin), trasformato in
    stutter (loop roll), o passato attraverso un eco (Beat FX). Ognuno di questi FX può
    cambiare la lunghezza del frammento: per questo, DOPO averli applicati, la lunghezza
    viene sempre ricontrollata contro lo spazio disponibile nel buffer — se non ci sta più,
    l'evento viene semplicemente scartato (si ripesca al tentativo successivo) invece di
    troncare bruscamente o sforare i limiti del buffer.
    Ritorna anche la lista degli eventi piazzati (per l'export MIDI della struttura)."""
    buffer = leader_y.astype(np.float64).copy()
    total = buffer.shape[1]
    if not overlay_segments or total == 0:
        return buffer, 0, []

    beat_grid = []
    fase_frase = 0  # indice del primo downbeat nella griglia (0 = non noto)
    grid_pre = False  # True se beat_grid e' la griglia precalcolata (gli indici delle sezioni valgono solo li')
    if beatmatch:
        if precomputed_beat_grid:
            if downbeat_phase is not None:
                fase_frase = int(downbeat_phase)  # la griglia filtrata conserva gli indici (si tagliano solo i beat finali)
            # Griglia gia' rilevata in analisi (e corretta per il trim): evita di rifare beat_track
            # sull'intero leader ad ogni generazione. Si scartano i beat oltre il buffer (anteprima).
            beat_grid = [b for b in precomputed_beat_grid if 0 <= b < total]
            grid_pre = bool(beat_grid)
        if not beat_grid:
            beat_grid = get_beat_grid_samples(leader_y, leader_sr)
            grid_pre = False
            fase_frase = 0  # griglia ricalcolata: la fase del downbeat non e' piu' riferita a questa
    if beatmatch and not beat_grid:
        beatmatch = False  # il leader non ha un beat rilevabile: fallback a piazzamento casuale
    if beatmatch and section_idx and grid_pre:
        # Aggancio ai confini di sezione (e alle prime battute dopo ciascuno): prevale sulle frasi
        sez_grid = espandi_confini_in_griglia(beat_grid, [i for i in section_idx if i < len(beat_grid)])
        if sez_grid:
            beat_grid = sez_grid
            phrase_beats = None
    if beatmatch and phrase_beats:
        frase_grid = group_beats_into_phrases(beat_grid, phrase_beats, phase=fase_frase)
        if frase_grid:
            beat_grid = frase_grid  # aggancio solo agli inizi di frase, non a ogni beat

    weights_list = None
    if deck_weights:
        counts = {}
        for s in overlay_segments:
            counts[s['deck']] = counts.get(s['deck'], 0) + 1
        weights_list = [deck_weights.get(s['deck'], 5) / max(1, counts.get(s['deck'], 1)) for s in overlay_segments]

    prep_cache = {}  # id(pick) -> (audio dopo rampa+HPSS+bass, flag alterato, onset), una volta per frammento
    placed = 0
    attempts = 0
    max_attempts = max(500, num_events * 20)  # guardia anti-loop-infinito, stessa logica del bug #2
    events = []  # per il report MIDI: (start_sample, deck)
    while placed < num_events and attempts < max_attempts:
        attempts += 1
        pick = rng.choices(overlay_segments, weights=weights_list, k=1)[0] if weights_list else rng.choice(overlay_segments)
        seg = pick['audio']
        if seg.shape[1] <= 0:
            continue

        # Catena deterministica per frammento, calcolata una sola volta (cache):
        # rampa di tempo -> isolamento HPSS -> bass swap (statico o graduale).
        catena_alterata = False
        onset_cache_val = None
        if hpss_mode or bass_cutoff_hz or pick.get('ramp_rate'):
            cache_key = id(pick)
            if cache_key not in prep_cache:
                prepared = seg
                if pick.get('ramp_rate'):
                    prepared = stretch_ramp(prepared, pick['ramp_rate'], leader_sr)
                if hpss_mode:
                    prepared = apply_hpss_isolation(prepared, leader_sr, mode=hpss_mode)
                if bass_cutoff_hz:
                    # EQ Bass Swap: tolgo i bassi dall'overlay PRIMA di sommarlo, non dopo — cosi'
                    # il leader sotto resta con la sua linea di basso intatta e pulita.
                    prepared = (apply_bass_sweep(prepared, leader_sr, cutoff_hz=bass_cutoff_hz) if bass_sweep
                                else apply_bass_cut(prepared, leader_sr, cutoff_hz=bass_cutoff_hz))
                altered = bool(pick.get('ramp_rate'))  # la rampa cambia lunghezza e posizione dell'attacco
                prep_cache[cache_key] = (
                    prepared, altered,
                    find_strongest_onset_offset(prepared, leader_sr) if (altered and beatmatch) else None)
            seg, catena_alterata, onset_cache_val = prep_cache[cache_key]

        # FX stocastici indipendenti: ognuno può cambiare la lunghezza del frammento,
        # quindi invalidano l'onset_offset precalcolato (lo ricalcoliamo sotto se serve).
        lunghezza_alterata = False
        if reverse_probability and rng.random() < reverse_probability:
            seg = reverse_segment(seg)
            lunghezza_alterata = True  # la posizione dell'attacco cambia anche se la lunghezza no
        if loop_roll_probability and rng.random() < loop_roll_probability:
            roll_piece_len = max(1, seg.shape[1] // 4)
            seg = build_loop_roll_segment(seg[:, :roll_piece_len], loop_roll_repeats, max_total_samples=total)
            lunghezza_alterata = True
        if echo_probability and rng.random() < echo_probability:
            seg = apply_echo_fx(seg, leader_sr)
            lunghezza_alterata = True
        # Effetti ritmici extra (tutti dopo i precedenti, cosi' a probabilità 0 non consumano
        # numeri casuali e i mix gia' riproducibili con la _8 restano identici).
        if gate_probability and rng.random() < gate_probability:
            seg = apply_gate_fx(seg, leader_sr, tempo=fx_tempo or 120.0)
            lunghezza_alterata = True  # il gate azzera parti: l'attacco piu' forte puo' spostarsi
        if tape_stop_probability and rng.random() < tape_stop_probability:
            seg = apply_tape_stop_fx(seg)  # tocca solo la coda: l'attacco non cambia
        if bitcrush_probability and rng.random() < bitcrush_probability:
            seg = apply_bitcrush_fx(seg)

        seg_len = seg.shape[1]
        if seg_len <= 0 or seg_len > total:
            continue  # un fx ha reso il frammento troppo lungo/corto: salto, non rompo nulla

        if beatmatch:
            beat_pos = rng.choice(beat_grid)
            if lunghezza_alterata:
                # il precalcolato non è più valido (segmento invertito/allungato/ripetuto):
                # ricalcolo l'attacco più forte sul frammento cosi' com'è ora.
                onset_offset = find_strongest_onset_offset(seg, leader_sr)
            else:
                # onset_offset e' pre-calcolato su ogni segmento (vedi sotto) per non rifare
                # la detection ad ogni singolo piazzamento, anche se lo stesso frammento
                # viene ripescato più volte.
                onset_offset = (onset_cache_val if (catena_alterata and onset_cache_val is not None)
                                else pick.get('onset_offset', 0))
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
        if placements_out is not None:
            placements_out.append({'start': int(start), 'length': int(seg_len), 'deck': pick['deck'],
                                   'audio': seg, 'gain': overlay_gain})
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


# --- Preparazione audio: loudness, trim, provino veloce, immagine finale ------------------
def apply_compressor(y, sr, threshold=-15.0, ratio=4.0):
    """Comprime dinamicamente il mix finale (leader+follower insieme): a differenza della
    normalizzazione LUFS (che pareggia il volume MEDIO di un intero deck prima di iniziare),
    il compressore lavora momento per momento — se un singolo overlay ha un picco locale più
    forte del resto, lo attenua *in quel preciso istante*, anche se il LUFS medio del deck era
    già a posto. Usa audiocomplib (stereo-linked, soft-knee, niente spostamento dell'immagine
    stereo). Se il pacchetto non è installato o l'array è troppo corto/degenere, restituisce
    l'audio invariato invece di rischiare un crash."""
    if not COMPRESSOR_DISPONIBILE or y.shape[1] == 0:
        return y
    try:
        compressor = AudioCompressor(
            threshold=threshold, ratio=ratio, attack_time_ms=5.0, release_time_ms=80.0,
            knee_width=3.0, makeup_gain=0.0
        )
        y_input = np.ascontiguousarray(y.astype(np.float32))
        compressed = compressor.process(y_input, sr)
        if compressed.shape != y.shape or not np.all(np.isfinite(compressed)):
            return y  # risultato inatteso: meglio l'originale che un mix rotto
        return compressed.astype(np.float64)
    except Exception:
        return y


def normalize_loudness(y, sr, target_lufs=-18.0):
    """Normalizza un deck stereo (2, n) a un target LUFS comune (standard ITU-R BS.1770-4,
    via pyloudnorm), cosi' deck mixati a volumi molto diversi non falsano più il bilanciamento
    reale in DJ Remix/Hyper-Mixer. Se la misura non è affidabile (audio troppo corto, o
    silenzio/quasi-silenzio dove il guadagno esploderebbe) restituisce l'audio invariato
    invece di rischiare un risultato assurdo."""
    if not LOUDNORM_DISPONIBILE or y.shape[1] == 0:
        return y
    try:
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(y.T)  # pyloudnorm vuole (n_campioni, canali)
        if not np.isfinite(current_lufs) or current_lufs < -60:
            return y  # silenzio o misura inaffidabile: non normalizzo
        normalized = pyln.normalize.loudness(y.T, current_lufs, target_lufs).T
        # pyloudnorm può amplificare oltre 0dBFS in casi estremi (deck originariamente molto
        # silenzioso): clip di sicurezza per non introdurre distorsione digitale.
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val
        return normalized
    except Exception:
        return y


def trim_silence_stereo(y, top_db=40, return_offset=False):
    """Rimuove il silenzio (sotto top_db) da inizio e fine di un deck stereo (2, n), usando
    l'energia combinata dei due canali per decidere i punti di taglio, poi applicando lo
    STESSO taglio a entrambi (cosi' L/R restano sincronizzati). Utile soprattutto per il
    deck leader in DJ Remix: senza trim, silenzio in testa/coda è 'spazio morto' dove un
    overlay può finire piazzato inutilmente.
    return_offset=True ritorna (audio, campioni_tagliati_in_testa): serve a spostare di pari
    passo la griglia dei beat rilevata PRIMA del trim."""
    def _ret(y_out, offset):
        return (y_out, offset) if return_offset else y_out

    if y.shape[1] == 0:
        return _ret(y, 0)
    try:
        y_mono = np.mean(np.abs(y), axis=0)
        _, index = librosa.effects.trim(y_mono, top_db=top_db)
        if index[1] <= index[0]:
            return _ret(y, 0)  # trim risultato vuoto/invertito: non rischio di azzerare il deck
        return _ret(y[:, index[0]:index[1]], int(index[0]))
    except Exception:
        return _ret(y, 0)


def invalidate_mix():
    """Invalida il mix precedente quando i segmenti cambiano (fix bug #4)."""
    st.session_state.pop("mix_ready", None)
    st.session_state.pop("mix_preset", None)
    st.session_state.pop("mix_events", None)
    st.session_state.pop("mix_events_sr", None)
    st.session_state.pop("mix_events_tempo", None)
    for _k in ("mix_placements", "mix_leader", "mix_total_samples", "mix_dry_peak", "mix_deck_names",
               "mix_key_nome", "mix_sections_sec", "mix_downbeat_sec", "mix_wav24", "mix_stems_zip",
               "mix_midi_trascritto", "mix_midi_conteggi", "mix_tempo_noto"):
        st.session_state.pop(_k, None)
    st.session_state.pop("dj_rendered_by_deck", None)
    st.session_state.pop("mix_is_preview", None)
    for _k in ("mix_waveform_video", "mix_waveform_video_info", "mix_bar_times", "mix_img_title"):
        st.session_state.pop(_k, None)
    st.session_state.pop("mix_waveform_image", None)
    st.session_state.audio_report = ""


def crea_rng_taglio(seed_input):
    """RNG locale e seedato per la fase di TAGLIO (modalità con casualità: Frenesia Casuale,
    range di battute). Prima il taglio usava il `random` globale non seedato, quindi il
    seed del preset garantiva solo l'assemblaggio del mix, non i tagli: il mix NON era davvero
    riproducibile. Il seed usato viene salvato in session_state per finire nel preset."""
    seed = int(seed_input) if seed_input and seed_input > 0 else random.randint(1, 2**31 - 1)
    st.session_state.cut_seed_used = seed
    return random.Random(f"cut:{seed}")


def applica_preparazione(y, sr, trim=False, trim_db=40, normalizza=False, target_lufs=-18.0):
    """Trim del silenzio + normalizzazione LUFS (stesso ordine usato per i deck: prima il trim,
    cosi' il silenzio tagliato non falsa la misura di loudness)."""
    if trim:
        y = trim_silence_stereo(y, top_db=trim_db)
    if normalizza:
        y = normalize_loudness(y, sr, target_lufs=target_lufs)
    return y


def prepara_deck(up_file, correggi_ottava_on, bpm_range, trim, trim_db, normalizza, target_lufs):
    """Analisi completa di un deck: carica, rileva BPM (+correzione ottava), trim, LUFS, e
    sposta la griglia dei beat di pari passo col trim. Ritorna (y, sr, tempo, beats, fase_downbeat)."""
    y, sr, t, beats, fase_db = analyze_track(up_file)
    if correggi_ottava_on and t > 0:
        t = correggi_errore_ottava(t, bpm_range[0], bpm_range[1])
    trim_off = 0
    if trim:
        # Dopo l'analisi BPM: tagliare il silenzio non altera il tempo del contenuto reale.
        y, trim_off = trim_silence_stereo(y, top_db=trim_db, return_offset=True)
    if normalizza:
        # Dopo il trim: il silenzio tagliato non influenza la misura di loudness.
        y = normalize_loudness(y, sr, target_lufs=target_lufs)
    n = y.shape[1]
    beats_adj = [b - trim_off for b in beats if 0 <= b - trim_off < n]
    if fase_db is not None:
        # I beat tagliati in testa dal trim fanno scalare l'indice: la fase si sposta di pari passo.
        persi = sum(1 for b in beats if b - trim_off < 0)
        fase_db = (fase_db - persi) % 4
        if fase_db >= len(beats_adj):
            fase_db = None
    return y, sr, t, beats_adj, fase_db


def stretch_segments_in_blocks(audios, rate, sr, max_block_seconds=30):
    """Time-stretch di una lista di frammenti dello STESSO deck, a blocchi: i frammenti
    consecutivi vengono concatenati (fino a ~max_block_seconds), stretchati UNA volta con la
    finestra STFT piena, poi ri-divisi in proporzione. Prima ogni frammento veniva stretchato
    da solo: su pezzi di poche centinaia di campioni la STFT degrada (n_fft ridotto), e
    l'operazione veniva rifatta per ogni singolo frammento ad ogni generazione. A blocchi la
    qualità è quella del deck intero, ma la RAM resta limitata (niente STFT sull'intero brano)."""
    out = [None] * len(audios)
    max_block = int(max_block_seconds * sr)
    block, block_idx, block_len = [], [], 0

    def _flush():
        if not block:
            return
        lengths = [a.shape[1] for a in block]
        big = np.concatenate(block, axis=1)
        stretched = time_stretch_stereo(big, rate)
        total_in = sum(lengths)
        scale = stretched.shape[1] / total_in if total_in > 0 else 1.0
        bounds = np.round(np.cumsum([0] + lengths) * scale).astype(int)
        for n_i, idx in enumerate(block_idx):
            out[idx] = stretched[:, bounds[n_i]:bounds[n_i + 1]].copy()

    for i, a in enumerate(audios):
        if a.shape[1] <= 0:
            out[i] = a
            continue
        if block and block_len + a.shape[1] > max_block:
            _flush()
            block, block_idx, block_len = [], [], 0
        block.append(a)
        block_idx.append(i)
        block_len += a.shape[1]
    _flush()
    return out


# =====================================================================================
# v5.8: transizioni/effetti DJ, tonalità attiva, sezioni, export per Logic (WAV24, stem, MIDI)
# =====================================================================================
def apply_bass_sweep(seg, sr, cutoff_hz=150):
    """Bass swap GRADUALE: il frammento entra con i bassi tagliati (LR4) e il filtro si 'apre'
    lungo la sua durata fino al suono pieno, come un DJ che porta su il low-EQ del deck entrante.
    Si fa un morph tra la copia filtrata e quella piena con una rampa smoothstep: nessun filtro
    a coefficienti variabili (niente click né instabilità) e costa un solo filtraggio."""
    filtered = apply_bass_cut(seg, sr, cutoff_hz=cutoff_hz)
    n = seg.shape[1]
    if filtered is seg or n < 8:
        return seg
    x = np.linspace(0.0, 1.0, n)
    ramp = x * x * (3.0 - 2.0 * x)
    return filtered * (1.0 - ramp) + seg * ramp


def stretch_ramp(seg, rate_end, sr, rate_start=1.0):
    """Rampa di tempo dentro un frammento: la velocità passa gradualmente da rate_start a
    rate_end (invece di un rate fisso su tutto). Si divide il frammento in blocchi (>= 0.25s),
    si stretcha ogni blocco col suo rate interpolato e si ricompone con un crossfade di 10ms.
    Sotto 0.5s una rampa non ha senso: si applica il rate medio."""
    n = seg.shape[1]
    min_block = int(0.25 * sr)
    if abs(rate_end - rate_start) < 1e-3:
        return seg
    if n < 2 * min_block:
        return time_stretch_stereo(seg, (rate_start + rate_end) / 2.0)
    nb = int(max(2, min(8, n // min_block)))
    edges = np.linspace(0, n, nb + 1).astype(int)
    rates = np.linspace(rate_start, rate_end, nb)
    pezzi = [time_stretch_stereo(seg[:, edges[i]:edges[i + 1]], float(rates[i])) for i in range(nb)]
    return concat_with_crossfade(pezzi, int(0.01 * sr))


def apply_gate_fx(seg, sr, tempo=120.0, division=8, duty=0.5):
    """Gate ritmico: il frammento si apre e si chiude a tempo (1/division di battuta), con
    rampe di 3ms per non fare click. Effetto 'trance gate' / sidechain secco."""
    n = seg.shape[1]
    if n < 64:
        return seg
    bpm = tempo if tempo and tempo > 20 else 120.0
    step = max(32, int(sr * 60.0 / bpm * 4.0 / division))
    mask = ((np.arange(n) % step) < int(step * duty)).astype(np.float64)
    ramp = max(1, int(0.003 * sr))
    mask = np.convolve(mask, np.ones(ramp) / ramp, mode='same')
    return seg * mask[None, :]


def apply_tape_stop_fx(seg, stop_fraction=0.6):
    """Tape stop: nella parte finale la velocità di lettura scende a quasi zero (il suono
    si 'siede' abbassando intonazione e velocità), con una dissolvenza per non lasciare DC."""
    n = seg.shape[1]
    start = int(n * (1.0 - stop_fraction))
    m = n - start
    if m < 64:
        return seg
    speed = np.linspace(1.0, 0.03, m)
    pos = np.concatenate([[0.0], np.cumsum(speed)[:-1]])  # posizione di lettura (campioni), sempre < m
    coda = seg[:, start:]
    idx = np.arange(m)
    fermo = np.stack([np.interp(pos, idx, coda[c]) for c in range(seg.shape[0])])
    fermo *= (np.linspace(1.0, 0.0, m) ** 0.5)[None, :]
    out = seg.copy()
    out[:, start:] = fermo
    return out


def apply_bitcrush_fx(seg, bits=6, downsample=4):
    """Bitcrusher: riduce la risoluzione (bit) e la frequenza di campionamento (sample-hold)."""
    if seg.shape[1] < 8:
        return seg
    q = float(2 ** (bits - 1))
    out = np.round(seg * q) / q
    idx = (np.arange(seg.shape[1]) // max(1, int(downsample))) * max(1, int(downsample))
    return out[:, idx]


# --- tonalità attiva: compatibilità Camelot usata davvero, non solo mostrata ---------------
_TONALITA_OK = ("identica", "adiacente", "relativa")


def parse_tonalita(nome):
    """'A minore' -> (9, 'minor'); 'C# maggiore' -> (1, 'major'). None se non interpretabile."""
    try:
        pitch, modo = str(nome).rsplit(' ', 1)
        return _PITCH_NAMES.index(pitch), ('major' if modo.startswith('magg') else 'minor')
    except Exception:
        return None


def camelot_da_tonalita(pc, modo):
    nome = _PITCH_NAMES[pc % 12]
    look = _ENHARMONIC.get(nome, nome)
    return (_CAMELOT_MAJOR if modo == 'major' else _CAMELOT_MINOR).get(look, '?')


def peso_compatibilita(compat):
    """Moltiplicatore del peso di un deck nel pescaggio degli overlay, in base alla compatibilità
    armonica col leader (Camelot): le tonalità che stanno bene insieme escono piu' spesso."""
    return {"identica": 1.0, "adiacente": 1.0, "relativa": 0.8, "dissonante": 0.3}.get(compat, 0.6)


def scegli_shift_tonalita(leader_camelot, f_nome, max_shift=2):
    """Semitoni (-2..+2, il più piccolo possibile) per portare un follower in una tonalità
    compatibile col leader. 0 se è gia' compatibile o non interpretabile."""
    parsed = parse_tonalita(f_nome)
    if not leader_camelot or not parsed:
        return 0
    pc, modo = parsed
    if camelot_compatibility(leader_camelot, camelot_da_tonalita(pc, modo)) in _TONALITA_OK:
        return 0
    for s in (1, -1, 2, -2):
        if abs(s) > max_shift:
            continue
        if camelot_compatibility(leader_camelot, camelot_da_tonalita(pc + s, modo)) in _TONALITA_OK:
            return s
    return 0


def shift_segments_in_blocks(audios, n_steps, sr, max_block_seconds=30):
    """Pitch-shift di una lista di frammenti dello stesso deck, a blocchi (come lo stretch):
    concatena, trasporta una volta, ri-divide. pitch_shift conserva la lunghezza, quindi i
    confini restano quelli originali. Se qualcosa va storto restituisce l'audio invariato."""
    out = list(audios)
    max_block = int(max_block_seconds * sr)
    blocco, indici, lung = [], [], 0

    def _flush():
        if not blocco:
            return
        try:
            big = np.concatenate(blocco, axis=1)
            sh = np.stack([librosa.effects.pitch_shift(big[c], sr=sr, n_steps=n_steps)
                           for c in range(big.shape[0])])
            pos = 0
            for a, idx in zip(blocco, indici):
                out[idx] = sh[:, pos:pos + a.shape[1]].copy()
                pos += a.shape[1]
        except Exception:
            pass

    for i, a in enumerate(audios):
        if a.shape[1] <= 0:
            continue
        if blocco and lung + a.shape[1] > max_block:
            _flush()
            blocco, indici, lung = [], [], 0
        blocco.append(a)
        indici.append(i)
        lung += a.shape[1]
    _flush()
    return out


# --- struttura del brano (solo DSP): confini di sezione -------------------------------------
def detect_section_boundaries(y_mono, sr, beat_samples, downbeat_phase=None,
                              min_gap_beats=16, kernel_beats=16):
    """Trova i confini di sezione (intro/break/drop...) come indici nella griglia dei beat.
    Metodo classico (Foote 2000): feature timbriche (mel in dB) mediate per beat, matrice di
    auto-similarità (coseno), novelty = correlazione lungo la diagonale con un kernel a
    scacchiera con finestra gaussiana, picchi con distanza minima. Se noto il downbeat, i
    confini vengono agganciati alla battuta piu' vicina. Ritorna [] se il brano e' troppo
    corto o l'analisi fallisce."""
    beats = np.asarray(beat_samples, dtype=np.int64)
    if len(beats) < 2 * kernel_beats + 4 or len(y_mono) < 8192:
        return []
    try:
        hop = 1024
        S = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_fft=2048, hop_length=hop, n_mels=40)
        L = librosa.power_to_db(S, ref=np.max)
        csum = np.concatenate([np.zeros((L.shape[0], 1)), np.cumsum(L, axis=1)], axis=1)
        F = L.shape[1]
        fb = np.clip(beats // hop, 0, F)
        feats = []
        for k in range(len(beats) - 1):
            a, b = fb[k], fb[k + 1]
            if b <= a:
                b = min(F, a + 1)
                a = max(0, b - 1)
            feats.append((csum[:, b] - csum[:, a]) / max(1, b - a))
        X = np.array(feats)
        X = X - X.mean(axis=0, keepdims=True)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        ssm = X @ X.T
        half = kernel_beats // 2
        ax = np.arange(-half, half) + 0.5
        g = np.exp(-0.5 * (ax / (half / 2.0)) ** 2)
        kernel = np.outer(g, g) * np.outer(np.sign(ax), np.sign(ax))
        padded = np.pad(ssm, half, mode='edge')
        nov = np.array([np.sum(kernel * padded[i:i + 2 * half, i:i + 2 * half]) for i in range(ssm.shape[0])])
        nov[:half] = 0.0
        nov[-half:] = 0.0
        from scipy.signal import find_peaks
        picchi, _ = find_peaks(nov, distance=min_gap_beats, height=float(np.mean(nov) + np.std(nov)))
        idx = []
        for p in picchi:
            p = int(p)
            if downbeat_phase is not None:
                r = (p - int(downbeat_phase)) % 4
                p = p - r if r <= 2 else p + (4 - r)
            if 0 < p < len(beats):
                idx.append(p)
        return sorted(set(idx))
    except Exception:
        return []


def espandi_confini_in_griglia(beat_grid, boundary_idx, bars=4, beats_per_bar=4):
    """Dai confini di sezione ai punti di piazzamento: l'inizio di ogni sezione piu' le sue
    prime `bars` battute (i DJ introducono un elemento nuovo sul cambio di sezione e nelle
    battute subito dopo, non su un unico istante)."""
    pos = set()
    for i in boundary_idx:
        for b in range(bars):
            j = i + b * beats_per_bar
            if 0 <= j < len(beat_grid):
                pos.add(beat_grid[j])
    return sorted(pos)


# --- export per Logic Pro: WAV 24-bit, stem, MIDI --------------------------------------------
def export_audio_wav24(y, sr, ceiling_db=-1.0, boost=True):
    """WAV PCM 24-bit stereo (nessuna perdita, formato nativo per Logic Pro)."""
    if y.shape[1] == 0:
        return None
    try:
        import soundfile as sf
    except Exception:
        return None
    ceiling = 10 ** (ceiling_db / 20.0)
    mx = float(np.max(np.abs(y)))
    if mx > 0 and (boost or mx > ceiling):
        y = y * (ceiling / mx)
    buf = BytesIO()
    sf.write(buf, np.ascontiguousarray(y.T.astype(np.float32)), sr, subtype='PCM_24', format='WAV')
    buf.seek(0)
    return buf


def _ascii_pulito(testo, limite=48):
    import unicodedata
    s = unicodedata.normalize('NFKD', str(testo)).encode('ascii', 'ignore').decode('ascii')
    s = ''.join(c if (c.isalnum() or c in ' -_.') else '_' for c in s).strip()
    return (s[:limite] or 'Deck')


def render_stems_zip(placements, leader, total_samples, sr, dry_peak, deck_names, info_txt=""):
    """ZIP di stem WAV 24-bit, uno per deck, TUTTI lunghi quanto il mix e allineati alla stessa
    griglia (stesso campione 0): trascinali in Logic sulla stessa posizione e si sommano al mix.
    Sono 'dry': senza compressione né normalizzazione finale (le fai tu nel DAW). Il livello e'
    comune e, se la somma superava 0 dBFS, abbassato in blocco di 1/dry_peak, cosi' gli stem
    non clippano e mantengono i rapporti tra loro. Un deck alla volta: la RAM resta quella di
    UNA traccia."""
    try:
        import soundfile as sf
    except Exception:
        return None
    scala = 1.0 / dry_peak if dry_peak and dry_peak > 1.0 else 1.0
    zbuf = BytesIO()
    lettere = sorted({p['deck'] for p in placements} | ({leader['deck']} if leader else set()))
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as zf:  # WAV non comprimono: niente deflate
        for lettera in lettere:
            buf = np.zeros((2, total_samples), dtype=np.float32)
            if leader and leader['deck'] == lettera:
                a = leader['audio']
                n = min(total_samples, a.shape[1])
                buf[:, :n] += a[:, :n].astype(np.float32)
            for p in placements:
                if p['deck'] != lettera:
                    continue
                a = p['audio']
                s = int(p['start'])
                e = min(total_samples, s + a.shape[1])
                if e > s:
                    buf[:, s:e] += (a[:, :e - s] * p.get('gain', 1.0)).astype(np.float32)
            buf *= scala
            w = BytesIO()
            sf.write(w, np.ascontiguousarray(buf.T), sr, subtype='PCM_24', format='WAV')
            nome = _ascii_pulito(deck_names.get(lettera, ''))
            ruolo = "_LEADER" if (leader and leader['deck'] == lettera) else ""
            zf.writestr(f"stem_deck{lettera.upper()}{ruolo}_{nome}.wav", w.getvalue())
            del buf, w
        if info_txt:
            zf.writestr("LEGGIMI.txt", info_txt)
    zbuf.seek(0)
    return zbuf


_MIDI_MAGGIORI = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_MIDI_MINORI = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']


def build_logic_midi(placements, sr, total_samples, tempo_bpm, deck_names, leader=None,
                     key_nome=None, markers=None, downbeat_sec=None, nota_info="", base_note=36):
    """MIDI finale per Logic Pro (Format 1) con le stesse accortezze che avevi chiesto per
    MIDI Forge: traccia 'conductor' con tempo, metro 4/4, tonalità (key signature) e MARKER;
    una traccia per deck con NOME, program change a tick 0 e canale proprio; note con la
    DURATA REALE di ogni frammento (prima erano colpi da 80ms); risoluzione 960 PPQ (quella
    interna di Logic). Una nota per frammento, altezza fissa per deck (A=C1, B=C#1...).
    Il leader (DJ Remix) e' una nota lunga quanto tutto il mix.
    Se due frammenti dello stesso deck si sovrappongono, il primo viene accorciato all'attacco
    del secondo: due note-on uguali senza note-off bloccherebbero la nota nel DAW.
    Tempo costante (BPM del leader o 120): i secondi restano esatti in ogni caso.
    I testi sono ASCII (i meta-event MIDI sono latin-1: accenti e emoji romperebbero il file)."""
    if not MIDI_DISPONIBILE or (not placements and not leader):
        return None
    PPQ = 960
    bpm = max(20.0, min(300.0, float(tempo_bpm or 120.0)))
    tempo_us = mido.bpm2tempo(bpm)

    def ticks(sec):
        return int(round(mido.second2tick(max(0.0, sec), PPQ, tempo_us)))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)

    cond = mido.MidiTrack()
    cond.append(mido.MetaMessage('track_name', name='Loop507 Mix', time=0))
    cond.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    cond.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                 clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    parsed = parse_tonalita(key_nome) if key_nome else None
    if parsed:
        pc, modo = parsed
        try:
            cond.append(mido.MetaMessage('key_signature',
                                         key=(_MIDI_MAGGIORI if modo == 'major' else _MIDI_MINORI)[pc], time=0))
        except Exception:
            pass
    if nota_info:
        cond.append(mido.MetaMessage('text', text=_ascii_pulito(nota_info, 200), time=0))
    marcatori = [(0.0, "Inizio mix")]
    if downbeat_sec is not None:
        marcatori.append((float(downbeat_sec), "Battuta 1 (downbeat leader)"))
    for sec, nome in (markers or []):
        marcatori.append((float(sec), _ascii_pulito(nome, 40)))
    abs_ev = sorted(((ticks(s), n) for s, n in marcatori), key=lambda e: e[0])
    prev = 0
    for t, nome in abs_ev:
        cond.append(mido.MetaMessage('marker', text=nome, time=max(0, t - prev)))
        prev = t
    cond.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(cond)

    lettere = sorted({p['deck'] for p in placements} | ({leader['deck']} if leader else set()))
    for i, lettera in enumerate(lettere):
        canale = i % 9  # evito il canale 10 (indice 9), riservato alla batteria GM
        nota = base_note + ('abcdefgh'.index(lettera) if lettera in 'abcdefgh' else 0)
        tr = mido.MidiTrack()
        ruolo = " (LEADER)" if (leader and leader['deck'] == lettera) else ""
        tr.append(mido.MetaMessage(
            'track_name', name=f"Deck {lettera.upper()}{ruolo} - {_ascii_pulito(deck_names.get(lettera, ''), 30)}", time=0))
        tr.append(mido.Message('program_change', channel=canale, program=0, time=0))
        eventi = []  # (start_sec, durata_sec, velocity)
        if leader and leader['deck'] == lettera:
            eventi.append((0.0, total_samples / sr, 80))
        for p in placements:
            if p['deck'] == lettera:
                vel = int(max(30, min(127, 60 + 70 * min(1.0, p.get('gain', 1.0)))))
                eventi.append((p['start'] / sr, p['length'] / sr, vel))
        eventi.sort(key=lambda e: e[0])
        # Frammenti dello stesso deck che partono sullo STESSO tick (succede: l'aggancio alla griglia
        # dei beat li fa cadere sullo stesso punto): ne tengo uno solo, il piu' lungo. Due note-on uguali
        # senza note-off in mezzo lascerebbero la nota bloccata o troncata nel DAW.
        per_tick = {}
        for s, d, vel in eventi:
            k = ticks(s)
            if k not in per_tick or d > per_tick[k][1]:
                per_tick[k] = (s, d, vel)
        eventi = [per_tick[k] for k in sorted(per_tick)]
        assoluti = []
        for k, (s, d, vel) in enumerate(eventi):
            t_on = ticks(s)
            t_off = ticks(s + d)
            if k + 1 < len(eventi):
                t_off = max(t_on + 1, min(t_off, ticks(eventi[k + 1][0])))  # mai oltre l'attacco successivo
            else:
                t_off = max(t_off, t_on + PPQ // 16)  # ultima nota: durata minima, mai zero
            assoluti.append((t_on, 1, mido.Message('note_on', channel=canale, note=nota, velocity=vel)))
            assoluti.append((t_off, 0, mido.Message('note_off', channel=canale, note=nota, velocity=0)))
        assoluti.sort(key=lambda e: (e[0], e[1]))  # a pari tick: prima i note_off
        prev = 0
        for t, _o, msg in assoluti:
            msg = msg.copy(time=max(0, t - prev))
            tr.append(msg)
            prev = t
        tr.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(tr)

    out = BytesIO()
    mid.save(file=out)
    out.seek(0)
    return out


# --- caricamento preset: rimette a posto i controlli ------------------------------------------
_TIPI_TAGLIO = ["Ritmo Musicale (BPM)", "Frenesia Casuale (Secondi)",
                "🎯 Slice per Transienti (Onset)", "👑 Deck Leader / Followers"]
_HPSS_ETICHETTE = {None: "Disattivato", "percussive": "Solo percussivo (batteria)",
                   "harmonic": "Solo armonico (melodia/basso)"}


def preset_a_widget(preset):
    """Converte un preset JSON in {chiave_widget: valore}. Funzione pura (testabile): non tocca
    session_state. Ogni voce e' validata; cio' che non e' riconosciuto viene ignorato e
    riportato negli avvisi invece di far esplodere un widget con un valore non valido."""
    v, avvisi = {}, []

    def put(chiave, valore):
        if valore is not None:
            v[chiave] = valore

    def tupla(x):
        try:
            return (float(x[0]), float(x[1]))
        except Exception:
            return None

    prep = preset.get("preparazione_audio") or {}
    put("w_correggi_ottava", prep.get("correggi_ottava"))
    if prep.get("range_bpm"):
        put("w_bpm_range_plausibile", tupla(prep["range_bpm"]))
    put("w_trim_silenzio", prep.get("trim_silenzio"))
    put("w_trim_soglia_db", prep.get("trim_soglia_db"))
    put("w_normalizza_loudness", prep.get("normalizza_lufs"))
    put("w_target_lufs", prep.get("target_lufs"))

    modo = preset.get("modalita_taglio")
    if modo in _TIPI_TAGLIO:
        v["w_tipo_taglio"] = modo
    else:
        avvisi.append("modalità di taglio non riconosciuta")
    par = preset.get("parametro_taglio") or {}
    put("w_seed_taglio_input", int(preset["seed_taglio"]) if preset.get("seed_taglio") else 0)
    put("w_usa_downbeat", preset.get("usa_downbeat"))
    if par.get("modalita") == "bpm":
        if par.get("beats") in (0.5, 1, 2, 4, 8, 1.0, 2.0, 4.0, 8.0):
            v["w_beats"] = par["beats"] if par["beats"] == 0.5 else int(par["beats"])
        if par.get("beats_range"):
            v["w_usa_range_battute"] = True
            put("w_beats_range", tupla(par["beats_range"]))
        else:
            v["w_usa_range_battute"] = False
    elif par.get("modalita") == "casuale" and par.get("range_sec"):
        put("w_range_sec", tupla(par["range_sec"]))
    elif par.get("modalita") == "onset" and par.get("durata_range_sec"):
        put("w_onset_range", tupla(par["durata_range_sec"]))

    put("w_apply_crossfade", bool(preset.get("crossfade_ms")))
    if preset.get("crossfade_ms"):
        put("w_crossfade_ms", int(preset["crossfade_ms"]))
    comp = preset.get("compressione") or {}
    put("w_apply_compressione", comp.get("attiva"))
    put("w_compressione_soglia", comp.get("soglia_db"))
    put("w_compressione_ratio", comp.get("ratio"))
    put("w_modalita_anteprima", preset.get("anteprima"))
    put("w_seed_input", int(preset["seed"]) if preset.get("seed") else 0)
    for dk, peso in (preset.get("pesi_deck") or {}).items():
        try:
            v[f"weight_{dk}"] = int(max(1, min(10, round(float(peso)))))
        except Exception:
            pass

    dj = preset.get("dj_remix")
    v["w_apply_dj_remix"] = bool(dj)
    if dj:
        put("dj_leader_key", dj.get("leader"))
        put("w_overlay_gain", dj.get("overlay_gain"))
        put("w_num_overlay_events", dj.get("num_eventi"))
        put("w_dj_bpm_align", dj.get("bpm_align"))
        put("w_dj_beatmatch", dj.get("beatmatch"))
        put("w_dj_phrase_match", bool(dj.get("beats_per_phrase")))
        if dj.get("beats_per_phrase") in (4, 8, 16, 32):
            v["w_dj_beats_per_phrase"] = dj["beats_per_phrase"]
        put("w_dj_key_check", dj.get("key_check"))
        put("w_dj_bass_swap", bool(dj.get("bass_cutoff_hz")))
        put("w_dj_bass_cutoff", dj.get("bass_cutoff_hz"))
        put("w_dj_hpss_label", _HPSS_ETICHETTE.get(dj.get("hpss")))
        put("w_dj_reverse_fx", bool(dj.get("backspin_prob")))
        put("w_dj_reverse_prob", dj.get("backspin_prob") or None)
        put("w_dj_loop_roll_fx", bool(dj.get("loop_roll_prob")))
        put("w_dj_loop_roll_prob", dj.get("loop_roll_prob") or None)
        put("w_dj_loop_roll_repeats", dj.get("loop_roll_repeats"))
        put("w_dj_echo_fx", bool(dj.get("echo_prob")))
        put("w_dj_echo_prob", dj.get("echo_prob") or None)
        put("dj_bass_sweep", dj.get("bass_sweep"))
        put("dj_tempo_ramp", dj.get("tempo_ramp"))
        put("dj_gate_prob", dj.get("gate_prob"))
        put("dj_tape_stop_prob", dj.get("tape_stop_prob"))
        put("dj_bitcrush_prob", dj.get("bitcrush_prob"))
        put("dj_key_weight", dj.get("key_weight"))
        put("dj_key_shift", dj.get("key_shift"))
        put("dj_section_match", dj.get("section_match"))
    put("export_wav24", preset.get("esporta_wav24"))
    return v, avvisi


def get_sezioni_leader(leader_d, sr, usa_downbeat):
    """Confini di sezione del leader (indici nella griglia dei beat), con cache in sessione:
    l'analisi e' costosa e il risultato non cambia finche' il leader e i suoi beat restano gli stessi."""
    cache = st.session_state.setdefault('_sezioni_cache', {})
    chiave = (leader_d.get('name'), leader_d.get('size'), len(leader_d.get('beats') or []), bool(usa_downbeat))
    if chiave not in cache:
        beats = leader_d.get('beats') or []
        y_mono = np.mean(leader_d['y'], axis=0)
        cache[chiave] = detect_section_boundaries(
            y_mono, sr, beats, leader_d.get('downbeat_phase') if usa_downbeat else None)
    return cache[chiave]


def build_waveform_image(final_y, sr, events=None, title="Loop507 Mix", placements=None,
                         bar_times=None, section_times=None):
    """PNG della forma d'onda del mix: FASCE colorate per deck dove e' stato piazzato un
    frammento (con la sua durata reale), linee sottili sulle battute della griglia del leader,
    linee tratteggiate sui confini di sezione. E' la 'foto' di cosa ha fatto il taglio: si vede
    a colpo d'occhio se i frammenti cadono sulle battute e se si agganciano alle sezioni."""
    if not MATPLOTLIB_DISPONIBILE or final_y.shape[1] == 0:
        return None
    try:
        fig, ax = plt.subplots(figsize=(12, 3.4), dpi=110)
        mono = np.mean(final_y, axis=0)
        step = max(1, len(mono) // 6000)
        tempo_asse = np.arange(0, len(mono), step) / sr
        durata = max(len(mono) / sr, 0.01)
        ax.set_xlim(0, durata)
        ax.set_facecolor("#f7f7f5")
        cmap = plt.get_cmap("tab10")
        decks = sorted({p['deck'] for p in (placements or [])} | {d for _, d in (events or [])})
        colori = {d: cmap(i % 10) for i, d in enumerate(decks)}
        if bar_times and len(bar_times) <= 400:
            for t in bar_times:
                if 0 <= t <= durata:
                    ax.axvline(t, color="#888888", alpha=0.25, linewidth=0.5, zorder=0)
        if placements:
            for p in placements:
                ax.axvspan(p['start'] / sr, (p['start'] + p['length']) / sr,
                           color=colori[p['deck']], alpha=0.20, linewidth=0, zorder=1)
        elif events:
            for start_sample, deck in events:
                ax.axvline(start_sample / sr, color=colori[deck], alpha=0.55, linewidth=1.1)
        ax.plot(tempo_asse, mono[::step], color="#2b2b2b", linewidth=0.6, zorder=2)
        ax.fill_between(tempo_asse, mono[::step], 0, color="#2b2b2b", alpha=0.15, zorder=2)
        for i, t in enumerate(section_times or []):
            if 0 <= t <= durata:
                ax.axvline(t, color="#c0392b", linestyle="--", linewidth=1.1, zorder=3)
                ax.text(t, ax.get_ylim()[1] * 0.92, f" S{i + 1}", color="#c0392b", fontsize=7, va="top")
        ax.set_xlabel("Tempo (s)")
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        if decks:
            handles = [plt.Line2D([0], [0], color=colori[d], lw=6, alpha=0.5, label=f"Deck {d.upper()}") for d in decks]
            if section_times:
                handles.append(plt.Line2D([0], [0], color="#c0392b", linestyle="--", label="sezioni"))
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=min(len(handles), 9), framealpha=0.8)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer
    except Exception:
        return None



# =====================================================================================
# v5.11: VIDEO DELL'ONDA — la stessa "foto" dell'immagine, ma animata e sincronizzata al mix.
# Solo DSP: frame disegnati con numpy + OpenCV, inviati a ffmpeg (gia' richiesto dall'app) che
# codifica H.264 + AAC. Nessuna AI, nessun matplotlib per frame (sarebbe lentissimo).
# =====================================================================================
_TAB10_HEX = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf"]
SR_VIDEO = 22050          # per disegnare la forma d'onda basta molto meno del 44.1kHz del mix
BLOCCO_VIDEO = 64         # campioni per cella dell'inviluppo min/max (~2.9ms)


def _hex_bgr(h):
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _mescola_bgr(c, bg, alpha):
    """Colore c sopra lo sfondo bg con trasparenza alpha, precalcolato (cosi' niente blending per frame)."""
    return tuple(int(round(bg[i] * (1.0 - alpha) + c[i] * alpha)) for i in range(3))


def _ascii_hud(testo):
    """I font Hershey di OpenCV non hanno accenti ne' trattini lunghi: li riduciamo ad ASCII."""
    t = str(testo).replace("\u2014", "-").replace("\u2013", "-")
    return t.encode("ascii", "ignore").decode("ascii")


def _colonne_picchi(env_min, env_max, a, b, ncols):
    """Min/max per colonna sulle celle [a, b) dell'inviluppo; fuori dal brano = silenzio (zeri)."""
    n = len(env_max)
    larg = max(int(b - a), 1)
    seg_max = np.zeros(larg, dtype=np.float32)
    seg_min = np.zeros(larg, dtype=np.float32)
    lo, hi = max(int(a), 0), min(int(b), n)
    if hi > lo:
        seg_max[lo - a:hi - a] = env_max[lo:hi]
        seg_min[lo - a:hi - a] = env_min[lo:hi]
    rel = np.minimum((np.arange(ncols) * larg / float(ncols)).astype(np.int64), larg - 1)
    return np.minimum.reduceat(seg_min, rel), np.maximum.reduceat(seg_max, rel)


def _poligono_onda(img, x0, mn, mx, yc, amp, colore):
    """Disegna la forma d'onda (area min/max) come un unico poligono: una chiamata, non una linea per colonna."""
    if len(mn) == 0:
        return
    mx = np.maximum(mx, 0.006)    # spessore minimo: anche il silenzio mostra la linea centrale, come nell'immagine
    mn = np.minimum(mn, -0.006)
    xs = np.arange(len(mn), dtype=np.float32) + x0
    alto = np.stack([xs, yc - mx * amp], axis=1)
    basso = np.stack([xs[::-1], (yc - mn * amp)[::-1]], axis=1)
    pts = np.concatenate([alto, basso]).astype(np.int32)
    cv2.fillPoly(img, [pts], colore, lineType=cv2.LINE_AA)


def _linea_tratteggiata_v(img, x, y0, y1, colore, passo=6):
    for y in range(int(y0), int(y1), passo * 2):
        cv2.line(img, (x, y), (x, min(y + passo, int(y1))), colore, 1)


def build_waveform_video(mp3_bytes, sr_eventi, placements=None, events=None, title="Loop507 Mix",
                         bar_times=None, section_times=None, larghezza=960, altezza=540, fps=24,
                         max_sec=None, finestra_sec=8.0, progress_cb=None):
    """Video MP4 dell'immagine-onda: in alto la panoramica di TUTTO il mix (come il PNG) con il cursore che
    avanza; in basso la finestra che scorre (+-4s) con fasce dei deck, battute e sezioni; la linea del
    cursore pulsa a ogni battuta del leader; sotto, VU-meter e deck attivi. Audio = l'MP3 esatto del mix.
    Ritorna (bytes_mp4, messaggio): se qualcosa non va, bytes_mp4 e' None e il messaggio dice cosa."""
    if not CV2_DISPONIBILE:
        return None, "OpenCV non disponibile: aggiungi 'opencv-python-headless' a requirements.txt."
    if not ffmpeg_bin:
        return None, "FFmpeg non trovato."
    import subprocess

    # --- 1) audio del mix -> mono 22.05kHz float, normalizzato per il disegno ---
    try:
        seg = AudioSegment.from_file(BytesIO(mp3_bytes))
        seg = seg.set_frame_rate(SR_VIDEO).set_channels(1)
        y = np.array(seg.get_array_of_samples()).astype(np.float32) / float(2 ** (8 * seg.sample_width - 1))
    except Exception as e:
        return None, f"Impossibile decodificare l'audio del mix: {e}"
    if len(y) < SR_VIDEO // 2:
        return None, "Mix troppo corto per un video."
    durata = len(y) / float(SR_VIDEO)
    dur_out = min(durata, float(max_sec)) if max_sec else durata
    n_frames = max(int(dur_out * fps), 1)

    picco = float(np.max(np.abs(y))) or 1.0
    n_blk = len(y) // BLOCCO_VIDEO
    blocchi = (y[:n_blk * BLOCCO_VIDEO] / picco).reshape(n_blk, BLOCCO_VIDEO)
    env_min, env_max = blocchi.min(axis=1), blocchi.max(axis=1)
    blk_al_sec = SR_VIDEO / float(BLOCCO_VIDEO)

    # RMS per frame (cumsum dei quadrati: vettoriale), normalizzato sul 95° percentile
    csum = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    centri = (np.arange(n_frames) / float(fps) * SR_VIDEO).astype(np.int64)
    meta = int(0.03 * SR_VIDEO)
    i0, i1 = np.clip(centri - meta, 0, len(y)), np.clip(centri + meta, 0, len(y))
    rms = np.sqrt(np.maximum(csum[i1] - csum[i0], 0.0) / np.maximum(i1 - i0, 1))
    rms = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0.0, 1.0)

    # --- 2) dati di struttura ---
    placements = placements or []
    if placements:
        ps = np.array([p['start'] / float(sr_eventi) for p in placements], dtype=np.float64)
        pe = np.array([(p['start'] + p['length']) / float(sr_eventi) for p in placements], dtype=np.float64)
        pdk = [p['deck'] for p in placements]
    elif events:
        ps = np.array([s / float(sr_eventi) for s, _ in events], dtype=np.float64)
        pe = ps + 0.25
        pdk = [d for _, d in events]
    else:
        ps = pe = np.zeros(0)
        pdk = []
    decks = sorted(set(pdk))
    bg = (245, 247, 247)  # #f7f7f5 in BGR, lo stesso sfondo dell'immagine
    colori = {d: _hex_bgr(_TAB10_HEX[i % 10]) for i, d in enumerate(decks)}
    c_band = {d: _mescola_bgr(c, bg, 0.22) for d, c in colori.items()}
    c_band_on = {d: _mescola_bgr(c, bg, 0.55) for d, c in colori.items()}
    pdk_idx = np.array([decks.index(d) for d in pdk], dtype=np.int32) if pdk else np.zeros(0, dtype=np.int32)
    bt = np.array(sorted(bar_times), dtype=np.float64) if bar_times else np.zeros(0)
    st_ = np.array(sorted(section_times), dtype=np.float64) if section_times else np.zeros(0)
    rosso, scuro, grigio = (43, 57, 192), (43, 43, 43), (150, 150, 150)  # BGR: #c0392b, #2b2b2b

    # --- 3) layout ---
    W, H = int(larghezza) // 2 * 2, int(altezza) // 2 * 2
    m = max(W // 40, 12)
    PW = W - 2 * m
    y_ov, h_ov = int(H * 0.095), int(H * 0.27)
    y_zm = y_ov + h_ov + int(H * 0.035)
    h_zm = int(H * 0.40)
    y_info = y_zm + h_zm + int(H * 0.045)
    y_meter = y_info + int(H * 0.025)
    y_leg = H - int(H * 0.06)
    fs = H / 540.0

    # --- 4) panoramica statica (due versioni: onda "da suonare" chiara e "gia' suonata" scura) ---
    mn_ov, mx_ov = _colonne_picchi(env_min, env_max, 0, n_blk, PW)

    def _panoramica(colore_onda):
        img = np.full((h_ov, PW, 3), bg, dtype=np.uint8)
        if len(bt) and len(bt) <= 400:
            for t in bt:
                if 0 <= t <= durata:
                    cv2.line(img, (int(t / durata * PW), 0), (int(t / durata * PW), h_ov), (225, 225, 225), 1)
        for k in range(len(ps)):
            x0, x1 = int(ps[k] / durata * PW), max(int(pe[k] / durata * PW), int(ps[k] / durata * PW) + 1)
            cv2.rectangle(img, (x0, 0), (x1, h_ov), c_band[decks[pdk_idx[k]]], -1)
        _poligono_onda(img, 0, mn_ov, mx_ov, h_ov // 2, h_ov * 0.46, colore_onda)
        for i, t in enumerate(st_):
            if 0 <= t <= durata:
                x = int(t / durata * PW)
                _linea_tratteggiata_v(img, x, 0, h_ov, rosso)
                cv2.putText(img, f"S{i + 1}", (x + 3, int(12 * fs) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * fs, rosso, 1, cv2.LINE_AA)
        return img

    ov_chiaro, ov_scuro = _panoramica((185, 185, 185)), _panoramica(scuro)

    # sfondo fisso del frame: titolo, cornici, legenda
    base = np.full((H, W, 3), bg, dtype=np.uint8)
    cv2.putText(base, _ascii_hud(title), (m, int(H * 0.058)), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * fs, scuro, 1, cv2.LINE_AA)
    cv2.rectangle(base, (m - 1, y_ov - 1), (m + PW, y_ov + h_ov), (205, 205, 205), 1)
    cv2.rectangle(base, (m - 1, y_zm - 1), (m + PW, y_zm + h_zm), (205, 205, 205), 1)
    cv2.putText(base, "LOOP507 // HYPER-MIXER", (m, H - int(H * 0.012)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * fs, grigio, 1, cv2.LINE_AA)

    # --- 5) ffmpeg: frame grezzi su stdin + audio dall'MP3 ---
    tmpdir = tempfile.mkdtemp(prefix="loop507_vid_")
    mp3_path, out_path, log_path = (os.path.join(tmpdir, n) for n in ("mix.mp3", "out.mp4", "ffmpeg.log"))
    try:
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)
        cmd = [ffmpeg_bin, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-i", mp3_path]
        if max_sec:
            cmd += ["-t", f"{dur_out:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
        logf = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf)
        n_win = max(int(finestra_sec * blk_al_sec), PW // 2)
        try:
            for fi in range(n_frames):
                t = fi / float(fps)
                fr = base.copy()

                # panoramica + cursore
                px = min(int(t / durata * PW), PW - 1)
                fr[y_ov:y_ov + h_ov, m:m + PW] = ov_chiaro
                fr[y_ov:y_ov + h_ov, m:m + px] = ov_scuro[:, :px]
                cv2.line(fr, (m + px, y_ov), (m + px, y_ov + h_ov), rosso, 2)

                # finestra che scorre
                t0, t1 = t - finestra_sec / 2.0, t + finestra_sec / 2.0
                pane = np.full((h_zm, PW, 3), bg, dtype=np.uint8)
                xs_of = lambda tt: (tt - t0) / finestra_sec * PW
                attivi = set()
                vis = np.nonzero((pe > t0) & (ps < t1))[0]
                for k in vis:
                    d = decks[pdk_idx[k]]
                    on = ps[k] <= t < pe[k]
                    if on:
                        attivi.add(d)
                    cv2.rectangle(pane, (int(xs_of(ps[k])), 0), (max(int(xs_of(pe[k])), int(xs_of(ps[k])) + 1), h_zm),
                                  c_band_on[d] if on else c_band[d], -1)
                if len(bt):
                    for tb in bt[(bt >= t0) & (bt <= t1)]:
                        cv2.line(pane, (int(xs_of(tb)), 0), (int(xs_of(tb)), h_zm), (215, 215, 215), 1)
                for i in np.nonzero((st_ >= t0) & (st_ <= t1))[0]:
                    x = int(xs_of(st_[i]))
                    _linea_tratteggiata_v(pane, x, 0, h_zm, rosso)
                    cv2.putText(pane, f"S{i + 1}", (x + 4, int(14 * fs) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * fs, rosso, 1, cv2.LINE_AA)
                a = int(t0 * blk_al_sec)
                mn, mx = _colonne_picchi(env_min, env_max, a, a + n_win, PW)
                _poligono_onda(pane, 0, mn[:PW // 2], mx[:PW // 2], h_zm // 2, h_zm * 0.46, scuro)
                _poligono_onda(pane, PW // 2, mn[PW // 2:], mx[PW // 2:], h_zm // 2, h_zm * 0.46, (120, 120, 120))

                # cursore: pulsa sulle battute
                flash = 0.0
                if len(bt):
                    j = np.searchsorted(bt, t, side="right") - 1
                    if j >= 0:
                        flash = max(0.0, 1.0 - (t - bt[j]) / 0.14)
                cv2.line(pane, (PW // 2, 0), (PW // 2, h_zm), rosso, 2 + int(round(4 * flash)))
                fr[y_zm:y_zm + h_zm, m:m + PW] = pane

                # testi: tempo, battuta, sezione
                mm, ss = divmod(t, 60.0)
                tm, ts = divmod(durata, 60.0)
                testo_t = f"{int(mm):02d}:{ss:04.1f} / {int(tm):02d}:{ts:04.1f}"
                (tw, _), _ = cv2.getTextSize(testo_t, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * fs, 1)
                cv2.putText(fr, testo_t, (W - m - tw, int(H * 0.058)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * fs, scuro, 1, cv2.LINE_AA)
                bar_n = int(np.searchsorted(bt, t, side="right")) if len(bt) else 0
                sez_n = int(np.searchsorted(st_, t, side="right")) if len(st_) else 0
                info = (f"BAR {bar_n}" if len(bt) else "BAR -") + f"   SEZ {sez_n if len(st_) else '-'}"
                cv2.putText(fr, info, (m, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs, scuro, 1, cv2.LINE_AA)

                # VU-meter
                cv2.rectangle(fr, (m, y_meter), (m + PW, y_meter + max(int(6 * fs), 4)), (225, 225, 225), -1)
                cv2.rectangle(fr, (m, y_meter), (m + int(rms[fi] * PW), y_meter + max(int(6 * fs), 4)),
                              colori[sorted(attivi)[0]] if attivi else scuro, -1)

                # legenda deck: pieno = sta suonando adesso
                x = m
                for d in decks:
                    lato = max(int(12 * fs), 8)
                    if d in attivi:
                        cv2.rectangle(fr, (x, y_leg), (x + lato, y_leg + lato), colori[d], -1)
                    else:
                        cv2.rectangle(fr, (x, y_leg), (x + lato, y_leg + lato), colori[d], 1)
                    cv2.putText(fr, f"DECK {str(d).upper()}", (x + lato + 5, y_leg + lato - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40 * fs, scuro, 1, cv2.LINE_AA)
                    x += int(88 * fs)

                proc.stdin.write(fr.tobytes())
                if progress_cb and fi % 12 == 0:
                    progress_cb(min(fi / float(n_frames), 1.0))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = proc.wait()
        logf.close()
        if rc != 0 or not os.path.exists(out_path):
            try:
                with open(log_path, "rb") as f:
                    err = f.read().decode("utf-8", "ignore").strip()[-400:]
            except Exception:
                err = ""
            return None, f"ffmpeg ha fallito (codice {rc}). {err}"
        with open(out_path, "rb") as f:
            dati = f.read()
        if progress_cb:
            progress_cb(1.0)
        return dati, f"Video {W}x{H} @ {fps}fps, {dur_out:.1f}s, {len(dati) / 1e6:.1f} MB"
    except Exception as e:
        return None, f"Errore nel rendering del video: {e}"
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


# =====================================================================================
# v5.10: MIDI TRASCRITTO DALL'AUDIO DEL MIX (batteria + basso + armonia), solo DSP
# =====================================================================================
# Note General MIDI per la batteria (canale 10)
GM_KICK, GM_SNARE, GM_HAT_CLOSED, GM_HAT_OPEN = 36, 38, 42, 46
SR_TRASCRIZIONE = 22050  # le bande che servono (basso, medi, hi-hat fino a ~10kHz) stanno sotto 11kHz


_BANDE_TRASC = [(30, 120), (120, 300), (300, 1200), (1200, 3500), (3500, 6500), (6500, 10000)]


def _band_features(y, sr, n_fft=1024, hop=256):
    """Energia (potenza lineare) in 6 bande per ogni frame, calcolata a BLOCCHI: la STFT intera di un brano pesa centinaia di MB.
    Bande: 30-120 / 120-300 / 300-1200 / 1200-3500 / 3500-6500 / 6500-10000 Hz."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in _BANDE_TRASC]
    chunk = 2048
    E = []
    n_frames = 1 + max(0, (len(y) - n_fft) // hop)
    for f0 in range(0, n_frames, chunk):
        a = f0 * hop
        seg = y[a:a + (chunk - 1) * hop + n_fft]
        if len(seg) < n_fft:
            break
        S = np.abs(librosa.stft(seg.astype(np.float32), n_fft=n_fft, hop_length=hop, center=False)) ** 2
        E.append(np.stack([S[ix].sum(axis=0) for ix in idx]))
    if not E:
        return {}
    return {'E': np.concatenate(E, axis=1), 'fps': sr / hop, 't0': (n_fft * 0.5) / sr}


def transcribe_drums(y, sr, sensibilita=5):
    """Batteria dal suono del mix: kick, rullante, hi-hat chiuso/aperto -> [(sec, nota GM, velocity)].
    1) ONSET: funzione di novita' = somma, sulle 6 bande, del flusso positivo dell'energia
       (log-compressa e normalizzata per banda); picchi sopra una frazione del 95° percentile.
    2) CLASSIFICAZIONE sull'aumento di energia LINEARE a ogni attacco (dopo - prima), cioe' su
       'cosa e' stato aggiunto' e non su cio' che gia' suonava (basso, pad):
       - KICK: l'aumento sta soprattutto nelle bande 30-300Hz e l'energia bassa decade in meno
         di ~0.2s (un'altra nota di basso, che si sostiene, non e' un kick);
       - HI-HAT: l'aumento sopra i 300Hz e' concentrato nelle bande 3.5-10kHz (>= 75%);
         APERTO se quell'energia resta sopra il 30% del picco per piu' di ~100ms;
       - RULLANTE/CLAP: l'aumento sopra i 300Hz e' rumoroso (flatness >= 0.10) e ha una quota
         significativa nei medi 300Hz-3.5kHz, senza essere un hi-hat.
       Le parti 'basse' e 'alte' si valutano separatamente: kick e rullante insieme danno due note.
    `sensibilita` 1-10: piu' alta = soglia piu' bassa = piu' colpi."""
    f = _band_features(y, sr)
    if not f or f['E'].shape[1] < 16:
        return []
    E, fps, t0 = f['E'], f['fps'], f['t0']
    nb, nf = E.shape
    # novita': flusso positivo della banda log-compressa e normalizzata sul suo 95° percentile
    L = np.log1p(20.0 * E / (np.percentile(E, 95, axis=1, keepdims=True) + 1e-12))
    odf = np.maximum(0.0, np.diff(L, axis=1, prepend=L[:, :1])).sum(axis=0)
    attivi = odf[odf > 0]
    if len(attivi) < 8:
        return []
    frac = float(np.interp(sensibilita, [1, 5, 10], [0.50, 0.28, 0.12]))
    from scipy.signal import find_peaks
    picchi, _ = find_peaks(odf, height=frac * float(np.percentile(attivi, 95)), distance=max(1, int(0.045 * fps)))
    # Gli hi-hat sono molto piu' deboli di kick e rullante: con una soglia unica sulla novita'
    # complessiva verrebbero scartati. Cerco quindi gli attacchi anche sulla sola novita' delle
    # bande alte (3.5-10kHz), con la sua soglia, e unisco i due insiemi (niente doppioni a < 30ms).
    odf_h = np.maximum(0.0, np.diff(L[4:], axis=1, prepend=L[4:, :1])).sum(axis=0)
    att_h = odf_h[odf_h > 0]
    if len(att_h) >= 8:
        ph, _ = find_peaks(odf_h, height=frac * float(np.percentile(att_h, 90)), distance=max(1, int(0.045 * fps)))
        extra = [p for p in ph if not any(abs(int(p) - int(q)) <= int(0.03 * fps) for q in picchi)]
        picchi = np.array(sorted(set(int(q) for q in picchi) | set(int(p) for p in extra)))
    if len(picchi) == 0:
        return []
    # Per ogni attacco: energia PRIMA e DOPO in ogni banda (lineare)
    pre = np.zeros((nb, len(picchi)))
    post = np.zeros((nb, len(picchi)))
    for n, p in enumerate(picchi):
        pre[:, n] = E[:, max(0, p - 3):max(1, p - 1)].mean(axis=1) if p > 1 else E[:, 0]
        post[:, n] = E[:, p:min(nf, p + 3)].max(axis=1)
    dE = np.maximum(0.0, post - pre)
    eps = 1e-12
    # L'attacco deve ALZARE l'energia della sua banda di almeno ~4dB rispetto a cio' che gia'
    # suonava (x2.5 in potenza; x2 per il kick): scarta le fluttuazioni di un fondo rumoroso.
    rel_low = (post[0] + post[1]) / (pre[0] + pre[1] + eps)
    rel_mid = (post[2] + post[3]) / (pre[2] + pre[3] + eps)
    rel_hi = (post[4] + post[5]) / (pre[4] + pre[5] + eps)
    low = dE[0] + dE[1]
    mid_inc = dE[2] + dE[3]
    hi_inc = dE[4] + dE[5]
    e_low = E[0] + E[1]
    e_hi = E[4] + E[5]
    n_fft, hop = 1024, 256
    fr = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bins_mid = np.where((fr >= 300) & (fr < 3500))[0]
    bins_hi = np.where((fr >= 3500) & (fr < 10000))[0]
    finestra = np.hanning(n_fft)

    def flatness_aggiunta(p, bins):
        """Flatness (1 = rumore bianco, ~0 = pochi picchi tonali) dello spettro AGGIUNTO
        dall'attacco (potenza dopo - prima) nelle bande indicate. Misurarla sull'aumento e
        non sul suono intero evita che un fondo rumoroso faccia sembrare 'rumoroso' un synth."""
        a_post = (p + 2) * hop
        a_pre = max(0, (p - 3) * hop)
        if a_post + n_fft > len(y):
            return 0.0
        X_post = np.abs(np.fft.rfft(y[a_post:a_post + n_fft] * finestra)) ** 2
        X_pre = np.abs(np.fft.rfft(y[a_pre:a_pre + n_fft] * finestra)) ** 2
        d = np.maximum(0.0, X_post - X_pre)[bins]
        if d.max() <= 0:
            return 0.0
        d = d + 1e-3 * d.max()
        return float(np.exp(np.mean(np.log(d))) / np.mean(d))

    def decadimento(e, p, frazione):
        pk = e[p:min(nf, p + 4)].max()
        dopo = e[p + 1:p + 1 + int(0.30 * fps)]
        sotto = np.where(dopo < frazione * pk)[0]
        return (sotto[0] if len(sotto) else len(dopo)) / fps

    # PASSO 1 - candidati per classe. La soglia d'ampiezza di ogni classe e' relativa alla
    # distribuzione DELLA CLASSE STESSA (un hi-hat e' molto piu' debole di un rullante).
    cand_kick, cand_hat, cand_snare = [], [], []
    for n, p in enumerate(picchi):
        p = int(p)
        if (low[n] > 0 and rel_low[n] >= 2.0 and low[n] >= 0.5 * (mid_inc[n] + hi_inc[n] + eps)
                and decadimento(e_low, p, 0.25) <= 0.20):
            cand_kick.append(n)
        rapporto = hi_inc[n] / (mid_inc[n] + eps)  # alti/medi: un hi-hat sposta l'energia verso gli alti piu' di un rullante
        f_mid = flatness_aggiunta(p, bins_mid) if mid_inc[n] > 0 else 0.0
        f_hi = flatness_aggiunta(p, bins_hi) if hi_inc[n] > 0 else 0.0
        # RULLANTE: l'aumento nei medi e' RUMOROSO, e il rumore sale anche negli alti (le casse
        # del rullante arrivano a 5-10kHz): un aumento solo nei medi e' un synth/voce, non un rullante.
        if mid_inc[n] > 0 and rel_mid[n] >= 2.5 and 0.15 <= rapporto < 4.0 and rel_hi[n] >= 1.5 and f_mid >= 0.25:
            cand_snare.append(n)
        # HI-HAT: aumento rumoroso negli alti. Se nei medi c'e' un aumento TONALE (una nota di synth
        # che cade sullo stesso battito) non conta contro l'hi-hat: il rapporto alti/medi si
        # valuta solo quando anche i medi sono rumorosi (cioe' potrebbe essere un rullante).
        if hi_inc[n] > 0 and rel_hi[n] >= 2.5 and f_hi >= 0.25 and (rapporto >= 4.0 or f_mid < 0.25):
            cand_hat.append(n)
    eventi = []
    soglia_kick = float(np.interp(sensibilita, [1, 5, 10], [0.60, 0.40, 0.20]))
    soglia_snare = float(np.interp(sensibilita, [1, 5, 10], [0.55, 0.35, 0.15]))

    def velocita(v, ref):
        # scala in dB: 0 dB (= 90° percentile della classe) -> 127, -24 dB -> 35
        db = 10.0 * np.log10(max(v / ref, 1e-6))
        return int(max(35, min(127, 35 + 92 * np.clip((db + 24.0) / 24.0, 0.0, 1.0))))

    if cand_kick:
        ref = float(np.percentile(low[cand_kick], 90)) + eps
        for n in cand_kick:
            if low[n] >= soglia_kick * ref:
                eventi.append((t0 + picchi[n] / fps, GM_KICK, velocita(low[n], ref)))
    if cand_snare:
        ref = float(np.percentile(mid_inc[cand_snare], 90)) + eps
        for n in cand_snare:
            if mid_inc[n] >= soglia_snare * ref:
                eventi.append((t0 + picchi[n] / fps, GM_SNARE, velocita(mid_inc[n], ref)))
    if cand_hat:
        ref = float(np.percentile(hi_inc[cand_hat], 90)) + eps
        for n in cand_hat:
            if hi_inc[n] >= 0.03 * ref:
                durata = decadimento(e_hi, int(picchi[n]), 0.30)
                eventi.append((t0 + picchi[n] / fps, GM_HAT_OPEN if durata > 0.10 else GM_HAT_CLOSED,
                               velocita(hi_inc[n], ref)))
    eventi.sort(key=lambda e: e[0])
    return eventi


def transcribe_bass(y, sr, kick_times=None):
    """Basso monofonico dal suono del mix -> [(start_s, durata_s, nota MIDI, velocity)].
    Passa-basso a 260Hz, pitch con YIN (algoritmo classico, niente ML), nota = f0 arrotondata,
    mediana su 7 frame, note solo se la zona e' energica e stabile per almeno ~80ms. Un kick
    ha un glide verso il basso che YIN scambia per una nota breve: scarto le note brevi
    (< 160ms) che partono sull'attacco di un kick."""
    try:
        from scipy.signal import butter, sosfiltfilt
        from scipy.ndimage import median_filter
        sos = butter(4, 260.0, btype='lowpass', fs=sr, output='sos')
        yl = sosfiltfilt(sos, y.astype(np.float64)).astype(np.float32)
        hop = 256
        f0 = librosa.yin(yl, fmin=32.0, fmax=260.0, sr=sr, frame_length=2048, hop_length=hop)
        rms = librosa.feature.rms(y=yl, frame_length=2048, hop_length=hop)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        gate = 0.12 * float(np.percentile(rms, 95))
        voiced = (rms > gate) & (f0 > 30) & (f0 < 270)
        midi = np.where(voiced, np.round(librosa.hz_to_midi(np.maximum(f0, 1.0))), -1).astype(int)
        midi = median_filter(midi, size=7, mode='nearest')
        fps = sr / hop
        note, i = [], 0
        while i < n:
            if midi[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < n and midi[j + 1] == midi[i]:
                j += 1
            # unisco micro-buchi (<= 2 frame) tra due run della stessa nota
            while j + 3 < n and midi[j + 1] < 0 and midi[j + 2] == midi[i] or (j + 3 < n and midi[j + 1] < 0 and midi[j + 3] == midi[i] and midi[j + 2] < 0 and False):
                j += 2
                while j + 1 < n and midi[j + 1] == midi[i]:
                    j += 1
            dur = (j - i + 1) / fps
            if dur >= 0.08:
                t = i / fps
                v = int(max(40, min(120, 40 + 80 * min(1.0, float(np.mean(rms[i:j + 1])) / (float(np.percentile(rms, 95)) + 1e-9)))))
                note.append((t, dur, int(midi[i]), v))
            i = j + 1
        if kick_times is not None and len(kick_times):
            kt = np.asarray(kick_times)
            note = [x for x in note if not (x[1] < 0.16 and np.any((kt > x[0] - 0.03) & (kt < x[0] + 0.06)))]
        return note
    except Exception:
        return []


def transcribe_harmony(y, sr, max_voci=4):
    """Armonia/lead polifonica (STIMA) dal suono del mix -> [(start_s, durata_s, nota MIDI, velocity)].
    CQT a 36 bin/ottava (C3-C7) raggruppata a semitoni, mediana nel tempo (tiene il sostenuto,
    scarta i colpi percussivi), fino a `max_voci` picchi per frame sopra una soglia relativa,
    con soppressione delle ARMONICHE (un picco a +12/+19/+24 semitoni da uno piu' forte e'
    quasi sempre una sua armonica, non una nota) e durata minima ~140ms. La trascrizione
    polifonica da audio mixato e' un problema aperto: aspettati note sbagliate/mancanti, utile
    come bozza da correggere nel DAW, non come partitura."""
    try:
        from scipy.ndimage import median_filter
        hop = 512
        fmin = librosa.note_to_hz('C3')
        C = np.abs(librosa.cqt(y.astype(np.float32), sr=sr, hop_length=hop, fmin=fmin,
                               n_bins=144, bins_per_octave=36))
        n_f = C.shape[1]
        sem = C.reshape(48, 3, n_f).max(axis=1)           # 48 semitoni: MIDI 48..95
        sem = median_filter(sem, size=(1, 9), mode='nearest')
        floor = 0.06 * float(sem.max()) + 1e-12
        A = np.zeros_like(sem, dtype=bool)
        for t in range(n_f):
            col = sem[:, t]
            fm = col.max()
            if fm < floor:
                continue
            cand = [i for i in range(1, 47) if col[i] > col[i - 1] and col[i] >= col[i + 1] and col[i] >= 0.35 * fm]
            cand.sort(key=lambda i: -col[i])
            scelti = []
            for i in cand:
                if any((i - s) in (12, 19, 24, 28) and col[i] < 0.8 * col[s] for s in scelti):
                    continue  # armonica di una nota piu' forte
                scelti.append(i)
                if len(scelti) >= max_voci:
                    break
            for i in scelti:
                A[i, t] = True
        fps = sr / hop
        out = []
        for i in range(48):
            r = A[i].copy()
            # chiudo i buchi di 1 frame
            for t in range(1, n_f - 1):
                if not r[t] and r[t - 1] and r[t + 1]:
                    r[t] = True
            t = 0
            while t < n_f:
                if not r[t]:
                    t += 1
                    continue
                u = t
                while u + 1 < n_f and r[u + 1]:
                    u += 1
                dur = (u - t + 1) / fps
                if dur >= 0.14:
                    ampl = float(np.mean(sem[i, t:u + 1])) / (float(sem.max()) + 1e-12)
                    out.append((t / fps, dur, 48 + i, int(max(40, min(115, 40 + 90 * min(1.0, ampl * 1.4))))))
                t = u + 1
        out.sort(key=lambda e: (e[0], e[2]))
        return out
    except Exception:
        return []


def build_transcribed_midi(drums, bass, harmony, tempo_bpm, key_nome=None, markers=None,
                           downbeat_sec=None, quantize=False, nota_info=""):
    """MIDI Format 1 per Logic dalle note trascritte: conductor (tempo, 4/4, tonalità, marker),
    traccia Batteria sul CANALE 10 (note General MIDI: kick 36, rullante 38, hi-hat 42/46),
    traccia Basso e traccia Armonia/Lead. 960 PPQ. Stesse accortezze del MIDI struttura:
    nomi traccia ASCII, program change a tick 0, note-off prima del note-on allo stesso tick,
    mai due note-on uguali senza note-off in mezzo. `quantize`: attacchi agganciati alla griglia
    di 1/16 (se entro il 40% del passo), con origine sul downbeat se noto."""
    if not MIDI_DISPONIBILE or not (drums or bass or harmony):
        return None
    PPQ = 960
    bpm = max(20.0, min(300.0, float(tempo_bpm or 120.0)))
    tempo_us = mido.bpm2tempo(bpm)
    passo = 60.0 / bpm / 4.0
    origine = float(downbeat_sec) if downbeat_sec is not None else 0.0

    def q(sec):
        if not quantize:
            return sec
        k = round((sec - origine) / passo)
        agg = origine + k * passo
        return agg if abs(sec - agg) <= 0.4 * passo else sec

    def ticks(sec):
        return int(round(mido.second2tick(max(0.0, sec), PPQ, tempo_us)))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    cond = mido.MidiTrack()
    cond.append(mido.MetaMessage('track_name', name='Loop507 Mix (trascrizione)', time=0))
    cond.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    cond.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                 clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    parsed = parse_tonalita(key_nome) if key_nome else None
    if parsed:
        pc, modo = parsed
        try:
            cond.append(mido.MetaMessage('key_signature',
                                         key=(_MIDI_MAGGIORI if modo == 'major' else _MIDI_MINORI)[pc], time=0))
        except Exception:
            pass
    if nota_info:
        cond.append(mido.MetaMessage('text', text=_ascii_pulito(nota_info, 200), time=0))
    marc = [(0.0, "Inizio mix")]
    if downbeat_sec is not None:
        marc.append((float(downbeat_sec), "Battuta 1 (downbeat leader)"))
    for sec, nome in (markers or []):
        marc.append((float(sec), _ascii_pulito(nome, 40)))
    prev = 0
    for t, nome in sorted(((ticks(s), n) for s, n in marc), key=lambda e: e[0]):
        cond.append(mido.MetaMessage('marker', text=nome, time=max(0, t - prev)))
        prev = t
    cond.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(cond)

    def aggiungi_traccia(nome, canale, programma, eventi):
        """eventi = [(start_s, durata_s, nota, vel)]"""
        if not eventi:
            return
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('track_name', name=nome, time=0))
        if programma is not None:
            tr.append(mido.Message('program_change', channel=canale, program=programma, time=0))
        per_nota = {}
        for s, d, n, v in eventi:
            per_nota.setdefault(n, []).append((q(s), d, v))
        assoluti = []
        for n, lst in per_nota.items():
            lst.sort(key=lambda e: e[0])
            ticks_on = []
            for s, d, v in lst:
                k = ticks(s)
                if ticks_on and ticks_on[-1][0] == k:
                    continue  # stesso tick, stessa nota: ne tengo una
                ticks_on.append((k, s, d, v))
            for idx, (t_on, s, d, v) in enumerate(ticks_on):
                t_off = ticks(s + d)
                if idx + 1 < len(ticks_on):
                    t_off = max(t_on + 1, min(t_off, ticks_on[idx + 1][0]))
                else:
                    t_off = max(t_off, t_on + PPQ // 16)
                assoluti.append((t_on, 1, mido.Message('note_on', channel=canale, note=int(n), velocity=int(v))))
                assoluti.append((t_off, 0, mido.Message('note_off', channel=canale, note=int(n), velocity=0)))
        assoluti.sort(key=lambda e: (e[0], e[1]))
        prev = 0
        for t, _o, msg in assoluti:
            tr.append(msg.copy(time=max(0, t - prev)))
            prev = t
        tr.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(tr)

    # batteria: durata fissa breve (1/32 di nota), canale 10 (indice 9)
    aggiungi_traccia("Batteria (GM)", 9, None, [(s, 60.0 / bpm / 8.0, n, v) for s, n, v in drums])
    aggiungi_traccia("Basso (stima)", 0, 38, bass)
    aggiungi_traccia("Armonia-Lead (stima)", 1, 80, harmony)
    out = BytesIO()
    mid.save(file=out)
    out.seek(0)
    return out


def trascrivi_mix(y_mono, sr, tempo_bpm, key_nome=None, markers=None, downbeat_sec=None,
                  sensibilita=5, con_basso=True, con_armonia=True, quantize=False, nota_info=""):
    """Pipeline completa: batteria, basso (se richiesto), armonia (se richiesta) -> (BytesIO MIDI, conteggi)."""
    drums = transcribe_drums(y_mono, sr, sensibilita)
    kick_t = [t for t, n, _v in drums if n == GM_KICK]
    bass = transcribe_bass(y_mono, sr, kick_t) if con_basso else []
    harm = transcribe_harmony(y_mono, sr) if con_armonia else []
    if harm:
        # Un rullante o un kick hanno un corpo tonale (es. ~190Hz) che la CQT scambia per una nota:
        # scarto le note BREVI (< 250ms) che partono entro 50ms da un attacco di batteria.
        colpi = np.array([t for t, n, _v in drums if n in (GM_KICK, GM_SNARE)])
        if len(colpi):
            harm = [h for h in harm if not (h[1] < 0.25 and np.any(np.abs(colpi - h[0]) <= 0.05))]
    midi = build_transcribed_midi(drums, bass, harm, tempo_bpm, key_nome, markers, downbeat_sec, quantize, nota_info)
    conteggi = {'kick': sum(1 for e in drums if e[1] == GM_KICK), 'rullante': sum(1 for e in drums if e[1] == GM_SNARE),
                'hihat': sum(1 for e in drums if e[1] in (GM_HAT_CLOSED, GM_HAT_OPEN)),
                'basso': len(bass), 'armonia': len(harm)}
    return midi, conteggi



# --- Interfaccia Utente Streamlit ---
st.set_page_config(page_title="Loop507 Hyper-Mixer", layout="wide")
st.title("🎧 Loop507: Audio Shuffler & Glitcher")

if 'decks' not in st.session_state:
    st.session_state.decks = {
        k: {'y': None, 'sr': None, 'tempo': 0.0, 'tempo_detected': 0.0, 'name': None, 'beats': [], 'downbeat_phase': None, 'size': None, 'prep_sig': None}
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
        _d.setdefault('beats', [])
        _d.setdefault('downbeat_phase', None)
        _d.setdefault('size', None)
        _d.setdefault('prep_sig', None)
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'audio_report' not in st.session_state:
    st.session_state.audio_report = ""

st.sidebar.header("⚙️ Precisione BPM")
_preset_up = st.sidebar.file_uploader(
    "📂 Carica preset (.json)", type=["json"], key="preset_loader",
    help="Rimette a posto i controlli con i valori di un preset scaricato (modalità, parametri, "
         "seed, pesi, opzioni DJ). I FILE audio vanno ricaricati nei deck; dopo aver caricato il "
         "preset rifai il taglio (stesso seed taglio) e genera col seed mix: otterrai lo stesso mix."
)
if _preset_up is not None:
    _pid = (_preset_up.name, getattr(_preset_up, 'size', None))
    if st.session_state.get('_preset_caricato') != _pid:
        try:
            _p = json.loads(_preset_up.getvalue().decode('utf-8'))
            if not (isinstance(_p, dict) and _p.get("loop507_hyper_mixer_preset")):
                raise ValueError("non e' un preset di Hyper-Mixer")
            _vals, _avv = preset_a_widget(_p)
            _caricati = {k for k, d in st.session_state.decks.items() if d.get('y') is not None}
            if "dj_leader_key" in _vals and _vals["dj_leader_key"] not in _caricati:
                _avv.append(f"leader {str(_vals['w_dj_leader_key']).upper()} non caricato: scegli il leader a mano")
                del _vals["dj_leader_key"]
            for _k, _val in _vals.items():
                st.session_state[_k] = _val  # prima che i widget vengano creati in questa esecuzione
            st.session_state['_preset_caricato'] = _pid
            st.session_state['_preset_msg'] = (
                f"Preset caricato ({len(_vals)} controlli)." + (" Avvisi: " + "; ".join(_avv) if _avv else ""))
        except Exception as _e:
            st.session_state['_preset_caricato'] = _pid
            st.session_state['_preset_msg'] = f"ERR:Preset non valido: {_e}"
    _m = st.session_state.get('_preset_msg', '')
    if _m.startswith("ERR:"):
        st.sidebar.error(_m[4:])
    elif _m:
        st.sidebar.success(_m)

correggi_ottava = st.sidebar.checkbox(
    "Correggi errori di ottava", value=True,
    help="I beat-tracker confondono spesso il doppio o la metà del BPM reale (es. 70 invece "
         "di 140). Se attivo, il BPM rilevato fuori dal range qui sotto viene automaticamente "
         "dimezzato/raddoppiato per rientrarci. Disattivalo per generi con BPM legittimamente "
         "molto lenti o molto veloci (es. drone, hardcore estremo)."
, key="w_correggi_ottava")
bpm_range_plausibile = st.sidebar.slider(
    "Range BPM plausibile:", 40, 220, (70, 180), disabled=not correggi_ottava
, key="w_bpm_range_plausibile")

st.sidebar.header("🎚️ Preparazione Audio")
normalizza_loudness = st.sidebar.checkbox(
    "Normalizza Loudness (LUFS)", value=False,
    help="Porta tutti i deck allo stesso volume percepito (standard ITU-R BS.1770-4, via "
         "pyloudnorm) prima di usarli. Senza questo, deck masterizzati a volumi molto diversi "
         "rendono impreciso qualunque bilanciamento (es. overlay_gain in DJ Remix)."
, key="w_normalizza_loudness")
target_lufs = st.sidebar.slider(
    "Target LUFS:", -30.0, -6.0, -18.0, step=0.5, disabled=not normalizza_loudness,
    help="-18 LUFS è un target comune per musica non da streaming. Valori più alti (es. -12) = più forte."
, key="w_target_lufs")
trim_silenzio = st.sidebar.checkbox(
    "Taglia silenzio in testa/coda", value=False,
    help="Rimuove il silenzio all'inizio e alla fine di ogni deck. Utile soprattutto per il "
         "deck leader in DJ Remix: senza questo, il silenzio è 'spazio morto' dove un overlay "
         "può finire piazzato inutilmente."
, key="w_trim_silenzio")
trim_soglia_db = st.sidebar.slider(
    "Soglia silenzio (dB sotto il picco):", 20, 60, 40, disabled=not trim_silenzio,
    help="Più alto = considera silenzio anche suoni non proprio a zero (taglio più aggressivo)."
, key="w_trim_soglia_db")

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 BATCH BPM MATCHER — tool separato e indipendente dall'Hyper-Mixer.
# Usa lo stesso session_state prefix "batch_" per non toccare in alcun modo lo
# stato dei deck/segmenti/mix: zero rischio di interferenza con il resto dell'app.
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📦 Batch BPM Matcher — porta più tracce alla stessa velocità", expanded=False):
    st.caption(
        "Carica 5-10 mp3/wav: vengono analizzati (BPM raffinato + tonalità Camelot, stessi "
        "algoritmi dell'Hyper-Mixer), poi puoi portarli tutti alla stessa velocità (intonazione "
        "preservata) e scaricarli in un unico ZIP."
    )

    if 'batch_results' not in st.session_state:
        # lista di dict (SOLO metadati, niente audio decodificato: name, tempo, key_nome, camelot, prep...)
        st.session_state.batch_results = []
    if 'batch_zip' not in st.session_state:
        st.session_state.batch_zip = None

    batch_files = st.file_uploader(
        "Carica le tracce (mp3/wav):", type=["mp3", "wav"], accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files and st.button("🔍 Analizza tutte le tracce", key="batch_analyze_btn"):
        st.session_state.batch_results = []
        st.session_state.batch_zip = None  # un nuovo batch invalida lo zip precedente
        st.session_state.batch_report = None
        progress = st.progress(0.0, text="Analizzo...")
        for i, bf in enumerate(batch_files):
            with st.spinner(f"Analizzo {bf.name}..."):
                # Niente cache e niente audio in session_state: il brano decodificato (~100MB per
                # 5 minuti stereo) vive solo il tempo dell'analisi, poi viene rilasciato. Prima
                # 10 tracce restavano tutte in RAM (~1GB, il limite di Streamlit Cloud) piu'
                # le copie nella cache. All'allineamento il brano viene riletto uno alla volta.
                y, sr, t, _beats, _fase_db = analyze_track_uncached(bf)
                artista = extract_artist(bf)
                if correggi_ottava and t > 0:
                    t = correggi_errore_ottava(t, bpm_range_plausibile[0], bpm_range_plausibile[1])
                prep = {
                    'trim': bool(trim_silenzio), 'trim_db': trim_soglia_db,
                    'normalizza': bool(normalizza_loudness), 'target_lufs': target_lufs,
                }
                y = applica_preparazione(y, sr, **prep)
                key_nome, camelot, _ = detect_key_camelot(y, sr)
                ambiguita = check_tempo_ambiguity(y, sr, t) if t > 0 else []
                crosscheck = cross_check_tempo(y, sr, t) if t > 0 else {"esito": "n/d", "feature_tempo": None, "ioi_tempo": None}
                st.session_state.batch_results.append({
                    'name': bf.name, 'artist': artista, 'tempo': t,
                    'key_nome': key_nome or "n/d", 'camelot': camelot or "?",
                    'ambiguita': ambiguita, 'crosscheck': crosscheck, 'prep': prep
                })
                del y  # rilascio subito l'audio decodificato
            progress.progress((i + 1) / len(batch_files), text=f"Analizzato {bf.name}")
        progress.empty()

    if st.session_state.batch_results:
        st.write("**Tracce analizzate** (correggi il BPM a mano se sai che è sbagliato — la rilevazione automatica non è mai garantita al 100%):")
        for i, r in enumerate(st.session_state.batch_results):
            col_info, col_bpm = st.columns([3, 1])
            with col_info:
                if r['tempo'] > 0:
                    ambiguita = r.get('ambiguita', [])
                    cc = r.get('crosscheck', {})
                    esito = cc.get('esito', 'n/d')
                    st.caption(f"🎵 {r['name']} — {r['key_nome']} ({r['camelot']})")
                    if esito.startswith("confermato"):
                        st.caption(f"✅ Verifica incrociata: {esito} (feature.tempo: {cc.get('feature_tempo')}, IOI: {cc.get('ioi_tempo')})")
                    elif esito.startswith("probabile errore"):
                        st.caption(f"🔴 {esito.upper()} — feature.tempo: {cc.get('feature_tempo')} BPM, IOI: {cc.get('ioi_tempo')} BPM. Controlla ad orecchio: probabilmente uno di questi due è quello giusto.")
                    elif esito.startswith("parzialmente"):
                        st.caption(f"🟡 Verifica incrociata: {esito} (feature.tempo: {cc.get('feature_tempo')}, IOI: {cc.get('ioi_tempo')})")
                    if ambiguita:
                        alt_str = ", ".join(f"{bpm} ({nome})" for bpm, nome, _ in ambiguita[:2])
                        st.caption(f"⚠️ Possibile ambiguità ritmica — BPM alternativi plausibili: {alt_str}. Verifica ad orecchio o dal tag del brano.")
                else:
                    st.caption(f"🎵 {r['name']} — BPM non rilevato — {r['key_nome']} ({r['camelot']})")
            with col_bpm:
                nuovo_bpm = st.number_input(
                    "BPM", min_value=0.0, max_value=300.0,
                    value=float(r['tempo']) if r['tempo'] > 0 else 120.0,
                    step=0.1, key=f"batch_bpm_override_{i}", label_visibility="collapsed"
                )
                st.session_state.batch_results[i]['tempo'] = nuovo_bpm

        tempi_validi = [r['tempo'] for r in st.session_state.batch_results if r['tempo'] > 0]
        if not tempi_validi:
            st.warning("Nessuna traccia ha un BPM rilevato: impossibile calcolare una velocità comune.")
        else:
            mediana_bpm = float(np.median(tempi_validi))
            target_bpm = st.number_input(
                "🎯 BPM target per tutte le tracce:", min_value=20.0, max_value=300.0,
                value=round(mediana_bpm, 1), step=1.0,
                help=f"Precompilato con la mediana del gruppo ({mediana_bpm:.1f} BPM), modificabile.",
                key="batch_target_bpm"
            )

            if st.button("🚀 Allinea tutte e genera ZIP", key="batch_align_btn"):
                zip_buffer = BytesIO()
                saltate = []
                mancanti = []
                # Rileggo i brani dal file_uploader (ancora in sessione): uno alla volta, cosi' il
                # picco di RAM e' quello di UNA traccia e non di tutte insieme.
                files_by_name = {f.name: f for f in (batch_files or [])}
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for r in st.session_state.batch_results:
                        if r['tempo'] <= 0:
                            saltate.append(r['name'])
                            continue  # nessun BPM di riferimento: non c'e' un rate calcolabile
                        sorgente = files_by_name.get(r['name'])
                        if sorgente is None:
                            mancanti.append(r['name'])  # file rimosso dall'uploader dopo l'analisi
                            continue
                        with st.spinner(f"Allineo {r['name']}..."):
                            y_src, sr_src = carica_audio_stereo(sorgente)
                            # Stessa preparazione (trim/LUFS) usata in analisi, cosi' il file
                            # esportato coincide con quello su cui sono stati misurati BPM e tonalità.
                            y_src = applica_preparazione(y_src, sr_src, **r['prep'])
                            rate = target_bpm / r['tempo']
                            y_stretched = time_stretch_stereo(y_src, rate)
                            del y_src
                            # boost=False: niente normalizzazione per-traccia al picco, che
                            # annullerebbe il bilanciamento LUFS tra le tracce dello ZIP.
                            mp3_buffer = export_audio(y_stretched, sr_src, boost=False)
                            del y_stretched
                        if mp3_buffer:
                            base_name = os.path.splitext(r['name'])[0]
                            zf.writestr(f"{base_name}_{target_bpm:.0f}bpm.mp3", mp3_buffer.read())

                    # Scaletta ideale per compatibilità armonica (solo tra le tracce con
                    # tonalità rilevabile e BPM valido, cosi' l'ordine ha davvero un senso)
                    tracce_ordinabili = [r for r in st.session_state.batch_results if r['tempo'] > 0]
                    ordine = suggest_camelot_order(tracce_ordinabili)
                    report_text = build_batch_report(tracce_ordinabili, ordine, target_bpm, saltate)
                    zf.writestr("report_batch_bpm_matcher.txt", report_text)
                    st.session_state.batch_report = report_text

                zip_buffer.seek(0)
                st.session_state.batch_zip = zip_buffer.getvalue()
                if saltate:
                    st.info(f"Saltate (nessun BPM rilevato): {', '.join(saltate)}")
                if mancanti:
                    st.warning(
                        f"Non allineate (file non più presente nell'uploader, ricaricalo e rianalizza): "
                        f"{', '.join(mancanti)}"
                    )

    if st.session_state.get('batch_report'):
        with st.expander("📄 Anteprima report (incluso nello ZIP)"):
            st.text(st.session_state.batch_report)

    if st.session_state.batch_zip:
        st.download_button(
            "📥 Scarica ZIP con le tracce allineate", st.session_state.batch_zip,
            "loop507_batch_allineato.zip", mime="application/zip", width="stretch"
        )

# Firma delle impostazioni di preparazione (conta solo cio' che e' davvero attivo: un target LUFS
# non cambia nulla se la normalizzazione e' spenta). Serve a segnalare i deck preparati con
# impostazioni diverse da quelle attuali (trim/LUFS/ottava cambiati dopo il caricamento).
prep_sig_now = (
    bool(correggi_ottava), tuple(bpm_range_plausibile) if correggi_ottava else None,
    bool(trim_silenzio), trim_soglia_db if trim_silenzio else None,
    bool(normalizza_loudness), target_lufs if normalizza_loudness else None,
)

deck_keys = list(st.session_state.decks.keys())
for row_idx in [0, 4]:
    cols = st.columns(4)
    for i, k in enumerate(deck_keys[row_idx: row_idx + 4]):
        with cols[i]:
            st.markdown(f"### Deck {k.upper()}")
            up = st.file_uploader(f"Carica {k.upper()}", type=["mp3", "wav"], key=f"up_{k}", label_visibility="collapsed")
            if up:
                # Stesso nome MA dimensione diversa = file diverso (prima non veniva rianalizzato)
                _d_old = st.session_state.decks[k]
                _file_cambiato = (
                    _d_old['name'] != up.name
                    or (_d_old.get('size') is not None and _d_old['size'] != getattr(up, 'size', None))
                )
                if _file_cambiato:
                    with st.spinner(f"Analizzando {k.upper()}..."):
                        y, sr, t, beats, fase_db = prepara_deck(
                            up, correggi_ottava, bpm_range_plausibile,
                            trim_silenzio, trim_soglia_db, normalizza_loudness, target_lufs
                        )
                        st.session_state.decks[k] = {
                            'y': y, 'sr': sr, 'tempo': t, 'tempo_detected': t, 'name': up.name,
                            'beats': beats, 'downbeat_phase': fase_db,
                            'size': getattr(up, 'size', None), 'prep_sig': prep_sig_now
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
                _d_now = st.session_state.decks[k]
                if _d_now['y'] is not None and _d_now.get('prep_sig') not in (None, prep_sig_now):
                    st.warning("⚠️ Trim/LUFS/correzione ottava sono cambiati dopo il caricamento: questo deck usa ancora le vecchie impostazioni.")
                    if st.button(f"🔄 Ricalcola Deck {k.upper()} con le impostazioni attuali", key=f"reprep_{k}"):
                        with st.spinner(f"Ricalcolo {k.upper()}..."):
                            y, sr, t, beats, fase_db = prepara_deck(
                                up, correggi_ottava, bpm_range_plausibile,
                                trim_silenzio, trim_soglia_db, normalizza_loudness, target_lufs
                            )
                            # Se l'utente aveva impostato un BPM manuale lo conservo; altrimenti
                            # riparto dal BPM appena rilevato (e azzero il widget per riflettere il nuovo valore).
                            _manuale = _d_now['tempo'] if abs(_d_now['tempo'] - _d_now.get('tempo_detected', 0.0)) > 1e-6 else None
                            st.session_state.decks[k].update({
                                'y': y, 'sr': sr, 'tempo': _manuale if _manuale else t, 'tempo_detected': t,
                                'beats': beats, 'downbeat_phase': fase_db, 'prep_sig': prep_sig_now
                            })
                            if not _manuale:
                                st.session_state.pop(f"bpm_manual_{k}", None)
                            st.session_state.segments = []
                            invalidate_mix()
                            st.session_state.pop(f"exported_rend_{k}", None)
                        st.rerun()
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

                    if st.button(f"🎛️ Esporta Renderizzato {k.upper()}", key=f"export_rend_{k}", width="stretch",
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
                            f"loop507_deck_{k}_renderizzato.mp3", key=f"dl_rend_{k}", width="stretch"
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
                        y, sr, t, beats, fase_db = prepara_deck(
                            up, correggi_ottava, bpm_range_plausibile,
                            trim_silenzio, trim_soglia_db, normalizza_loudness, target_lufs
                        )
                        st.session_state.decks[k]['y'] = y
                        st.session_state.decks[k]['sr'] = sr
                        st.session_state.decks[k]['beats'] = beats
                        st.session_state.decks[k]['downbeat_phase'] = fase_db
                        st.session_state.decks[k]['prep_sig'] = prep_sig_now
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
, key="w_tipo_taglio")

    seed_taglio_input = st.sidebar.number_input(
        "🎲 Seed taglio (0 = casuale ogni volta)", min_value=0, value=0, step=1,
        help="Rende riproducibili i tagli che usano la casualità (Frenesia Casuale, range di "
             "battute). Le altre modalità sono già deterministiche. Il seed usato finisce nel "
             "preset come 'seed_taglio'; il seed del MIX si imposta più in basso."
, key="w_seed_taglio_input")

    usa_downbeat = st.sidebar.checkbox(
        "🎼 Usa i downbeat (battuta 1) stimati",
        value=False,
        help="Stima, con la differenza spettrale tra beat (Davies & Plumbley, solo DSP), dove "
             "inizia davvero ogni misura (4/4). Se attivo: i tagli a tempo partono dalla battuta 1 "
             "invece che da un beat qualunque (così i tagli da 4 o 8 battute cadono sulle misure) e "
             "l'aggancio a frasi del DJ Remix parte dal downbeat del leader. È una stima euristica: "
             "dove l'affidabilità è bassa il downbeat viene scartato e si usa il primo beat."
, key="w_usa_downbeat")

    taglio_meta = {}  # metadati del taglio corrente, valorizzati nel ramo attivo qui sotto;
                      # evita di dover ri-derivare i parametri più avanti (rischio NameError
                      # ora che le modalità sono 4 e non più solo BPM/Casuale)

    if tipo_taglio == "Ritmo Musicale (BPM)":
        usa_range_battute = st.sidebar.checkbox(
            "🎲 Usa un range di battute invece di un valore fisso",
            value=False,
            help="Invece di tagliare sempre alla stessa lunghezza (es. sempre 2 battute), "
                 "ogni taglio pesca una lunghezza casuale nel range che imposti (es. tra 1 e 2 battute)."
, key="w_usa_range_battute")
        if usa_range_battute:
            beats_range = st.sidebar.slider(
                "Battute per segmento (min-max):", 0.25, 8.0, (1.0, 2.0), step=0.25
, key="w_beats_range")
            beats = None
        else:
            beats = st.sidebar.selectbox("Battute per segmento:", [0.5, 1, 2, 4, 8], index=2, key="w_beats")
            beats_range = None
        taglio_meta = {"modalita": "bpm", "beats": beats, "beats_range": list(beats_range) if beats_range else None,
                       "downbeat": usa_downbeat}
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
            cut_rng = crea_rng_taglio(seed_taglio_input) if beats_range is not None else None
            if cut_rng is None:
                st.session_state.cut_seed_used = None  # taglio a lunghezza fissa: nessuna casualità
            for k, d in active_decks.items():
                y_for_cut = d['y']
                fase_rate = 1.0  # time-stretch applicato (1.0 = nessuno): riporta il primo beat nella stessa scala
                # Applico il time-stretch solo se: l'utente lo ha attivato, esiste un BPM
                # rilevato affidabile come riferimento, e il BPM manuale è effettivamente diverso.
                if apply_stretch and d['tempo_detected'] > 1.0 and d['tempo'] > 0 \
                        and abs(d['tempo'] - d['tempo_detected']) > 0.5:
                    rate = d['tempo'] / d['tempo_detected']
                    with st.spinner(f"Time-stretch Deck {k.upper()} (rate {rate:.3f}x)..."):
                        y_for_cut = time_stretch_stereo(d['y'], rate)
                    stretched_decks += 1
                    fase_rate = rate
                # I tagli partono dal primo beat reale del deck, non dal campione 0
                fase = fase_deck(d, d['tempo'], d['sr'], fase_rate, usa_downbeat)
                if beats_range is not None:
                    segs = get_beat_segments_range(y_for_cut, d['sr'], d['tempo'], beats_range[0], beats_range[1],
                                                   start_offset=fase, rng=cut_rng)
                else:
                    segs = get_beat_segments(y_for_cut, d['sr'], d['tempo'], beats, start_offset=fase)
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
        range_sec = st.sidebar.slider("Range durata (sec):", 0.05, 2.0, (0.8, 1.2), step=0.05, key="w_range_sec")
        taglio_meta = {"modalita": "casuale", "range_sec": list(range_sec)}
        if st.sidebar.button("🌪️ Frulla Audio (Glitch)"):
            st.session_state.segments = []
            cut_rng = crea_rng_taglio(seed_taglio_input)
            for k, d in active_decks.items():
                segs = get_random_segments(d['y'], d['sr'], range_sec[0], range_sec[1], rng=cut_rng)
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
, key="w_onset_range")
        taglio_meta = {"modalita": "onset", "durata_range_sec": list(onset_range)}
        if st.sidebar.button("⚡ Slice ai Transienti"):
            st.session_state.segments = []
            st.session_state.cut_seed_used = None  # onset detection deterministica: nessun seed di taglio
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
            cut_rng = crea_rng_taglio(seed_taglio_input) if (leader_mode == "bpm" and leader_beats_range is not None) else None
            if cut_rng is None:
                st.session_state.cut_seed_used = None  # nessuna casualità nel taglio del leader
            leader_fase = (
                fase_deck(leader_d, leader_d['tempo'], leader_d['sr'], 1.0, usa_downbeat)
                if leader_mode == "bpm" else 0
            )
            with st.spinner(f"Calcolo la struttura dal Deck {leader_key.upper()}..."):
                lengths = get_leader_cut_lengths(
                    leader_d['y'], leader_d['sr'], leader_mode,
                    tempo=leader_d['tempo'], num_beats=leader_beats,
                    beats_range=leader_beats_range,
                    onset_min_dur=leader_onset_range[0] if leader_onset_range else None,
                    onset_max_dur=leader_onset_range[1] if leader_onset_range else None,
                    start_offset=leader_fase, rng=cut_rng,
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
                    fase_rate, tempo_fase = 1.0, d['tempo']  # scala/tempo con cui leggere il primo beat del deck
                    # Allineo il BPM reale del follower a quello del leader solo se: l'utente lo ha
                    # attivato, siamo in sotto-modo BPM (ha senso solo con un riferimento numerico),
                    # non è il leader stesso, e il follower ha un BPM effettivo valido da cui partire.
                    if apply_bpm_align and leader_mode == "bpm" and k != leader_key and d['tempo'] > 0:
                        rate = leader_d['tempo'] / d['tempo']
                        with st.spinner(f"Allineo BPM Deck {k.upper()} al leader (rate {rate:.3f}x)..."):
                            y_for_cut = time_stretch_stereo(d['y'], rate)
                        allineati += 1
                        fase_rate, tempo_fase = rate, leader_d['tempo']
                    # Ogni deck parte dal proprio primo beat: i tagli del leader cadono cosi' sui
                    # beat anche dei follower (solo in sotto-modo BPM; con gli onset la fase non c'entra).
                    fase = (
                        fase_deck(d, tempo_fase, d['sr'], fase_rate, usa_downbeat)
                        if leader_mode == "bpm" else 0
                    )
                    segs = apply_cut_lengths(y_for_cut, lengths, start_offset=fase)
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
, key="w_apply_dj_remix")

        dj_leader_key = None
        overlay_gain = 0.6
        num_overlay_events = 60
        dj_bpm_align = True
        dj_beatmatch = True
        dj_phrase_match = False
        dj_beats_per_phrase = 8
        dj_key_check = False
        dj_bass_swap = False
        dj_bass_cutoff = 150
        dj_hpss_mode = None
        dj_reverse_prob = 0.0
        dj_loop_roll_prob = 0.0
        dj_loop_roll_repeats = 4
        dj_echo_prob = 0.0
        dj_bass_sweep = False
        dj_tempo_ramp = False
        dj_gate_prob = 0.0
        dj_tape_stop_prob = 0.0
        dj_bitcrush_prob = 0.0
        dj_key_weight = False
        dj_key_shift = False
        dj_section_match = False
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
, key="w_overlay_gain")
            num_overlay_events = st.sidebar.slider(
                "🎯 Densità overlay (numero di frammenti sparsi):", 5, 400, 60
, key="w_num_overlay_events")
            dj_bpm_align = st.sidebar.checkbox(
                "🎚️ Allinea automaticamente il BPM di tutti i follower al leader", value=True,
                help="Attivo di default: ogni deck (tranne il leader) viene accelerato/rallentato "
                     "col time-stretch per suonare davvero allo stesso tempo del leader, prima "
                     "di essere sparso come overlay."
, key="w_dj_bpm_align")
            dj_beatmatch = st.sidebar.checkbox(
                "🎯 Aggancia i picchi ritmici (beatmatching)", value=True,
                help="Attivo di default: ogni frammento overlay non viene piazzato a un punto "
                     "casuale, ma il suo attacco più forte viene fatto combaciare esattamente "
                     "con un beat del leader — proprio come farebbe un DJ mixando ad orecchio."
, key="w_dj_beatmatch")

            st.sidebar.markdown("**🧪 Sperimentale — tecniche DJ avanzate**")

            dj_phrase_match = st.sidebar.checkbox(
                "🎼 Aggancio a frasi musicali (non al singolo beat)", value=False,
                help="I DJ non allineano solo i beat, allineano le FRASI (blocchi di 8/16 "
                     "battute: intro, verse, drop...). Se attivo, gli overlay si agganciano "
                     "solo all'inizio di una frase, non a un beat qualsiasi in mezzo. Richiede "
                     "il beatmatching attivo qui sopra."
, key="w_dj_phrase_match")
            dj_beats_per_phrase = 8
            if dj_phrase_match:
                dj_beats_per_phrase = st.sidebar.select_slider(
                    "Battute per frase:", options=[4, 8, 16, 32], value=8
, key="w_dj_beats_per_phrase")

            dj_key_check = st.sidebar.checkbox(
                "🎵 Rilevamento tonalità (Camelot) tra leader e follower", value=False,
                help="Rileva la tonalità di ogni deck (algoritmo Krumhansl-Schmuckler, come "
                     "Mixed In Key) e mostra la compatibilità armonica col leader secondo la "
                     "Camelot Wheel. Solo informativo: non altera l'audio, ti aiuta a scegliere."
, key="w_dj_key_check")

            dj_bass_swap = st.sidebar.checkbox(
                "🎚️ EQ Bass Swap (taglia i bassi degli overlay)", value=False,
                help="I DJ tagliano i bassi di una traccia quando ne sovrappongono un'altra, "
                     "per evitare che le due linee di basso 'impastino'. Se attivo, ogni "
                     "frammento overlay viene filtrato con un crossover Linkwitz-Riley (LR4, "
                     "lo standard professionale, taglio più netto di un semplice passa-alto) "
                     "prima di essere sommato, lasciando i bassi puliti al solo leader."
, key="w_dj_bass_swap")
            dj_bass_cutoff = 150
            if dj_bass_swap:
                dj_bass_cutoff = st.sidebar.slider(
                    "Frequenza di taglio (Hz):", 60, 400, 150, step=10,
                    help="Sotto questa frequenza, i bassi dell'overlay vengono rimossi."
, key="w_dj_bass_cutoff")
                dj_bass_sweep = st.sidebar.checkbox(
                    "🌗 Swap graduale (il filtro si apre nel tempo)", value=False, key="dj_bass_sweep",
                    help="Invece di tagliare i bassi per tutto il frammento, l'overlay entra con i bassi "
                         "tagliati e il filtro si apre fino al suono pieno, come un DJ che porta su il "
                         "low-EQ del deck entrante."
                )

            dj_hpss_label = st.sidebar.selectbox(
                "🥁 Isolamento HPSS overlay (stem separation 'leggera')", 
                ["Disattivato", "Solo percussivo (batteria)", "Solo armonico (melodia/basso)"],
                index=0,
                help="Come la Track Separation delle console top (Opus Quad, DDJ-FLX10...), "
                     "ma con matematica pura invece di modelli ML pesanti (stessa tecnica "
                     "HPSS che usi in VIDEODECOMPOSER). Isola solo batteria o solo melodia "
                     "da ogni frammento overlay prima di sommarlo."
, key="w_dj_hpss_label")
            dj_hpss_mode = {"Disattivato": None, "Solo percussivo (batteria)": "percussive",
                             "Solo armonico (melodia/basso)": "harmonic"}[dj_hpss_label]

            dj_reverse_fx = st.sidebar.checkbox(
                "↩️ Backspin casuale su alcuni overlay", value=False,
                help="Come il 'risucchio' che senti quando un DJ tocca il jogwheel all'indietro: "
                     "alcuni frammenti overlay vengono invertiti nel tempo prima di essere piazzati."
, key="w_dj_reverse_fx")
            dj_reverse_prob = st.sidebar.slider(
                "Probabilità backspin:", 0.0, 1.0, 0.15, step=0.05
, key="w_dj_reverse_prob") if dj_reverse_fx else 0.0

            dj_loop_roll_fx = st.sidebar.checkbox(
                "🔁 Loop Roll (stutter) su alcuni overlay", value=False,
                help="Come il pulsante Loop Roll delle console: un microframmento si ripete "
                     "a raffica invece di suonare una volta sola."
, key="w_dj_loop_roll_fx")
            dj_loop_roll_prob, dj_loop_roll_repeats = 0.0, 4
            if dj_loop_roll_fx:
                dj_loop_roll_prob = st.sidebar.slider("Probabilità loop roll:", 0.0, 1.0, 0.15, step=0.05, key="w_dj_loop_roll_prob")
                dj_loop_roll_repeats = st.sidebar.slider("Ripetizioni:", 2, 8, 4, key="w_dj_loop_roll_repeats")

            dj_echo_fx = st.sidebar.checkbox(
                "🔊 Beat FX Echo su alcuni overlay", value=False,
                help="Come il tasto Echo dei mixer da DJ: alcuni frammenti overlay vengono "
                     "sommati a copie ritardate e attenuate di sé stessi."
, key="w_dj_echo_fx")
            dj_echo_prob = st.sidebar.slider(
                "Probabilità echo:", 0.0, 1.0, 0.15, step=0.05
, key="w_dj_echo_prob") if dj_echo_fx else 0.0

            dj_tempo_ramp = st.sidebar.checkbox(
                "📈 Rampa di tempo (invece dello stretch fisso)", value=False, key="dj_tempo_ramp",
                help="Ogni overlay parte al suo tempo originale e accelera/rallenta gradualmente fino al "
                     "tempo del leader, invece di essere stirato subito al 100%. Richiede 'allinea BPM'. "
                     "Il risultato è un ingresso più 'scivolato', ma con più artefatti sui frammenti corti."
            )
            with st.sidebar.expander("🎛️ Effetti ritmici extra (probabilità per overlay)"):
                dj_gate_prob = st.slider("Gate ritmico:", 0.0, 1.0, 0.0, step=0.05, key="dj_gate_prob",
                                         help="Il frammento si apre e chiude a 1/8 di battuta (trance gate).")
                dj_tape_stop_prob = st.slider("Tape stop:", 0.0, 1.0, 0.0, step=0.05, key="dj_tape_stop_prob",
                                              help="Nella parte finale il suono rallenta e si ferma, abbassando l'intonazione.")
                dj_bitcrush_prob = st.slider("Bitcrush:", 0.0, 1.0, 0.0, step=0.05, key="dj_bitcrush_prob",
                                             help="Riduce bit e frequenza di campionamento: grana digitale cruda.")
            dj_key_weight = st.sidebar.checkbox(
                "🎼 Preferisci overlay in tonalità compatibile", value=False, key="dj_key_weight",
                help="Usa la Camelot Wheel per davvero: i deck compatibili col leader (stessa tonalità, "
                     "adiacente o relativa) vengono pescati più spesso, quelli dissonanti meno."
            )
            dj_key_shift = st.sidebar.checkbox(
                "🎹 Trasporta (±2 semitoni) i deck non compatibili", value=False, key="dj_key_shift",
                help="Se un deck è dissonante col leader, lo trasporta del minimo numero di semitoni "
                     "(massimo 2) necessario per renderlo compatibile. Costa tempo (pitch-shift) e "
                     "può lasciare artefatti sui materiali percussivi."
            )
            dj_section_match = st.sidebar.checkbox(
                "🏛️ Aggancia gli overlay ai confini di sezione del leader", value=False, key="dj_section_match",
                help="Trova (solo DSP: matrice di auto-similarità) i cambi di sezione del leader — "
                     "intro, break, drop — e piazza gli overlay su di essi e nelle battute subito dopo. "
                     "Richiede il beatmatching attivo. Le sezioni trovate compaiono sull'immagine d'onda "
                     "e come marker nel MIDI."
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
, key="w_apply_crossfade")
        crossfade_ms = st.sidebar.slider("Durata crossfade (ms)", 1, 100, 15, disabled=not apply_crossfade or apply_dj_remix, key="w_crossfade_ms") if apply_crossfade and not apply_dj_remix else 0

        apply_compressione = st.sidebar.checkbox(
            "🎚️ Compressione dinamica finale (leader+follower più bilanciati)", value=False,
            help="A differenza della normalizzazione LUFS (pareggia il volume MEDIO di un "
                 "deck intero prima di iniziare), questa comprime il MIX FINALE momento per "
                 "momento: se un overlay ha un picco locale più forte del resto, lo attenua "
                 "in quel preciso istante. Utile quando leader e follower restano sbilanciati "
                 "anche dopo la normalizzazione LUFS."
, key="w_apply_compressione")
        compressione_soglia, compressione_ratio = -15.0, 4.0
        if apply_compressione:
            compressione_soglia = st.sidebar.slider("Soglia compressione (dB):", -30.0, -5.0, -15.0, step=1.0, key="w_compressione_soglia")
            compressione_ratio = st.sidebar.slider("Rapporto (ratio):", 1.5, 8.0, 4.0, step=0.5, key="w_compressione_ratio")

        seed_input = st.sidebar.number_input(
            "🎲 Seed mix (0 = casuale ogni volta)", min_value=0, value=0, step=1,
            help="Rende riproducibile la scelta e il piazzamento dei frammenti. Il seed usato viene "
                 "salvato nel preset scaricabile (insieme al seed del taglio), così puoi ritrovare un mix riuscito per caso."
, key="w_seed_input")

        export_wav24 = st.sidebar.checkbox(
            "💽 Esporta anche WAV 24-bit (per Logic Pro)", value=False, key="export_wav24",
            help="Oltre all'MP3 320, prepara un WAV PCM 24-bit senza perdite. Con mix lunghi pesa "
                 "circa 16 MB al minuto (stereo 44.1 kHz) e resta in memoria finché non rigeneri."
        )

        modalita_anteprima = st.sidebar.checkbox(
            "⚡ Modalità Anteprima (genera solo ~15s, veloce)", value=False,
            help="Utile per sentire rapidamente l'effetto dei parametri correnti prima di "
                 "generare il mix completo, che con molti eventi/deck può richiedere più tempo. "
                 "Disattiva per generare la versione finale a durata piena."
, key="w_modalita_anteprima")

        if not apply_dj_remix:
            durata_mix = st.sidebar.number_input("Durata Mix Finale (sec):", 10, 600, 60)
        else:
            durata_mix = None  # in DJ Remix la durata è quella del leader, non impostabile

        if st.sidebar.button("🚀 GENERA MIX FINALE"):
            with st.spinner("Rimescolando il mazzo..."):
                seed_used = seed_input if seed_input > 0 else random.randint(1, 2**31 - 1)
                rng = random.Random(seed_used)  # RNG locale e seedato: rende il mix riproducibile
                ref_sr = TARGET_SR
                placements = []          # frammenti piazzati (per MIDI, stem, immagine)
                mix_leader = None
                mix_key_nome = None
                mix_sections_sec = []
                mix_downbeat_sec = None

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

                        # --- Rilevamento tonalità (Camelot): solo informativo, non altera
                        # l'audio, ti aiuta a scegliere quali deck stanno bene insieme. ---
                        camelot_info = []
                        if dj_key_check or dj_key_weight or dj_key_shift:
                            with st.spinner("Rilevo le tonalità (Camelot)..."):
                                leader_nome, leader_camelot, leader_conf = detect_key_camelot(dj_leader_d['y'], ref_sr)
                                mix_key_nome = leader_nome
                                follower_decks_coinvolti = sorted({s['deck'] for s in overlay_pool_raw})
                                for fk in follower_decks_coinvolti:
                                    # NON uso active_decks[fk]: un deck può aver contribuito segmenti
                                    # al pool e poi essere stato "liberato" dalla RAM (bottone Libera
                                    # RAM) — in quel caso non è più in active_decks e l'indicizzazione
                                    # diretta andrebbe in KeyError. Controllo esplicitamente.
                                    fdeck = st.session_state.decks.get(fk)
                                    if not fdeck or fdeck.get('y') is None:
                                        camelot_info.append((fk, "RAM liberata", "?", None))
                                        continue
                                    f_nome, f_camelot, f_conf = detect_key_camelot(fdeck['y'], fdeck['sr'])
                                    compat = camelot_compatibility(leader_camelot, f_camelot)
                                    camelot_info.append((fk, f_nome, f_camelot, compat))
                            if dj_key_check and leader_camelot:
                                st.sidebar.info(f"🎵 Leader {dj_leader_key.upper()}: {leader_nome} ({leader_camelot})")
                                emoji_compat = {"identica": "🟢", "adiacente": "🟢", "relativa": "🟡", "dissonante": "🔴", None: "⚪"}
                                for fk, f_nome, f_camelot, compat in camelot_info:
                                    label = compat if compat else "non rilevabile"
                                    st.sidebar.caption(f"{emoji_compat.get(compat,'⚪')} Deck {fk.upper()}: {f_nome} ({f_camelot}) — {label}")

                        # --- Auto-allineamento BPM: ogni follower viene stretchato al tempo del
                        # leader PRIMA di essere sparso come overlay, cosi' suona davvero alla
                        # stessa velocità (non solo tagliato alla stessa lunghezza). ---
                        # Tonalità attiva: peso nel pescaggio (compatibilità) e trasposizione ai deck dissonanti.
                        deck_weights_eff = dict(deck_weights)
                        shift_per_deck = {}
                        if (dj_key_weight or dj_key_shift) and camelot_info:
                            for fk, f_nome, f_camelot, compat in camelot_info:
                                if dj_key_weight and compat:
                                    deck_weights_eff[fk] = deck_weights_eff.get(fk, 5) * peso_compatibilita(compat)
                                if dj_key_shift and compat is not None and compat not in _TONALITA_OK:
                                    s_semi = scegli_shift_tonalita(leader_camelot, f_nome)
                                    if s_semi:
                                        shift_per_deck[fk] = s_semi

                        overlay_pool = []
                        ramp_rates = {}
                        decks_allineati = set()
                        decks_non_allineabili = set()
                        pool_per_deck = {}
                        for s in overlay_pool_raw:
                            pool_per_deck.setdefault(s['deck'], []).append(s)
                        for dk, segs_dk in pool_per_deck.items():
                            audios_dk = [s['audio'] for s in segs_dk]
                            if dj_bpm_align and dj_leader_d['tempo'] > 0:
                                # Il tempo del follower resta nel dict del deck anche se la RAM e' stata
                                # liberata (active_decks lo escluderebbe e lo farebbe sembrare "senza BPM").
                                follower_tempo = st.session_state.decks.get(dk, {}).get('tempo', 0)
                                if follower_tempo > 0 and abs(dj_leader_d['tempo'] - follower_tempo) > 0.5:
                                    rate = dj_leader_d['tempo'] / follower_tempo
                                    if dj_tempo_ramp:
                                        ramp_rates[dk] = rate  # rampa dentro ogni frammento, al piazzamento
                                    else:
                                        # Stretch del deck a blocchi (una volta per deck), non per frammento.
                                        with st.spinner(f"Allineo BPM Deck {dk.upper()} al leader (rate {rate:.3f}x)..."):
                                            audios_dk = stretch_segments_in_blocks(audios_dk, rate, ref_sr)
                                    decks_allineati.add(dk)
                                elif follower_tempo <= 0:
                                    decks_non_allineabili.add(dk)
                            if shift_per_deck.get(dk):
                                with st.spinner(f"Trasporto Deck {dk.upper()} di {shift_per_deck[dk]:+d} semitoni..."):
                                    audios_dk = shift_segments_in_blocks(audios_dk, shift_per_deck[dk], ref_sr)
                            for a in audios_dk:
                                overlay_pool.append({'audio': a, 'sr': ref_sr, 'deck': dk,
                                                     'ramp_rate': ramp_rates.get(dk)})

                        # --- Pre-calcolo dell'attacco più forte di ogni segmento, una volta sola,
                        # cosi' il beatmatching non deve rifare la detection ad ogni piazzamento
                        # anche se lo stesso frammento viene ripescato più volte. ---
                        if dj_beatmatch:
                            with st.spinner("Rilevo i picchi ritmici per l'aggancio..."):
                                for s in overlay_pool:
                                    s['onset_offset'] = find_strongest_onset_offset(s['audio'], ref_sr)

                        # In modalità anteprima: uso solo i primi ~15s del leader e riduco gli
                        # eventi proporzionalmente, riusando IDENTICA la logica di generazione
                        # (build_dj_remix_overlay gestisce già correttamente buffer di qualunque
                        # lunghezza), invece di duplicare il codice per un percorso "veloce" a parte.
                        if modalita_anteprima:
                            anteprima_samples = int(15 * ref_sr)
                            leader_y_per_generazione = dj_leader_d['y'][:, :anteprima_samples]
                            frazione = leader_y_per_generazione.shape[1] / max(1, dj_leader_d['y'].shape[1])
                            num_eventi_effettivi = max(3, int(num_overlay_events * frazione))
                        else:
                            leader_y_per_generazione = dj_leader_d['y']
                            num_eventi_effettivi = num_overlay_events

                        # Sezioni del leader (solo se richieste e con beat noti)
                        section_idx = None
                        if dj_section_match and dj_leader_d.get('beats'):
                            with st.spinner("Cerco i confini di sezione del leader..."):
                                section_idx = get_sezioni_leader(dj_leader_d, ref_sr, usa_downbeat) or None
                        if dj_leader_d.get('beats'):
                            _beats_l = dj_leader_d['beats']
                            _tot_l = leader_y_per_generazione.shape[1]
                            if section_idx:
                                mix_sections_sec = [_beats_l[i] / ref_sr for i in section_idx
                                                    if i < len(_beats_l) and _beats_l[i] < _tot_l]
                            _db = primo_downbeat_deck(dj_leader_d)
                            if _db is not None and _db < _tot_l:
                                mix_downbeat_sec = _db / ref_sr
                        mix_leader = {'deck': dj_leader_key, 'audio': leader_y_per_generazione}

                        with st.spinner("Sovrappongo i frammenti (DJ Remix)..."):
                            final_y, eventi_piazzati, dj_events = build_dj_remix_overlay(
                                leader_y_per_generazione, ref_sr, overlay_pool, overlay_gain, num_eventi_effettivi,
                                rng, deck_weights_eff, beatmatch=dj_beatmatch,
                                phrase_beats=dj_beats_per_phrase if dj_phrase_match else None,
                                bass_cutoff_hz=dj_bass_cutoff if dj_bass_swap else None,
                                hpss_mode=dj_hpss_mode,
                                reverse_probability=dj_reverse_prob,
                                loop_roll_probability=dj_loop_roll_prob,
                                loop_roll_repeats=dj_loop_roll_repeats,
                                echo_probability=dj_echo_prob,
                                precomputed_beat_grid=dj_leader_d.get('beats') or None,
                                downbeat_phase=dj_leader_d.get('downbeat_phase') if usa_downbeat else None,
                                bass_sweep=bool(dj_bass_sweep and dj_bass_swap),
                                gate_probability=dj_gate_prob,
                                tape_stop_probability=dj_tape_stop_prob,
                                bitcrush_probability=dj_bitcrush_prob,
                                fx_tempo=dj_leader_d['tempo'] if dj_leader_d['tempo'] > 0 else None,
                                section_idx=section_idx,
                                placements_out=placements,
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
                    target_samples = int((15 if modalita_anteprima else durata_mix) * ref_sr)
                    crossfade_samples = int(ref_sr * crossfade_ms / 1000) if apply_crossfade else 0
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
                            # Ogni giunzione con crossfade SOVRAPPONE i due segmenti: il mix si accorcia
                            # di `overlap` campioni. Senza contarlo, con centinaia di tagli la durata
                            # reale finiva diversi secondi sotto il target (e gli eventi MIDI sfasati).
                            overlap = min(crossfade_samples, seg_len, curr_samples) if chosen else 0
                            dj_events.append((curr_samples - overlap, pick['deck']))  # posizione nel mix finale
                            placements.append({'start': curr_samples - overlap, 'length': seg_len,
                                               'deck': pick['deck'], 'audio': pick['audio'], 'gain': 1.0})
                            chosen.append(pick['audio'])
                            curr_samples += seg_len - overlap
                        attempts += 1

                    if chosen:
                        # Concatenazione lungo l'asse temporale, con crossfade opzionale per
                        # evitare i click secchi tra un segmento e il successivo.
                        final_y = concat_with_crossfade(chosen, crossfade_samples)
                        # L'ultimo segmento poteva sforare il target: taglio alla durata richiesta con
                        # una dissolvenza brevissima (5ms) per non lasciare un click in coda al file.
                        if final_y.shape[1] > target_samples > 0:
                            final_y = final_y[:, :target_samples].copy()
                            fade_n = min(int(ref_sr * 0.005), final_y.shape[1])
                            if fade_n > 1:
                                final_y[:, -fade_n:] *= np.linspace(1.0, 0.0, fade_n)
                        eventi_piazzati = len(chosen)

                if not chosen:
                    st.sidebar.error("Impossibile generare il mix: nessun segmento valido disponibile.")
                else:
                    dry_peak = float(np.max(np.abs(final_y))) if final_y.shape[1] else 1.0
                    if apply_compressione:
                        with st.spinner("Comprimo il mix finale..."):
                            final_y = apply_compressor(final_y, ref_sr, threshold=compressione_soglia, ratio=compressione_ratio)
                    out = export_audio(final_y, ref_sr)
                    wav_out = export_audio_wav24(final_y, ref_sr) if export_wav24 else None
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
                        "versione_app": APP_VERSION,
                        "seed": seed_used,
                        "seed_taglio": st.session_state.get('cut_seed_used'),
                        "modalita_taglio": tipo_taglio,
                        "parametro_taglio": taglio_meta,
                        "dj_remix": {
                            "attivo": apply_dj_remix,
                            "leader": dj_leader_key,
                            "overlay_gain": overlay_gain,
                            "num_eventi": num_overlay_events,
                            "bpm_align": dj_bpm_align,
                            "beatmatch": dj_beatmatch,
                            "phrase_match": dj_phrase_match,
                            "beats_per_phrase": dj_beats_per_phrase if dj_phrase_match else None,
                            "key_check": dj_key_check,
                            "bass_swap": dj_bass_swap,
                            "bass_cutoff_hz": dj_bass_cutoff if dj_bass_swap else None,
                            "hpss": dj_hpss_mode,
                            "backspin_prob": dj_reverse_prob,
                            "loop_roll_prob": dj_loop_roll_prob,
                            "loop_roll_repeats": dj_loop_roll_repeats if dj_loop_roll_prob > 0 else None,
                            "echo_prob": dj_echo_prob,
                            "bass_sweep": bool(dj_bass_sweep and dj_bass_swap),
                            "tempo_ramp": dj_tempo_ramp,
                            "gate_prob": dj_gate_prob,
                            "tape_stop_prob": dj_tape_stop_prob,
                            "bitcrush_prob": dj_bitcrush_prob,
                            "key_weight": dj_key_weight,
                            "key_shift": dj_key_shift,
                            "section_match": dj_section_match,
                        } if apply_dj_remix else None,
                        "crossfade_ms": crossfade_ms if (apply_crossfade and not apply_dj_remix) else 0,
                        "compressione": {
                            "attiva": apply_compressione,
                            "soglia_db": compressione_soglia if apply_compressione else None,
                            "ratio": compressione_ratio if apply_compressione else None,
                        },
                        "anteprima": modalita_anteprima,
                        "usa_downbeat": usa_downbeat,
                        "esporta_wav24": export_wav24,
                        "preparazione_audio": {
                            "correggi_ottava": correggi_ottava,
                            "range_bpm": list(bpm_range_plausibile) if correggi_ottava else None,
                            "trim_silenzio": trim_silenzio,
                            "trim_soglia_db": trim_soglia_db if trim_silenzio else None,
                            "normalizza_lufs": normalizza_loudness,
                            "target_lufs": target_lufs if normalizza_loudness else None,
                        },
                        "pesi_deck": deck_weights,
                        "durata_mix_sec": round(durata_effettiva, 2),
                        "deck": [
                            {"deck": k, "file": d['name'], "bpm_usato": d['tempo']}
                            for k, d in active_decks.items()
                        ],
                        "nota": (
                            "Per riottenere questo stesso mix: ricarica gli stessi file negli stessi "
                            "deck con la stessa preparazione audio, imposta la stessa modalità/parametro "
                            "di taglio, inserisci 'seed_taglio' nel campo Seed taglio e rifai il taglio "
                            "(se seed_taglio è null la modalità non usa casualità), poi imposta gli "
                            "stessi pesi e inserisci 'seed' nel campo Seed mix prima di generare."
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
                        sperimentale_bits_it = []
                        sperimentale_bits_en = []
                        if dj_phrase_match:
                            sperimentale_bits_it.append(f"Aggancio a frasi ({dj_beats_per_phrase} battute)")
                            sperimentale_bits_en.append(f"Phrase matching ({dj_beats_per_phrase} beats)")
                        if dj_key_check:
                            sperimentale_bits_it.append("Rilevamento tonalità Camelot")
                            sperimentale_bits_en.append("Camelot key detection")
                        if dj_bass_swap:
                            sperimentale_bits_it.append(f"EQ Bass Swap (taglio sotto {dj_bass_cutoff}Hz)")
                            sperimentale_bits_en.append(f"EQ Bass Swap (cut below {dj_bass_cutoff}Hz)")
                        if dj_hpss_mode:
                            sperimentale_bits_it.append(f"Isolamento HPSS ({dj_hpss_mode})")
                            sperimentale_bits_en.append(f"HPSS isolation ({dj_hpss_mode})")
                        if dj_reverse_prob > 0:
                            sperimentale_bits_it.append(f"Backspin {dj_reverse_prob:.0%}")
                            sperimentale_bits_en.append(f"Backspin {dj_reverse_prob:.0%}")
                        if dj_loop_roll_prob > 0:
                            sperimentale_bits_it.append(f"Loop Roll {dj_loop_roll_prob:.0%} x{dj_loop_roll_repeats}")
                            sperimentale_bits_en.append(f"Loop Roll {dj_loop_roll_prob:.0%} x{dj_loop_roll_repeats}")
                        if dj_echo_prob > 0:
                            sperimentale_bits_it.append(f"Beat FX Echo {dj_echo_prob:.0%}")
                            sperimentale_bits_en.append(f"Beat FX Echo {dj_echo_prob:.0%}")
                        sperimentale_info_it = f" / {' / '.join(sperimentale_bits_it)}" if sperimentale_bits_it else ""
                        sperimentale_info_en = f" / {' / '.join(sperimentale_bits_en)}" if sperimentale_bits_en else ""
                        processo_it = f"DJ Remix: Deck {dj_leader_key.upper()} intatto come base / {eventi_piazzati} overlay sparsi (seed {seed_used}){allineamento_info}{beatmatch_info}{sperimentale_info_it} / Stereo Preservato"
                        processo_en = f"DJ Remix: Deck {dj_leader_key.upper()} kept intact as base / {eventi_piazzati} scattered overlays (seed {seed_used}){sperimentale_info_en} / Stereo Preserved"
                        extra_it = f"* Deck Base (intatto): {dj_leader_key.upper()}\n* Overlay Piazzati: {eventi_piazzati} (volume {overlay_gain:.2f}x)\n* Allineamento BPM: {'attivo' if dj_bpm_align else 'disattivo'}\n* Beatmatching: {'attivo' if dj_beatmatch else 'disattivo'}\n* Aggancio a frasi: {f'{dj_beats_per_phrase} battute' if dj_phrase_match else 'disattivo'}\n* EQ Bass Swap: {f'sotto {dj_bass_cutoff}Hz' if dj_bass_swap else 'disattivo'}\n* Compressione Finale: {f'soglia {compressione_soglia}dB, ratio {compressione_ratio}' if apply_compressione else 'disattivata'}"
                        extra_en = f"* Base Deck (intact): {dj_leader_key.upper()}\n* Overlays Placed: {eventi_piazzati} (gain {overlay_gain:.2f}x)\n* BPM Alignment: {'on' if dj_bpm_align else 'off'}\n* Beatmatching: {'on' if dj_beatmatch else 'off'}\n* Phrase Matching: {f'{dj_beats_per_phrase} beats' if dj_phrase_match else 'off'}\n* EQ Bass Swap: {f'below {dj_bass_cutoff}Hz' if dj_bass_swap else 'off'}\n* Final Compression: {f'threshold {compressione_soglia}dB, ratio {compressione_ratio}' if apply_compressione else 'disabled'}"
                        extra_it += (
                            f"\n* HPSS: {dj_hpss_mode or 'disattivo'}"
                            f"\n* Backspin: {f'{dj_reverse_prob:.0%}' if dj_reverse_prob > 0 else 'disattivo'}"
                            f"\n* Loop Roll: {f'{dj_loop_roll_prob:.0%} x{dj_loop_roll_repeats}' if dj_loop_roll_prob > 0 else 'disattivo'}"
                            f"\n* Beat FX Echo: {f'{dj_echo_prob:.0%}' if dj_echo_prob > 0 else 'disattivo'}"
                        )
                        extra_en += (
                            f"\n* HPSS: {dj_hpss_mode or 'off'}"
                            f"\n* Backspin: {f'{dj_reverse_prob:.0%}' if dj_reverse_prob > 0 else 'off'}"
                            f"\n* Loop Roll: {f'{dj_loop_roll_prob:.0%} x{dj_loop_roll_repeats}' if dj_loop_roll_prob > 0 else 'off'}"
                            f"\n* Beat FX Echo: {f'{dj_echo_prob:.0%}' if dj_echo_prob > 0 else 'off'}"
                        )
                        _nuovi_it = [n for n, a in [("Bass swap graduale", dj_bass_sweep and dj_bass_swap), ("Rampa di tempo", dj_tempo_ramp),
                                                    ("Pesi per tonalità", dj_key_weight), ("Trasposizione ±2 semitoni", dj_key_shift),
                                                    ("Aggancio alle sezioni", dj_section_match)] if a]
                        _nuovi_en = [n for n, a in [("Gradual bass swap", dj_bass_sweep and dj_bass_swap), ("Tempo ramp", dj_tempo_ramp),
                                                    ("Key-compatibility weighting", dj_key_weight), ("Key shift ±2 semitones", dj_key_shift),
                                                    ("Section-boundary matching", dj_section_match)] if a]
                        extra_it += (
                            f"\n* Gate: {dj_gate_prob:.0%} | Tape stop: {dj_tape_stop_prob:.0%} | Bitcrush: {dj_bitcrush_prob:.0%}"
                            f"\n* Opzioni DJ: {', '.join(_nuovi_it) if _nuovi_it else 'nessuna extra'}"
                        )
                        extra_en += (
                            f"\n* Gate: {dj_gate_prob:.0%} | Tape stop: {dj_tape_stop_prob:.0%} | Bitcrush: {dj_bitcrush_prob:.0%}"
                            f"\n* DJ options: {', '.join(_nuovi_en) if _nuovi_en else 'no extras'}"
                        )
                    else:
                        processo_it = f"Shuffling Ricorsivo (seed {seed_used}) / Cross-Deck Fragmentation / Sample Rate Uniformato / Stereo Preservato / Pesi Deck Personalizzati"
                        processo_en = f"Recursive Shuffling (seed {seed_used}) / Cross-Deck Fragmentation / Uniform Sample Rate / Stereo Preserved / Custom Deck Weights"
                        extra_it = f"* Crossfade: {f'{crossfade_ms}ms' if apply_crossfade else 'disattivato'}\n* Compressione Finale: {f'soglia {compressione_soglia}dB, ratio {compressione_ratio}' if apply_compressione else 'disattivata'}"
                        extra_en = f"* Crossfade: {f'{crossfade_ms}ms' if apply_crossfade else 'disabled'}\n* Final Compression: {f'threshold {compressione_soglia}dB, ratio {compressione_ratio}' if apply_compressione else 'disabled'}"

                    cut_seed_txt = st.session_state.get('cut_seed_used') or "—"
                    st.session_state.audio_report = f"""
╔════════════════════════════════════════════════════════════════╗
  HYPER-MIXER v{APP_VERSION} - AUDIO RECONSTRUCTION LOG (STEREO + DJ REMIX + MIDI)
  Generated on: {ts_audio}
╚════════════════════════════════════════════════════════════════╝

[AUDIO_RECONSTRUCTION_LOG] // VOL_01

═══════════════════ ITALIANO ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v{APP_VERSION}]
:: ANALISI: Beat Tracking / RMS Envelope / Onset Detection
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
* Seed Taglio: {cut_seed_txt}
* Durata Output: {durata_effettiva:.1f}s

═══════════════════ ENGLISH ═══════════════════

:: ENGINE: hyper_mixer_loop507 [v{APP_VERSION}]
:: ANALYSIS: Beat Tracking / RMS Envelope / Onset Detection
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
* Cut Seed: {cut_seed_txt}
* Output Duration: {durata_effettiva:.1f}s

> Regia e Algoritmo / Direction & Algorithm: Loop507

#AudioGlitch #SoundDesign #GranularSynthesis #ExperimentalMusic
#AudioDecomposition #NoiseArt #SignalCorruption #RecursiveCollapse
"""
                    st.session_state.mix_ready = out
                    st.session_state.mix_is_preview = modalita_anteprima
                    # Dati per l'export per Logic (MIDI con durate reali, stem, WAV 24-bit)
                    st.session_state.mix_wav24 = wav_out
                    st.session_state.mix_placements = placements
                    st.session_state.mix_leader = mix_leader
                    st.session_state.mix_total_samples = int(final_y.shape[1])
                    st.session_state.mix_dry_peak = dry_peak
                    st.session_state.mix_deck_names = {k: (v.get('name') or '') for k, v in st.session_state.decks.items()}
                    st.session_state.mix_key_nome = mix_key_nome
                    st.session_state.mix_sections_sec = mix_sections_sec
                    st.session_state.mix_downbeat_sec = mix_downbeat_sec
                    st.session_state.pop('mix_stems_zip', None)
                    st.session_state.pop('mix_midi_trascritto', None)
                    st.session_state.pop('mix_midi_conteggi', None)
                    # True se il BPM del mix e' noto (leader DJ): altrimenti la trascrizione lo stima dall'audio
                    st.session_state.mix_tempo_noto = bool(apply_dj_remix and dj_leader_d['tempo'] > 0)

                    # --- Immagine forma d'onda: 'foto' di cosa è successo nel mix ---
                    titolo_img = f"Loop507 {'DJ Remix' if apply_dj_remix else 'Hyper-Mixer'}" + (" — ANTEPRIMA" if modalita_anteprima else "")
                    _bar_times = None
                    if apply_dj_remix and dj_leader_d.get('beats'):
                        _fase0 = dj_leader_d.get('downbeat_phase') if dj_leader_d.get('downbeat_phase') is not None else 0
                        _bar_times = [b / ref_sr for b in dj_leader_d['beats'][_fase0::4] if b < final_y.shape[1]]
                    st.session_state.mix_waveform_image = build_waveform_image(
                        final_y, ref_sr, dj_events, title=titolo_img, placements=placements,
                        bar_times=_bar_times, section_times=mix_sections_sec)
                    # Dati per il video onda (generato a richiesta dopo il mix)
                    st.session_state.mix_bar_times = _bar_times
                    st.session_state.mix_img_title = titolo_img
                    st.session_state.pop('mix_waveform_video', None)
                    st.session_state.pop('mix_waveform_video_info', None)

if st.session_state.get('mix_ready'):
    st.divider()
    st.subheader("🎵 Risultato del Mix")
    if st.session_state.get('mix_is_preview'):
        st.warning("⚡ Questa è un'ANTEPRIMA veloce (~15s) — disattiva 'Modalità Anteprima' e rigenera per il mix completo.")
    st.audio(st.session_state.mix_ready)
    if st.session_state.get('mix_waveform_image'):
        st.image(st.session_state.mix_waveform_image, width="stretch")
    # --- VIDEO dell'onda: la stessa immagine, animata e a tempo con l'audio ---
    with st.expander("🎬 Video dell'onda (come l'immagine, animato e a tempo con l'audio)", expanded=False):
        if not CV2_DISPONIBILE:
            st.caption("🎬 Video non disponibile: aggiungi 'opencv-python-headless' a requirements.txt.")
        else:
            st.caption(
                "In alto la panoramica di tutto il mix con il cursore che avanza; in basso la finestra che scorre "
                "(±4 s) con le fasce colorate dei deck, le battute del leader e le sezioni; il cursore pulsa a ogni "
                "battuta. Audio = l'MP3 del mix. Per provare lo stile usa 'Anteprima (30 s)': il rendering è "
                "più veloce della durata del video, ma su Streamlit Cloud un mix lungo in HD può richiedere minuti."
            )
            _v1, _v2, _v3 = st.columns(3)
            with _v1:
                _ris = st.selectbox("Risoluzione", ["854x480", "960x540", "1280x720"], index=1, key="vid_ris")
            with _v2:
                _fps = st.selectbox("Fotogrammi/s", [24, 30], index=0, key="vid_fps")
            with _v3:
                _durv = st.selectbox("Durata", ["Anteprima (30 s)", "Primi 90 s", "Intero mix"], index=0, key="vid_dur")
            if st.button("🎬 Genera video onda", width="stretch", key="vid_btn"):
                _w, _h = (int(v) for v in _ris.split("x"))
                _max = {"Anteprima (30 s)": 30, "Primi 90 s": 90}.get(_durv)
                _barra = st.progress(0.0, text="Renderizzo i fotogrammi...")
                _dati_v, _msg_v = build_waveform_video(
                    st.session_state.mix_ready.getvalue(), st.session_state.get('mix_events_sr', TARGET_SR),
                    placements=st.session_state.get('mix_placements') or None,
                    events=st.session_state.get('mix_events') or None,
                    title=st.session_state.get('mix_img_title', "Loop507 Mix"),
                    bar_times=st.session_state.get('mix_bar_times'),
                    section_times=st.session_state.get('mix_sections_sec'),
                    larghezza=_w, altezza=_h, fps=_fps, max_sec=_max,
                    progress_cb=lambda f: _barra.progress(float(f), text=f"Renderizzo i fotogrammi... {int(f * 100)}%"))
                _barra.empty()
                if _dati_v:
                    st.session_state.mix_waveform_video = _dati_v
                    st.session_state.mix_waveform_video_info = _msg_v
                else:
                    st.session_state.pop('mix_waveform_video', None)
                    st.error(f"🎬 {_msg_v}")
            if st.session_state.get('mix_waveform_video'):
                st.video(st.session_state.mix_waveform_video, format="video/mp4")
                st.caption(st.session_state.get('mix_waveform_video_info', ""))
                st.download_button("🎬 Scarica Video Onda (MP4)", st.session_state.mix_waveform_video,
                                   f"{NOME_BASE_OUT}.mp4", mime="video/mp4", width="stretch")

    # --- MIDI della MUSICA del mix (batteria + basso + armonia), trascritto dall'audio ---
    if st.session_state.get('mix_ready') is not None:
        with st.expander("🥁 MIDI della musica del mix (batteria + basso + armonia)", expanded=True):
            st.caption(
                "Trascrive dal SUONO del mix finale (non dai deck): batteria su canale 10 (kick, rullante, "
                "hi-hat chiuso/aperto), basso monofonico e armonia/lead. È una trascrizione automatica solo DSP: "
                "la batteria è affidabile su ritmi puliti, il basso è buono, l'armonia è una bozza da correggere nel DAW."
            )
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _sens = st.slider("Sensibilità batteria", 1, 10, 5, key="trasc_sens",
                                  help="Più alta = più colpi rilevati (ma anche più falsi positivi).")
            with _c2:
                _con_basso = st.checkbox("Includi basso", value=True, key="trasc_basso")
                _con_arm = st.checkbox("Includi armonia/lead", value=True, key="trasc_arm")
            with _c3:
                _quant = st.checkbox("Quantizza a 1/16", value=False, key="trasc_quant",
                                     help="Aggancia gli attacchi alla griglia del tempo (se entro il 40% del passo). "
                                          "Utile se il BPM è giusto; altrimenti lascialo spento e quantizza in Logic.")
            if st.session_state.get('mix_midi_trascritto') is None:
                if st.button("🎼 Genera MIDI dal suono del mix", width="stretch", key="trasc_btn"):
                    with st.spinner("Trascrivo batteria, basso e armonia dal mix..."):
                        _seg = AudioSegment.from_file(BytesIO(st.session_state.mix_ready.getvalue()))
                        _seg = _seg.set_frame_rate(SR_TRASCRIZIONE).set_channels(1)
                        _y = np.array(_seg.get_array_of_samples()).astype(np.float32) / float(2 ** (8 * _seg.sample_width - 1))
                        _bpm = st.session_state.get('mix_events_tempo', 120.0)
                        if not st.session_state.get('mix_tempo_noto'):
                            _stima = ioi_tempo_estimate(_y, SR_TRASCRIZIONE)
                            if _stima:
                                _bpm = correggi_errore_ottava(_stima, 70, 180)
                        _midi, _cont = trascrivi_mix(
                            _y, SR_TRASCRIZIONE, _bpm, key_nome=st.session_state.get('mix_key_nome'),
                            markers=[(t, f"Sezione {i + 1}") for i, t in enumerate(st.session_state.get('mix_sections_sec') or [])],
                            downbeat_sec=st.session_state.get('mix_downbeat_sec'), sensibilita=_sens,
                            con_basso=_con_basso, con_armonia=_con_arm, quantize=_quant,
                            nota_info=f"Loop507 Hyper-Mixer v{APP_VERSION} - trascrizione dal mix - {_bpm:.2f} BPM")
                        _cont['bpm'] = _bpm
                        st.session_state.mix_midi_trascritto = _midi
                        st.session_state.mix_midi_conteggi = _cont
                    st.rerun()
            else:
                _cont = st.session_state.get('mix_midi_conteggi') or {}
                st.download_button("🎹 Scarica MIDI del mix (batteria + basso + armonia)",
                                   st.session_state.mix_midi_trascritto, "loop507_mix_trascritto.mid",
                                   mime="audio/midi", width="stretch")
                st.caption(f"Trascritti: {_cont.get('kick', 0)} kick, {_cont.get('rullante', 0)} rullanti, "
                           f"{_cont.get('hihat', 0)} hi-hat, {_cont.get('basso', 0)} note di basso, "
                           f"{_cont.get('armonia', 0)} note di armonia · tempo {_cont.get('bpm', 0):.1f} BPM")
                if st.button("↩️ Cambia opzioni e rigenera", key="trasc_reset"):
                    st.session_state.pop('mix_midi_trascritto', None)
                    st.session_state.pop('mix_midi_conteggi', None)
                    st.rerun()

    # --- Export per Logic Pro: WAV 24-bit e stem allineati ---
    _pl = st.session_state.get('mix_placements') or []
    _ld = st.session_state.get('mix_leader')
    if st.session_state.get('mix_wav24') is not None or _pl or _ld:
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if st.session_state.get('mix_wav24') is not None:
                st.download_button("💽 Scarica Mix WAV 24-bit", st.session_state.mix_wav24,
                                   "loop507_custom_mix_24bit.wav", mime="audio/wav", width="stretch")
            else:
                st.caption("💽 WAV 24-bit: attiva «Esporta anche WAV 24-bit» nella barra laterale e rigenera.")
        with col_w2:
            _n_stem = len({p['deck'] for p in _pl} | ({_ld['deck']} if _ld else set()))
            _mb = _n_stem * st.session_state.get('mix_total_samples', 0) / TARGET_SR * TARGET_SR * 2 * 3 / 1e6
            if st.session_state.get('mix_stems_zip') is None:
                if st.button(f"🎛️ Prepara stem per deck ({_n_stem} file WAV 24-bit, ~{_mb:.0f} MB)", width="stretch",
                             help="Un WAV per deck, tutti lunghi quanto il mix e allineati allo stesso campione 0: "
                                  "trascinali in Logic sulla stessa posizione e si sommano al mix. Sono 'dry' (senza "
                                  "compressione). Con mix lunghi pesano molto: usa l'anteprima per provare."):
                    with st.spinner("Renderizzo gli stem (un deck alla volta)..."):
                        _bpm_info = (st.session_state.get('mix_events_tempo') or 120.0)
                        st.session_state.mix_stems_zip = render_stems_zip(
                            _pl, _ld, st.session_state.get('mix_total_samples', 0), TARGET_SR,
                            st.session_state.get('mix_dry_peak', 1.0), st.session_state.get('mix_deck_names', {}),
                            info_txt=(f"Loop507 Hyper-Mixer v{APP_VERSION}\nStem WAV 24-bit allineati (stesso campione 0), "
                                      f"{TARGET_SR} Hz stereo.\nBPM di riferimento: {_bpm_info:.2f}\n"
                                      "Dry: senza compressione né normalizzazione finale."))
                    st.rerun()
            else:
                st.download_button("📦 Scarica Stem (ZIP)", st.session_state.mix_stems_zip,
                                   "loop507_stem.zip", mime="application/zip", width="stretch")

    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    with col_d1:
        st.download_button("📥 Scarica Mix MP3", st.session_state.mix_ready, f"{NOME_BASE_OUT}.mp3", width="stretch")
    with col_d2:
        st.download_button("📄 Scarica Report Audio", st.session_state.audio_report, f"{NOME_BASE_OUT}.txt", width="stretch")
    with col_d3:
        st.download_button(
            "🎲 Scarica Preset (seed)", st.session_state.get('mix_preset', '{}'),
            "loop507_preset.json", width="stretch",
            help="Contiene il seed e i parametri usati: ricaricalo per riprovare a ritrovare questo mix."
        )
    with col_d4:
        if MIDI_DISPONIBILE:
            _tempo_midi = st.session_state.get('mix_events_tempo', 120.0)
            if _pl or _ld:
                midi_buffer = build_logic_midi(
                    _pl, st.session_state.get('mix_events_sr', TARGET_SR),
                    st.session_state.get('mix_total_samples', 0), _tempo_midi,
                    st.session_state.get('mix_deck_names', {}), leader=_ld,
                    key_nome=st.session_state.get('mix_key_nome'),
                    markers=[(t, f"Sezione {i + 1}") for i, t in enumerate(st.session_state.get('mix_sections_sec') or [])],
                    downbeat_sec=st.session_state.get('mix_downbeat_sec'),
                    nota_info=f"Loop507 Hyper-Mixer v{APP_VERSION} - {_tempo_midi:.2f} BPM")
            else:
                midi_buffer = build_structure_midi(
                    st.session_state.get('mix_events', []),
                    st.session_state.get('mix_events_sr', TARGET_SR), _tempo_midi)
            if midi_buffer:
                st.download_button(
                    "🎹 Scarica MIDI struttura (1 traccia per deck)", midi_buffer, "loop507_struttura.mid",
                    width="stretch",
                    help="NON è la musica del mix: è la struttura (una nota lunga quanto ogni frammento, una traccia per deck). "
                         "Per batteria/basso/armonia del suono finale usa il MIDI trascritto qui sopra. "
                         "MIDI Format 1: traccia conductor (tempo, 4/4, tonalità, marker di sezione e di "
                         "battuta 1), una traccia per deck con nome e note lunghe quanto ogni frammento "
                         "(A=C1, B=C#1...). Importa il WAV/MP3 a battuta 1 e imposta il tempo del progetto "
                         "al BPM indicato; il marker 'Battuta 1' dice dove cade il primo downbeat del leader."
                )
            else:
                st.caption("🎹 Nessuna struttura MIDI disponibile per questo mix.")
        else:
            st.caption("🎹 Export MIDI non disponibile: aggiungi 'mido' a requirements.txt.")
    with col_d5:
        if st.session_state.get('mix_waveform_image'):
            st.download_button(
                "🖼️ Scarica Immagine Onda", st.session_state.mix_waveform_image,
                "loop507_waveform.png", mime="image/png", width="stretch",
                help="Forma d'onda del mix con marcatori colorati per deck nei punti di taglio/overlay."
            )
        else:
            st.caption("🖼️ Immagine onda non disponibile: aggiungi 'matplotlib' a requirements.txt.")

st.markdown("---")
st.caption(f"Loop507 Hyper-Mixer | Modalità Glitch & BPM attiva | Stereo + DJ Remix + Preparazione Audio + Waveform v{APP_VERSION}")
