"""Core protocol for the Oruk speech-emotion benchmark.

Everything an entrant is scored with lives here: the 7-label space, the label
normalizer that maps every public model's taxonomy into that space, the eval
data loader, the audio clipping rule (16 kHz mono, 16 s truncation), the
stratified subsampler used for API-priced models, and the single scoring
implementation shared by every arm of the benchmark.

Eval shards are parquet files with columns:
  audio_flac : bytes  (FLAC-encoded audio)
  label      : int    (index into LABELS)
  language   : str    (BCP-47-ish tag or name; may be null -> "unknown")
  source_id  : str    (corpus/source identifier)

The shards themselves are not distributed with this repository; see
BENCHMARK_CARD.md for access details.
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ["anger", "happiness", "sadness", "fear", "disgust", "surprise", "neutral"]
TARGET_SR = 16000
MAX_SECONDS = 16.0

# Normalizes every label spelling seen across public SER models to our space.
#
# "positive" and "enthusiasm" are the only entries that widen a source taxonomy
# rather than respell ours, so both are justified against PROMPT below, which
# defines happiness as "joy, amusement, enthusiasm, warm or smiling voice":
#   - "enthusiasm" (Aniemore RESD) is named verbatim in that definition.
#   - "positive" (DUSHA) is the sole non-neutral positive class in a taxonomy
#     whose other classes are neutral/angry/sad/other, so it carries exactly
#     the happiness mass and nothing else.
LABEL_ALIASES = {
    "anger": "anger", "angry": "anger", "ang": "anger",
    "happiness": "happiness", "happy": "happiness", "hap": "happiness", "joy": "happiness",
    "enthusiasm": "happiness", "positive": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "fear": "fear", "fearful": "fear", "fea": "fear",
    "disgust": "disgust", "disgusted": "disgust", "dis": "disgust",
    "surprise": "surprise", "surprised": "surprise", "sur": "surprise",
    "neutral": "neutral", "neu": "neutral", "calm": "neutral",
    "suprised": "surprise",  # Hatman/audio-emotion-detection config typo
}

# Shared best-chance prompt for every prompted model (Gemini, OpenAI audio,
# open-weight audio-LLMs). Published so entrants can audit exactly what the
# closed models were asked.
PROMPT = """You are an expert speech-emotion annotator working on a multilingual \
corpus. Listen to the audio clip and decide which single emotion best describes \
how the SPEAKER sounds.

Choose exactly one of:
- anger: irritation, hostility, raised or tense voice, sharp attacks
- happiness: joy, amusement, enthusiasm, warm or smiling voice
- sadness: sorrow, dejection, low energy, flat or trembling voice
- fear: anxiety, panic, tense or shaky voice
- disgust: revulsion, contempt, disdainful tone
- surprise: astonishment, sudden rising intonation, exclamations
- neutral: calm, matter-of-fact, no marked emotion

Rules:
- The clip may be in ANY language. Judge prosody (pitch, tempo, loudness, voice \
quality) AND the meaning of the words when you understand the language.
- Acted, elicited and spontaneous speech all appear; judge what you hear.
- If several emotions are present, pick the dominant one.
- Only pick neutral when no other emotion is clearly expressed.

Answer with the single best label."""


def norm_label(raw):
    """Map any label spelling (or tag like ``<|HAPPY|>``) to our space, or None."""
    key = str(raw).split("/")[-1].strip().lower().replace("<|", "").replace("|>", "")
    return LABEL_ALIASES.get(key)


# --------------------------------------------------------------------------- data


def load_eval(shard_dir):
    """Load all parquet shards in deterministic (sorted-filename) order.

    Returns (tables, index, labels, langs, sources) where ``index`` maps a flat
    example number to (table_idx, row) and the three arrays are aligned to it.
    """
    tables, index = [], []
    for path in sorted(Path(shard_dir).glob("*.parquet")):
        try:
            t = pq.read_table(path)
        except Exception as e:
            print(f"[warn] skip {path.name}: {e}", flush=True)
            continue
        if t.num_rows == 0:
            continue
        ti = len(tables)
        tables.append(t)
        index.extend((ti, r) for r in range(t.num_rows))
    if not index:
        raise FileNotFoundError(f"no readable parquet shards in {shard_dir}")
    labels = np.array([tables[t]["label"][r].as_py() for t, r in index], dtype=np.int64)
    langs = np.array(
        [tables[t]["language"][r].as_py() or "unknown" for t, r in index], dtype=object
    )
    sources = np.array([tables[t]["source_id"][r].as_py() for t, r in index], dtype=object)
    return tables, index, labels, langs, sources


def clip_audio(tables, index, i):
    """Decode example ``i`` to 16 kHz mono float32, truncated to MAX_SECONDS."""
    t, r = index[i]
    y, _ = sf.read(io.BytesIO(tables[t]["audio_flac"][r].as_py()), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y[: int(MAX_SECONDS * TARGET_SR)])


def wav_bytes(audio):
    """Encode a float32 waveform as 16-bit PCM WAV bytes (for API arms)."""
    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def stratified_subsample(labels, n_total, seed=0):
    """Fixed label-stratified subsample used for API-priced models (seed 0)."""
    rng = np.random.default_rng(seed)
    idx_all = []
    n = len(labels)
    for c in range(len(LABELS)):
        c_idx = np.where(labels == c)[0]
        take = max(1, int(round(n_total * len(c_idx) / n)))
        idx_all.append(rng.choice(c_idx, size=min(take, len(c_idx)), replace=False))
    return np.sort(np.concatenate(idx_all))


# --------------------------------------------------------------------------- scoring


def score(y, preds, langs, sources, supported):
    """Single scoring implementation for every entrant.

    Full-set metrics over all 7 classes, plus fair-subset metrics restricted to
    clips whose gold label is in the model's ``supported`` set (so e.g. 4-class
    IEMOCAP models get an honest subset score next to their penalized full-set
    score). Per-language breakdowns are reported for languages with >=100 clips.
    """
    y = np.asarray(y)
    preds = np.asarray(preds)
    langs = np.asarray(langs, dtype=object)
    sources = np.asarray(sources, dtype=object)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, preds, labels=range(len(LABELS)), zero_division=0
    )
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "per_class": {
            LABELS[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                        "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(LABELS))
        },
        "per_language": {},
        "per_source": {},
    }
    sup_idx = [LABELS.index(s) for s in supported]
    if len(sup_idx) < len(LABELS):
        m = np.isin(y, sup_idx)
        out["supported_labels"] = list(supported)
        out["subset_n"] = int(m.sum())
        out["subset_macro_f1"] = float(
            f1_score(y[m], preds[m], labels=sup_idx, average="macro", zero_division=0)
        )
        out["subset_accuracy"] = float(accuracy_score(y[m], preds[m]))
    for lang in sorted(set(langs)):
        m = langs == lang
        if m.sum() < 100:
            continue
        out["per_language"][str(lang)] = {
            "n": int(m.sum()),
            "macro_f1": float(f1_score(y[m], preds[m], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y[m], preds[m])),
        }
    for src in sorted(set(sources)):
        m = sources == src
        out["per_source"][str(src)] = {
            "n": int(m.sum()),
            "macro_f1": float(f1_score(y[m], preds[m], average="macro", zero_division=0)),
        }
    return out
