"""Open-source SER arm: adapters for HF / FunASR / SpeechBrain checkpoints.

Identical protocol for every entrant: same deterministic shard order, same 16 s
truncation, same 7-label space, same scoring code (see ``oruk_bench.core``).

Every adapter returns predictions as int indices into LABELS (argmax over the
model's probability mass on our 7 classes; unmapped label mass is ignored).

Requires the ``[open]`` extra: torch, transformers, funasr, speechbrain.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..core import LABELS, TARGET_SR, clip_audio, load_eval, norm_label, score


class FunasrAdapter:
    """emotion2vec family via funasr AutoModel.generate."""

    def __init__(self, cfg, device):
        from funasr import AutoModel

        self.model = AutoModel(model=cfg["model_id"], hub=cfg.get("hub", "hf"),
                               device=device, disable_update=True, log_level="ERROR")

    def predict(self, audios):
        res = self.model.generate(input=list(audios), fs=TARGET_SR,
                                  granularity="utterance", extract_embedding=False,
                                  disable_pbar=True)
        preds = []
        for r in res:
            vec = np.full(len(LABELS), -1.0)
            for lab, sc in zip(r["labels"], r["scores"]):
                ours = norm_label(lab)
                if ours is not None:
                    vec[LABELS.index(ours)] = max(vec[LABELS.index(ours)], float(sc))
            preds.append(int(np.argmax(vec)))
        return preds


class SenseVoiceAdapter:
    """SenseVoice via funasr; emotion arrives as an <|EMO|> tag in the text."""

    def __init__(self, cfg, device):
        from funasr import AutoModel

        self.model = AutoModel(model=cfg["model_id"], hub=cfg.get("hub", "hf"),
                               device=device, disable_update=True, log_level="ERROR",
                               trust_remote_code=False)

    def predict(self, audios):
        res = self.model.generate(input=list(audios), fs=TARGET_SR, disable_pbar=True,
                                  language="auto", use_itn=False, ban_emo_unk=True)
        preds = []
        for r in res:
            text = r.get("text", "")
            found = None
            for tag in ("HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"):
                if f"<|{tag}|>" in text:
                    found = norm_label(tag)
                    break
            preds.append(LABELS.index(found) if found else LABELS.index("neutral"))
        return preds


class HFAudioClsAdapter:
    """Any AutoModelForAudioClassification checkpoint."""

    def __init__(self, cfg, device):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(cfg["model_id"])
        # Whisper encoders require fixed 3000-frame mels; others pad to batch max.
        self.pad_kwargs = (
            {"padding": "max_length"}
            if "whisper" in type(self.fe).__name__.lower()
            else {"padding": True}
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", False)
        )
        self.model = self.model.float().to(device).eval()
        # map model output indices -> our label indices (or None)
        self.out_map = []
        for i in range(self.model.config.num_labels):
            raw = self.model.config.id2label.get(i, str(i))
            ours = norm_label(raw)
            self.out_map.append(LABELS.index(ours) if ours else None)
        mapped = [LABELS[m] for m in self.out_map if m is not None]
        print(f"  id2label -> {dict(self.model.config.id2label)} | mapped: {mapped}", flush=True)

    def predict(self, audios):
        feats = self.fe(list(audios), sampling_rate=TARGET_SR, return_tensors="pt",
                        **self.pad_kwargs)
        key = "input_values" if "input_values" in feats else "input_features"
        kwargs = {}
        if "attention_mask" in feats:
            kwargs["attention_mask"] = feats["attention_mask"].to(self.device)
        with self.torch.no_grad():
            logits = self.model(feats[key].to(self.device), **kwargs).logits.float()
        probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        preds = []
        for p in probs:
            vec = np.full(len(LABELS), -1.0)
            for j, m in enumerate(self.out_map):
                if m is not None:
                    vec[m] = max(vec[m], float(p[j]))
            preds.append(int(np.argmax(vec)))
        return preds


class SpeechBrainAdapter:
    """speechbrain.inference.interfaces.foreign_class SER checkpoints."""

    def __init__(self, cfg, device):
        import torch
        from speechbrain.inference.interfaces import foreign_class

        self.torch = torch
        self.device = device
        self.clf = foreign_class(
            source=cfg["model_id"],
            pymodule_file=cfg.get("pymodule_file", "custom_interface.py"),
            classname=cfg.get("classname", "CustomEncoderWav2vec2Classifier"),
            run_opts={"device": device},
        )

    def predict(self, audios):
        max_len = max(len(a) for a in audios)
        wav = self.torch.zeros(len(audios), max_len)
        lens = self.torch.zeros(len(audios))
        for i, a in enumerate(audios):
            wav[i, : len(a)] = self.torch.from_numpy(a)
            lens[i] = len(a) / max_len
        _, _, _, text_lab = self.clf.classify_batch(wav.to(self.device), lens.to(self.device))
        preds = []
        for lab in text_lab:
            ours = norm_label(lab)
            preds.append(LABELS.index(ours) if ours else LABELS.index("neutral"))
        return preds


class MspSerAdapter:
    """3loi/SER-Odyssey-* custom SERModel: manual mean/std norm + attention mask."""

    def __init__(self, cfg, device):
        import torch
        from transformers import AutoModelForAudioClassification

        self.torch = torch
        self.device = device
        self.model = AutoModelForAudioClassification.from_pretrained(
            cfg["model_id"], trust_remote_code=True
        ).float().to(device).eval()
        self.mean = float(self.model.config.mean)
        self.std = float(self.model.config.std)
        self.out_map = []
        for i in range(len(self.model.config.id2label)):
            ours = norm_label(self.model.config.id2label[i])
            self.out_map.append(LABELS.index(ours) if ours else None)
        print(f"  id2label -> {dict(self.model.config.id2label)}", flush=True)

    def predict(self, audios):
        max_len = max(len(a) for a in audios)
        wav = self.torch.zeros(len(audios), max_len)
        mask = self.torch.zeros(len(audios), max_len)
        for i, a in enumerate(audios):
            wav[i, : len(a)] = self.torch.from_numpy((a - self.mean) / (self.std + 1e-6))
            mask[i, : len(a)] = 1
        with self.torch.no_grad():
            out = self.model(wav.to(self.device), mask.to(self.device))
        logits = (out.logits if hasattr(out, "logits") else out).float()
        probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        preds = []
        for p in probs:
            vec = np.full(len(LABELS), -1.0)
            for j, m in enumerate(self.out_map):
                if m is not None:
                    vec[m] = max(vec[m], float(p[j]))
            preds.append(int(np.argmax(vec)))
        return preds


class VoxProfileAdapter:
    """tiantiaf/wavlm-large-categorical-emotion via the vox-profile-release repo.

    Requires a local clone of https://github.com/tiantiaf0627/vox-profile-release;
    point ``repo_path`` at it (defaults to ~/vox-profile-release).
    """

    VOX_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral",
                  "Sadness", "Surprise", "Other"]

    def __init__(self, cfg, device):
        import sys

        import torch

        repo = cfg.get("repo_path", str(Path.home() / "vox-profile-release"))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.model.emotion.wavlm_emotion import WavLMWrapper

        self.torch = torch
        self.device = device
        self.model = WavLMWrapper.from_pretrained(cfg["model_id"]).to(device).eval()
        self.out_map = []
        for lab in self.VOX_LABELS:
            ours = norm_label(lab)
            self.out_map.append(LABELS.index(ours) if ours else None)

    def predict(self, audios):
        preds = []
        with self.torch.no_grad():
            for a in audios:  # model card recommends 3-15 s single clips
                data = self.torch.from_numpy(a[: 15 * TARGET_SR]).float()
                data = data.unsqueeze(0).to(self.device)
                out = self.model(data)
                # forward returns a tuple; first element is the 9-class logits
                logits = out[0] if isinstance(out, tuple) else out
                p = self.torch.softmax(logits, dim=1)[0].cpu().numpy()
                vec = np.full(len(LABELS), -1.0)
                for j, m in enumerate(self.out_map):
                    if m is not None:
                        vec[m] = max(vec[m], float(p[j]))
                preds.append(int(np.argmax(vec)))
        return preds


class W2v2CtcPoolAdapter:
    """r-f/wav2vec-english-* : checkpoint saved as Wav2Vec2ForCTC; per its model
    card, predictions come from mean-pooling the frame logits over time."""

    def __init__(self, cfg, device):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(cfg["model_id"])
        self.model = Wav2Vec2ForCTC.from_pretrained(cfg["model_id"])
        self.model = self.model.float().to(device).eval()
        id2label = self.model.config.id2label
        self.out_map = []
        for i in range(len(id2label)):
            ours = norm_label(id2label.get(i, str(i)))
            self.out_map.append(LABELS.index(ours) if ours else None)
        print(f"  id2label -> {dict(id2label)}", flush=True)

    def predict(self, audios):
        feats = self.fe(list(audios), sampling_rate=TARGET_SR, return_tensors="pt",
                        padding=True)
        with self.torch.no_grad():
            logits = self.model(feats["input_values"].to(self.device)).logits.float()
        probs = self.torch.softmax(logits.mean(dim=1), dim=-1).cpu().numpy()
        preds = []
        for p in probs:
            vec = np.full(len(LABELS), -1.0)
            for j, m in enumerate(self.out_map):
                if m is not None:
                    vec[m] = max(vec[m], float(p[j]))
            preds.append(int(np.argmax(vec)))
        return preds


ADAPTERS = {
    "funasr": FunasrAdapter,
    "sensevoice": SenseVoiceAdapter,
    "hf_audio_cls": HFAudioClsAdapter,
    "speechbrain": SpeechBrainAdapter,
    "msp_ser": MspSerAdapter,
    "voxprofile": VoxProfileAdapter,
    "w2v2_ctc_pool": W2v2CtcPoolAdapter,
}

# Registry of the open-source models on the leaderboard. All model_ids are
# public Hugging Face / ModelScope checkpoints. oruk models are evaluated with
# the same core protocol; their adapter will be added here when the weights are
# published.
MODELS = [
    {"name": "emotion2vec-plus-large", "adapter": "funasr",
     "model_id": "emotion2vec/emotion2vec_plus_large", "hub": "hf",
     "params_m": 300, "batch_size": 64},
    {"name": "emotion2vec-plus-base", "adapter": "funasr",
     "model_id": "emotion2vec/emotion2vec_plus_base", "hub": "hf",
     "params_m": 90, "batch_size": 64},
    {"name": "emotion2vec-plus-seed", "adapter": "funasr",
     "model_id": "emotion2vec/emotion2vec_plus_seed", "hub": "hf",
     "params_m": 90, "batch_size": 64},
    {"name": "emotion2vec-base-finetuned", "adapter": "funasr",
     "model_id": "iic/emotion2vec_base_finetuned", "hub": "ms",
     "params_m": 90, "batch_size": 64},
    {"name": "sensevoice-small", "adapter": "sensevoice",
     "model_id": "FunAudioLLM/SenseVoiceSmall", "hub": "hf",
     "params_m": 234, "batch_size": 32},
    {"name": "wavlm-odyssey-msp-3loi", "adapter": "msp_ser",
     "model_id": "3loi/SER-Odyssey-Baseline-WavLM-Categorical",
     "params_m": 319, "batch_size": 16},
    {"name": "wavlm-voxprofile-tiantiaf", "adapter": "voxprofile",
     "model_id": "tiantiaf/wavlm-large-categorical-emotion",
     "params_m": 320, "batch_size": 16},
    {"name": "hubert-large-superb-er", "adapter": "hf_audio_cls",
     "model_id": "superb/hubert-large-superb-er",
     "params_m": 316, "batch_size": 16},
    {"name": "wav2vec2-base-superb-er", "adapter": "hf_audio_cls",
     "model_id": "superb/wav2vec2-base-superb-er",
     "params_m": 95, "batch_size": 32},
    {"name": "xlsr-ravdess-ehcalabres", "adapter": "hf_audio_cls",
     "model_id": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
     "params_m": 316, "batch_size": 16},
    {"name": "whisper-large-v3-ser-firdhokk", "adapter": "hf_audio_cls",
     "model_id": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
     "params_m": 637, "batch_size": 16},
    {"name": "xlsr-ser-firdhokk", "adapter": "hf_audio_cls",
     "model_id": "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53",
     "params_m": 316, "batch_size": 16},
    {"name": "xlsr-ser-hughlan1214", "adapter": "hf_audio_cls",
     "model_id": ("hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53"
                  "_240304_SER_fine-tuned2.0"),
     "params_m": 316, "batch_size": 16},
    {"name": "w2v2-emotion-dpngtm", "adapter": "hf_audio_cls",
     "model_id": "Dpngtm/wav2vec2-emotion-recognition",
     "params_m": 95, "batch_size": 32},
    {"name": "xlsr-hatman", "adapter": "hf_audio_cls",
     "model_id": "Hatman/audio-emotion-detection",
     "params_m": 316, "batch_size": 16},
    {"name": "wav2vec-english-ser-rf", "adapter": "w2v2_ctc_pool",
     "model_id": "r-f/wav2vec-english-speech-emotion-recognition",
     "params_m": 95, "batch_size": 32},
    {"name": "speechbrain-w2v2-iemocap", "adapter": "speechbrain",
     "model_id": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
     "params_m": 95, "batch_size": 16},
    {"name": "xlsr-russian-aniemore", "adapter": "hf_audio_cls",
     "model_id": "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition",
     "trust_remote_code": True, "params_m": 316, "batch_size": 16},
    {"name": "wav2vec2-large-superb-er", "adapter": "hf_audio_cls",
     "model_id": "superb/wav2vec2-large-superb-er",
     "params_m": 316, "batch_size": 16},
    # DUSHA is spontaneous Russian call-centre speech, not acted studio audio.
    # id2label {neutral, angry, positive, sad, other}: 'other' stays unmapped,
    # leaving a 4-class model in our space.
    {"name": "hubert-dusha-russian-xbgoose", "adapter": "hf_audio_cls",
     "model_id": ("xbgoose/hubert-large-speech-emotion-recognition-russian-"
                  "dusha-finetuned"),
     "params_m": 316, "batch_size": 16},
    # Second Russian entrant: xlsr-russian-aniemore currently scores below the
    # 14.3% random baseline, and a same-language sibling is the cheapest way to
    # tell a weak checkpoint from a broken adapter.
    {"name": "wavlm-resd-russian-aniemore", "adapter": "hf_audio_cls",
     "model_id": "Aniemore/wavlm-emotion-russian-resd",
     "params_m": 317, "batch_size": 16},
]


def get_model_cfg(name):
    for cfg in MODELS:
        if cfg["name"] == name:
            return cfg
    return None


def run_eval(cfg, data_dir, device="cuda:0", out_dir="bench_results", max_examples=0):
    """Run one registered open-source model over the eval shards and score it."""
    tables, index, labels, langs, sources = load_eval(data_dir)
    n = len(index)
    order = np.arange(n)
    if max_examples and n > max_examples:
        rng = np.random.default_rng(0)
        order = np.sort(rng.choice(n, size=max_examples, replace=False))
        n = len(order)
    y = labels[order]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    print(f"=== {name} ({cfg['adapter']}: {cfg['model_id']}) on {n} clips ===", flush=True)
    adapter = ADAPTERS[cfg["adapter"]](cfg, device)

    bs = int(cfg.get("batch_size", 32))
    preds = np.full(n, -1, dtype=np.int64)
    t0 = time.time()
    for start in range(0, n, bs):
        idxs = order[start: start + bs]
        audios = [clip_audio(tables, index, int(i)) for i in idxs]
        preds[start: start + len(idxs)] = adapter.predict(audios)
        if (start // bs) % 20 == 0:
            done = start + len(idxs)
            rate = done / max(1e-9, time.time() - t0)
            print(f"  {done}/{n} ({rate:.1f} clips/s, "
                  f"eta {(n - done) / max(rate, 1e-9) / 60:.0f} min)", flush=True)

    supported = cfg.get("supported")
    if supported is None:
        if cfg["adapter"] == "hf_audio_cls":
            supported = sorted({LABELS[m] for m in adapter.out_map if m is not None},
                               key=LABELS.index)
        else:
            supported = list(LABELS)
    result = {
        "name": name,
        "model_id": cfg["model_id"],
        "adapter": cfg["adapter"],
        "params_m": cfg.get("params_m"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        **score(y, preds, langs[order], sources[order], supported),
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(out_dir / f"{name}.npz", preds=preds, labels=y, order=order)
    print(f"BENCH_MODEL_DONE {name} acc={result['accuracy']:.4f} "
          f"macro_f1={result['macro_f1']:.4f}", flush=True)
    return result
