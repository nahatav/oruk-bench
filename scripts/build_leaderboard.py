"""Build leaderboard/leaderboard.json from benchmark result files.

Inputs:
  --report-data  aggregated open/closed results (report_data.json from the
                 benchmark run: {"open_full": {...}, "closed_5k": {...}})
  --closed-dir   directory of per-model result JSONs for the closed-API and
                 audio-LLM arms (gemini_*.json, openai_*.json, anthropic_*.json,
                 audiollm_*.json)
  --out          output path (default leaderboard/leaderboard.json)

Validity rule: any run whose API errors defaulted-to-neutral equal the full
subsample size (a 100%-failure run) is excluded and must be redone.

Usage:
  python scripts/build_leaderboard.py --report-data report_data.json \
      --closed-dir results_closed --out leaderboard/leaderboard.json
"""

import argparse
import datetime
import json
from pathlib import Path

FULL_SET_N = 64384
SUBSAMPLE_N = 5000

# Public display metadata for oruk models. Any run whose internal name starts
# with "ours" is published as oruk-spectra; model internals are not disclosed.
OURS_PREFIX = "ours"
OURS_META = {
    "model": "oruk-spectra",
    "params_m": None,
    "notes": "Oruk AI model, trained in-distribution; all other entrants "
             "are zero-shot cross-corpus.",
}

OPEN_PARAMS_M = {
    "emotion2vec-plus-large": 300,
    "emotion2vec-plus-base": 90,
    "emotion2vec-plus-seed": 90,
    "emotion2vec-base-finetuned": 90,
    "sensevoice-small": 234,
    "wavlm-odyssey-msp-3loi": 319,
    "wavlm-voxprofile-tiantiaf": 320,
    "hubert-large-superb-er": 316,
    "wav2vec2-base-superb-er": 95,
    "xlsr-ravdess-ehcalabres": 316,
    "whisper-large-v3-ser-firdhokk": 637,
    "xlsr-ser-firdhokk": 316,
    "xlsr-ser-hughlan1214": 316,
    "w2v2-emotion-dpngtm": 95,
    "xlsr-hatman": 316,
    "wav2vec-english-ser-rf": 95,
    "speechbrain-w2v2-iemocap": 95,
    "xlsr-russian-aniemore": 316,
    "wav2vec2-large-superb-er": 316,
    "hubert-dusha-russian-xbgoose": 316,
    "wavlm-resd-russian-aniemore": 317,
}

# nominal parameter counts (millions) for the open-weight audio-LLMs
AUDIO_LLM_PARAMS_M = {
    "audiollm-emotionthinker": 7000,
    "audiollm-qwen2.5-omni-7b": 7000,
    "audiollm-voxtral-mini-3b": 3000,
    "audiollm-audio-flamingo-3": 8000,
    "audiollm-meralion-2-10b": 10000,
}


def entry(model, group, params_m, accuracy, macro_f1, modality, subsample,
          ours=False, in_distribution=False, errors=None, notes=None):
    e = {
        "model": model,
        "group": group,
        "params_m": params_m,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "modality": modality,
        "subsample": subsample,
        "ours": ours,
        "in_distribution": in_distribution,
    }
    if errors is not None:
        e["api_errors_defaulted_to_neutral"] = int(errors)
    if notes:
        e["notes"] = notes
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-data", required=True)
    ap.add_argument("--closed-dir", required=True)
    ap.add_argument("--out", default="leaderboard/leaderboard.json")
    args = ap.parse_args()

    report = json.loads(Path(args.report_data).read_text(encoding="utf-8"))
    entries = []

    # -- open models, full 64,384-clip set --------------------------------
    for name, m in report["open_full"].items():
        if name.startswith(OURS_PREFIX):
            meta = OURS_META
            entries.append(entry(meta["model"], "oruk", meta["params_m"],
                                 m["acc"], m["mf1"], "audio", subsample=False,
                                 ours=True, in_distribution=True, notes=meta["notes"]))
        else:
            entries.append(entry(name, "open_ser", OPEN_PARAMS_M.get(name),
                                 m["acc"], m["mf1"], "audio", subsample=False))

    # -- closed API + audio-LLM arms, 5k stratified subsample --------------
    for path in sorted(Path(args.closed_dir).glob("*.json")):
        if path.name.endswith(".errors.json") or path.name.startswith("open_"):
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        name = j["name"]
        errors = j.get("api_errors_defaulted_to_neutral")
        n = j.get("subsample_n") or j.get("n")
        if errors is not None and n and errors >= n:
            print(f"[skip] {name}: 100% errors ({errors}/{n}) -- invalid run, redo")
            continue
        modality = j.get("modality") or "audio"
        if name.startswith("audiollm-"):
            group = "audio_llm"
            params = AUDIO_LLM_PARAMS_M.get(name)
        elif name.startswith("anthropic_"):
            group = "closed_api"
            params = None
            modality = "transcript-only"
        else:
            group = "closed_api"
            params = None
        entries.append(entry(name, group, params, j["accuracy"], j["macro_f1"],
                             modality, subsample=True, errors=errors))

    entries.sort(key=lambda e: -e["accuracy"])
    out = {
        "benchmark": "oruk-bench",
        "version": "0.1.0",
        "generated": datetime.date.today().isoformat(),
        "full_set_n": FULL_SET_N,
        "subsample_n": SUBSAMPLE_N,
        "labels": ["anger", "happiness", "sadness", "fear", "disgust",
                   "surprise", "neutral"],
        "metric_notes": {
            "subsample": "true = scored on the fixed 5,000-clip stratified "
                         "subsample (seed 0) used for API-priced and audio-LLM "
                         "models; false = full set. Rescoring open models on the "
                         "subsample shifts scores <2 points.",
            "in_distribution": "true = trained in-distribution for this "
                               "benchmark; other entrants are zero-shot cross-corpus.",
            "api_errors_defaulted_to_neutral": "refusals/parse failures/API "
                                               "errors count as a neutral "
                                               "prediction, never dropped.",
        },
        "entries": entries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
