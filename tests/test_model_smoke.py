"""Opt-in smoke tests that actually load checkpoints and run a forward pass.

Skipped by default. CI must not download models (see CONTRIBUTING), and the
heavier entries need tens of GB and a GPU, so this is gated behind an env var
rather than collected automatically::

    # every Hub-loadable open SER checkpoint (~20 GB of downloads)
    ORUK_BENCH_SMOKE=1 pytest tests/test_model_smoke.py -v

    # or just the ones you care about
    ORUK_BENCH_SMOKE=1 ORUK_BENCH_SMOKE_MODELS=wav2vec2-large-superb-er \
        pytest tests/test_model_smoke.py -v

What this catches that tests/test_adapters.py cannot: a checkpoint that does not
exist, has no feature-extractor config, needs trust_remote_code we did not set,
is wired to the wrong adapter, or whose published id2label maps onto none of our
seven labels. It runs on CPU and does not need the eval shards -- it proves the
adapter loads and produces well-formed predictions, not that the model is good.

Adapters needing setup beyond the Hub are excluded: ``voxprofile`` wants a local
clone of vox-profile-release, ``funasr``/``speechbrain`` pull their own runtimes,
and the audio-LLM arm needs a GPU. Point ``repo_path`` at a clone and add
``voxprofile`` to HUB_LOADABLE_ADAPTERS to cover that one locally.
"""

import os

import numpy as np
import pytest

from oruk_bench.adapters.open_models import ADAPTERS, MODELS
from oruk_bench.core import LABELS, TARGET_SR

SMOKE_ENABLED = os.environ.get("ORUK_BENCH_SMOKE") == "1"
ONLY = [s.strip() for s in os.environ.get("ORUK_BENCH_SMOKE_MODELS", "").split(",") if s.strip()]

# adapters that need nothing but a Hub download
HUB_LOADABLE_ADAPTERS = {"hf_audio_cls", "msp_ser", "w2v2_ctc_pool"}

CANDIDATES = [
    cfg for cfg in MODELS
    if cfg["adapter"] in HUB_LOADABLE_ADAPTERS and (not ONLY or cfg["name"] in ONLY)
]

pytestmark = pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="model smoke tests are opt-in; set ORUK_BENCH_SMOKE=1 (downloads weights)",
)


def _tone(seconds, freq=220.0):
    """A short deterministic clip; content is irrelevant, shape is not."""
    t = np.arange(int(seconds * TARGET_SR)) / TARGET_SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="module")
def clips():
    # different lengths on purpose: exercises each adapter's padding path
    return [_tone(1.5), _tone(3.0)]


@pytest.mark.parametrize("cfg", CANDIDATES, ids=lambda c: c["name"])
def test_checkpoint_loads_and_predicts(cfg, clips):
    adapter = ADAPTERS[cfg["adapter"]](cfg, "cpu")

    out_map = getattr(adapter, "out_map", None)
    assert out_map is not None, f"{cfg['name']}: adapter exposed no out_map"
    mapped = [LABELS[m] for m in out_map if m is not None]
    assert mapped, (
        f"{cfg['name']}: published id2label maps onto none of our 7 labels -- "
        f"every prediction would collapse to one class"
    )

    preds = adapter.predict(clips)

    assert len(preds) == len(clips)
    assert all(isinstance(p, (int, np.integer)) for p in preds)
    assert all(0 <= int(p) < len(LABELS) for p in preds)
    # a prediction outside the model's own supported set means the mapping is
    # wired wrong, not that the model was surprising
    assert all(LABELS[int(p)] in mapped for p in preds)


@pytest.mark.parametrize("cfg", CANDIDATES, ids=lambda c: c["name"])
def test_registered_params_m_is_in_the_right_ballpark(cfg):
    """params_m is published on the leaderboard; a wrong figure is a wrong
    public number. Tolerance is wide -- this catches order-of-magnitude slips
    (e.g. counting a Whisper encoder when the wrapper loads encoder+decoder),
    not rounding."""
    adapter = ADAPTERS[cfg["adapter"]](cfg, "cpu")
    actual_m = sum(p.numel() for p in adapter.model.parameters()) / 1e6
    claimed = cfg["params_m"]
    assert 0.75 * claimed <= actual_m <= 1.25 * claimed, (
        f"{cfg['name']}: registry says {claimed}M, checkpoint has {actual_m:.0f}M"
    )
