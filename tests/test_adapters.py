"""Registry and label-mapping tests for the model adapters.

Runs on literal taxonomies, not weights: no checkpoint is downloaded and neither
torch nor transformers is imported, so this stays inside the CI contract in
CONTRIBUTING ("CI runs the score path on a synthetic fixture; it must not
download models"). See tests/test_model_smoke.py for the opt-in tests that do
load checkpoints.
"""

import importlib.util
from pathlib import Path

import pytest

from oruk_bench.adapters.open_models import ADAPTERS, MODELS, VoxProfileAdapter, get_model_cfg
from oruk_bench.core import LABELS, norm_label


def _map(id2label):
    """The label mapping every adapter builds: source label -> ours, or None."""
    return [norm_label(v) for v in id2label]


class TestOpenModelRegistry:
    @pytest.mark.parametrize("cfg", MODELS, ids=lambda c: c["name"])
    def test_entry_is_well_formed(self, cfg):
        for key in ("name", "adapter", "model_id", "params_m", "batch_size"):
            assert key in cfg, f"{cfg.get('name')} missing {key!r}"
        assert cfg["adapter"] in ADAPTERS, f"unknown adapter {cfg['adapter']!r}"
        assert isinstance(cfg["params_m"], int) and cfg["params_m"] > 0
        assert isinstance(cfg["batch_size"], int) and cfg["batch_size"] > 0
        assert cfg["model_id"].count("/") == 1, "model_id should be org/name"

    def test_names_are_unique_and_resolvable(self):
        names = [c["name"] for c in MODELS]
        assert len(names) == len(set(names))
        for cfg in MODELS:
            assert get_model_cfg(cfg["name"]) is cfg
        assert get_model_cfg("not-a-model") is None


class TestLabelMapping:
    def test_new_aliases(self):
        assert norm_label("enthusiasm") == "happiness"
        assert norm_label("positive") == "happiness"
        assert norm_label("calm") == "neutral"   # unchanged
        assert norm_label("bogus") is None

    # Published taxonomies of the checkpoints added with these aliases. If an
    # upstream config.json changes, this is what should fail.
    @pytest.mark.parametrize(
        "id2label, expected",
        [
            # xbgoose DUSHA -- 'other' is unmappable, leaving 4 classes
            (["neutral", "angry", "positive", "sad", "other"],
             ["neutral", "anger", "happiness", "sadness", None]),
            # Aniemore RESD -- 'enthusiasm' merges into happiness
            (["anger", "disgust", "enthusiasm", "fear", "happiness", "neutral", "sadness"],
             ["anger", "disgust", "happiness", "fear", "happiness", "neutral", "sadness"]),
            # superb ER -- IEMOCAP four-class short forms
            (["neu", "hap", "ang", "sad"],
             ["neutral", "happiness", "anger", "sadness"]),
        ],
        ids=["dusha", "resd", "superb-er"],
    )
    def test_new_checkpoint_taxonomies(self, id2label, expected):
        assert _map(id2label) == expected

    def test_voxprofile_head_leaves_contempt_and_other_unmapped(self):
        labels = VoxProfileAdapter.VOX_LABELS
        mapped = _map(labels)
        unmapped = [lab for lab, m in zip(labels, mapped) if m is None]
        assert unmapped == ["Contempt", "Other"]
        assert len(labels) == 9  # config.json output_class_num
        assert {m for m in mapped if m} == set(LABELS)


def test_leaderboard_params_match_the_registry():
    """build_leaderboard.py keeps its own copy of every params_m, and those go
    straight onto the public leaderboard."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_leaderboard.py"
    spec = importlib.util.spec_from_file_location("build_leaderboard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for cfg in MODELS:
        assert cfg["name"] in module.OPEN_PARAMS_M, f"{cfg['name']} missing"
        assert module.OPEN_PARAMS_M[cfg["name"]] == cfg["params_m"], cfg["name"]
