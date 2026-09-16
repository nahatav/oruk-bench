"""Registry and label-mapping tests for the model adapters.

Runs on fixture taxonomies, not weights: no checkpoint is downloaded and
neither torch nor transformers is imported, so this stays inside the CI
contract in CONTRIBUTING ("CI runs the score path on a synthetic fixture; it
must not download models").

Scope note: these tests pin what ``norm_label`` does to a *known* taxonomy.
They read no ``config.json``, so they cannot detect a checkpoint changing its
labels upstream -- such a change would leave them passing. That guarantee lives
in tests/test_model_smoke.py, which asserts the same taxonomies against the
real downloaded config and is opt-in for exactly that reason.
"""

import importlib.util
from pathlib import Path

import pytest
from taxonomies import EXPECTED_ID2LABEL

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
        assert norm_label("positive") == "happiness"
        assert norm_label("calm") == "neutral"   # unchanged
        assert norm_label("bogus") is None

    def test_enthusiasm_is_deliberately_unmapped(self):
        """RESD and Aniemore's xlsr checkpoint expose enthusiasm *and*
        happiness. Aliasing enthusiasm would collapse two mutually exclusive
        softmax classes onto one of ours, and the adapters merge duplicates
        with max() rather than summing. Leaving it unmapped keeps every
        registered model free of that collapse."""
        assert norm_label("enthusiasm") is None

    # Fixtures, not drift detection: the taxonomy comes from
    # tests/taxonomies.py, so this pins what norm_label does to a known label
    # set. It cannot fail if a checkpoint changes its labels upstream -- see
    # test_model_smoke.py::test_id2label_matches_the_pinned_taxonomy for that.
    @pytest.mark.parametrize(
        "name, expected",
        [
            # DUSHA -- 'other' is unmappable, leaving 4 classes
            ("hubert-dusha-russian-xbgoose",
             ["neutral", "anger", "happiness", "sadness", None]),
            # RESD -- 'enthusiasm' stays unmapped (see above), leaving a
            # 6-class model that covers everything but surprise
            ("wavlm-resd-russian-aniemore",
             ["anger", "disgust", None, "fear", "happiness", "neutral", "sadness"]),
            # superb ER -- IEMOCAP four-class short forms
            ("wav2vec2-large-superb-er",
             ["neutral", "happiness", "anger", "sadness"]),
        ],
        ids=["dusha", "resd", "superb-er"],
    )
    def test_fixture_taxonomies_map_as_expected(self, name, expected):
        assert _map(EXPECTED_ID2LABEL[name]) == expected

    def test_every_pinned_taxonomy_names_a_registered_model(self):
        registered = {c["name"] for c in MODELS}
        assert set(EXPECTED_ID2LABEL) <= registered

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
