"""Published ``id2label`` taxonomies of the registered checkpoints, in order.

Order is load-bearing: adapters build ``out_map`` positionally, so a checkpoint
that reorders its labels upstream would silently misalign every prediction
while still looking healthy.

These serve two tests with different guarantees, and the difference matters:

- ``tests/test_adapters.py`` uses them as **fixtures**. Those tests pin what
  ``norm_label`` does to a known taxonomy. They never read a ``config.json``,
  so they cannot detect upstream drift -- an upstream change would leave them
  passing.
- ``tests/test_model_smoke.py`` asserts these against the **real** downloaded
  ``config.json``. That is where drift is actually caught, and it is opt-in
  (``ORUK_BENCH_SMOKE=1``) so default CI stays download-free.
"""

EXPECTED_ID2LABEL = {
    "wav2vec2-large-superb-er": [
        "neu", "hap", "ang", "sad",
    ],
    "hubert-dusha-russian-xbgoose": [
        "neutral", "angry", "positive", "sad", "other",
    ],
    "wavlm-resd-russian-aniemore": [
        "anger", "disgust", "enthusiasm", "fear", "happiness", "neutral", "sadness",
    ],
}
