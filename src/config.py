# ─────────────────────────────────────────────────────────────────────────────
#  config.py — Global constants only.
#  Business logic lives in src/models/ and src/datasets/.
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # fraction held out for final evaluation
CV_FOLDS     = 5      # StratifiedKFold splits
