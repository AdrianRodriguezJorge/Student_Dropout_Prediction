# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/__init__.py
#
#  Single import point for the dataset registry.
#  To add a new dataset: import its module and add its config here.
# ─────────────────────────────────────────────────────────────────────────────

from src.datasets import (
    ds1_uci_student,
    ds2_dropout,
    ds3_higher_ed,
    ds4_mathe,
    ds5_academics,
)

# Ordered dict: display name -> DatasetConfig
DATASET_REGISTRY = {
    ds1_uci_student.config.name: ds1_uci_student.config,
    ds2_dropout.config.name:     ds2_dropout.config,
    ds3_higher_ed.config.name:   ds3_higher_ed.config,
    ds4_mathe.config.name:       ds4_mathe.config,
    ds5_academics.config.name:   ds5_academics.config,
}

DATASET_NAMES = list(DATASET_REGISTRY.keys())
