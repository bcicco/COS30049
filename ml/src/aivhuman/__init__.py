"""Sentence-scored AI-text detection.

Phase 1 turns RAID, MAGE and SeqXGPT into one JSONL schema with character-exact
sentence spans. See ml/PLAN.md for the dev plan.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
