"""Text primitives shared by every source adapter.

Order matters and is enforced by tests: NFC normalisation happens once, before
any offset is computed, and nothing re-normalises afterwards.
"""
