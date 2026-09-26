"""Derive RAID's 11.8 GB CSV into parquet, once."""

# big data nugget of wisdom here..... this is an 11gb csv
# chose to read as parquet because parquet stores data by colum, we only want subsect of data where
# attack == none, integrity check only takes 8 seconds XD
# replaced the prompt column with prompt_sha and prompt_len_chars so still have identifier with sig.
# less storage

# also parquet column types in general just makes me happier inside

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import csv as pacsv
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from aivhuman.text.normalize import stable_hash

# train.csv's columns, in order.
COLUMNS: Final = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation",
]

OUTPUT_COLUMNS: Final = [
    *[c for c in COLUMNS if c != "prompt"],
    "prompt_sha",
    "prompt_len_chars",
]

CLEAN_ATTACK: Final = "none"
CLEAN_FILE: Final = "clean.parquet"
ATTACK_DIR: Final = "by_attack"

# 64 MB CSV blocks: ~35k rows per batch, which keeps parquet row groups a
# sensible size
_BLOCK_SIZE: Final = 1 << 26  # 2^26 = 64Mb


# ----------------- MEMORY MANAGEMENT EXPLANATION, CAN SKIP ------------------------------- #
# Clean rows are ~7% of each block (from data. exploration), so they are buffered to this many rows
# before a row group is written. Without it the output gets one 2.5k-row row
# group per input block, and every later read pays for the fragmentation.
_FLUSH_ROWS: Final = 50_000

_COMPRESSION: Final = "zstd"


class RaidDeriveStats(BaseModel):
    """What the pass saw"""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    rows: NonNegativeInt = 0
    clean_rows: NonNegativeInt = 0
    blocks: NonNegativeInt = 0
    by_attack: dict[str, int] = Field(default_factory=dict)
    clean_by_model: dict[str, int] = Field(default_factory=dict)
    clean_by_domain: dict[str, int] = Field(default_factory=dict)
    clean_by_decoding: dict[str, int] = Field(default_factory=dict)

    @property
    def clean_frac(self) -> float:
        return self.clean_rows / self.rows if self.rows else 0.0

    @property
    def clean_human_rows(self) -> int:
        """Rows whose model column is human."""
        return self.clean_by_model.get("human", 0)

    @property
    def clean_machine_rows(self) -> int:
        return self.clean_rows - self.clean_human_rows

    @property
    def machine_per_human(self) -> float:
        """For class balancing later down the track"""
        return self.clean_machine_rows / self.clean_human_rows if self.clean_human_rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "clean_rows": self.clean_rows,
            "clean_frac": round(self.clean_frac, 4),
            "clean_human_rows": self.clean_human_rows,
            "clean_machine_rows": self.clean_machine_rows,
            "machine_per_human": round(self.machine_per_human, 2),
            "blocks": self.blocks,
            "by_attack": dict(sorted(self.by_attack.items())),
            "clean_by_model": dict(sorted(self.clean_by_model.items())),
            "clean_by_domain": dict(sorted(self.clean_by_domain.items())),
            "clean_by_decoding": dict(sorted(self.clean_by_decoding.items())),
        }


def derive(
    csv_path: Path,
    out_dir: Path,
    *,
    include_attacks: bool = True,
    block_size: int = _BLOCK_SIZE,
    limit_blocks: int | None = None,
    stats: RaidDeriveStats | None = None,
    progress: bool = False,
) -> RaidDeriveStats:
    """Scan csv_path once, writing the clean subset and the attack partitions."""

    # this function is gross, but its works so nobody question it #
    st = stats if stats is not None else RaidDeriveStats()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_writer: pq.ParquetWriter | None = None
    attack_writers: dict[str, pq.ParquetWriter] = {}
    buffered: list[pa.RecordBatch] = []
    buffered_rows = 0

    try:
        for batch in _read_batches(csv_path, block_size):
            st.blocks += 1
            st.rows += batch.num_rows
            _tally(st.by_attack, batch, "attack")
            out = _transform(batch)

            if include_attacks:
                for attack in _attack_values(batch):
                    part = out.filter(pc.equal(batch.column("attack"), attack))
                    writer = attack_writers.get(attack)
                    if writer is None:
                        path = out_dir / ATTACK_DIR / f"attack={attack}" / "part-0000.parquet"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(path, out.schema, compression=_COMPRESSION)
                        attack_writers[attack] = writer
                    writer.write_batch(part)

            clean = out.filter(pc.equal(batch.column("attack"), CLEAN_ATTACK))
            if clean.num_rows:
                st.clean_rows += clean.num_rows
                _tally(st.clean_by_model, clean, "model")
                _tally(st.clean_by_domain, clean, "domain")
                _tally(st.clean_by_decoding, clean, "decoding")
                buffered.append(clean)
                buffered_rows += clean.num_rows

            if buffered_rows >= _FLUSH_ROWS:
                if clean_writer is None:
                    clean_writer = pq.ParquetWriter(
                        out_dir / CLEAN_FILE, out.schema, compression=_COMPRESSION
                    )
                clean_writer.write_table(pa.Table.from_batches(buffered))
                buffered, buffered_rows = [], 0

            if progress and st.blocks % 20 == 0:
                print(
                    f"  block {st.blocks}: {st.rows:,} rows, {st.clean_rows:,} clean",
                    flush=True,
                )
            if limit_blocks is not None and st.blocks >= limit_blocks:
                break

        if buffered:
            if clean_writer is None:
                clean_writer = pq.ParquetWriter(
                    out_dir / CLEAN_FILE,
                    pa.Table.from_batches(buffered).schema,
                    compression=_COMPRESSION,
                )
            clean_writer.write_table(pa.Table.from_batches(buffered))
    finally:
        if clean_writer is not None:
            clean_writer.close()
        for writer in attack_writers.values():
            writer.close()

    return st


def open_clean(path: Path) -> pq.ParquetFile:
    """Open the derived clean parquet, checking it is the file we think it is."""
    handle = pq.ParquetFile(path)
    names = list(handle.schema_arrow.names)
    if names != OUTPUT_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {OUTPUT_COLUMNS}, got {names}")
    return handle


def _read_batches(csv_path: Path, block_size: int) -> Iterator[pa.RecordBatch]:
    """Stream the CSV in file order."""
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(COLUMNS, pa.string())),
    )
    names = list(reader.schema.names)
    if names != COLUMNS:
        raise ValueError(f"{csv_path.name}: expected columns {COLUMNS}, got {names}")
    return iter(reader)


def _transform(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Drop prompt, adding its digest and length in its place."""
    prompt = batch.column(COLUMNS.index("prompt"))
    columns = [batch.column(i) for i, name in enumerate(COLUMNS) if name != "prompt"]
    return pa.RecordBatch.from_arrays(
        [*columns, _prompt_sha(prompt), pc.utf8_length(prompt)],
        names=OUTPUT_COLUMNS,
    )


def _prompt_sha(prompt: pa.Array) -> pa.Array:
    """Hash the prompt column via its dictionary, not row by row."""
    encoded = pc.dictionary_encode(prompt)
    digests = pa.array([stable_hash(v or "") for v in encoded.dictionary.to_pylist()])
    return pa.DictionaryArray.from_arrays(encoded.indices, digests).cast(pa.string())


def _attack_values(batch: pa.RecordBatch) -> list[str]:
    """The distinct attacks in this batch. Rows are attack-sorted, so usually one."""
    return [v for v in pc.unique(batch.column("attack")).to_pylist() if v is not None]


def _tally(counter: dict[str, int], batch: pa.RecordBatch, column: str) -> None:
    """Accumulate value counts in Arrow rather than per row in Python."""
    for entry in pc.value_counts(batch.column(column)).to_pylist():
        key = entry["values"] if entry["values"] is not None else ""
        counter[key] = counter.get(key, 0) + entry["counts"]
