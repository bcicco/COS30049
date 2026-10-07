# aivhuman

Sentence level AI text detection using multiple instance learning (MIL). Run everything
from inside `ml/`.

## Layout

```
ml/
  environment.yml          conda env
  pyproject.toml           package + the three cli tools
  src/aivhuman/            code (see Source below)
  tests/                   pytest
  manifests/               split membership (doc_id -> group_id)
  data/processed/phase1/   processed corpora, one doc per line (JSONL)
  data/features/           prepared dataset, one row per sentence (Parquet)
  data/checkpoints/p5-adv/ final model (model.pt) + calibrator (calibrator.json)
  reports/                 results used in the report (JSON, PNG)
```

## Data

`data/processed/phase1/` has the normalised, sentence segmented docs for each corpus
(`raid`, `raid-attacks`, `mage`, `seqxgpt`, `daigt`). One JSON object per line, schema is in
`src/aivhuman/schema.py` (text, label, source, domain, generator, group, sentence offsets).
Only docs that are in a split in `manifests/` are kept.

`data/features/` is one Parquet file per split. A row is one sentence: doc id, position,
doc label (0 = human, 1 = machine), domain, generator, sentence label (SeqXGPT only) and
the 21 features. Its Parquet bc there are millions of rows, `pandas.read_parquet(path)`
opens them fine.

| File | Role |
| --- | --- |
| `train`, `train-spliced`, `train-adv`, `train-adv-spliced` | Training data of the final model (RAID; spliced human/machine documents; attacked RAID documents) |
| `dev` | Model selection (document partial AUROC) |
| `seqxgpt-calib` | Model selection (sentence AUROC on a 20% slice) and calibration fit |
| `seqxgpt-test` | Sentence-level evaluation |
| `dev-spliced` | Calibration check on RAID spliced documents |
| `raid-ood`, `mage-x`, `mage-para`, `daigt` | Document-level evaluation; never trained on |
| `dev-adv`, `raid-ood-adv` | Evaluation on attacked RAID documents |

Corpora: RAID (Dugan et al. 2024), MAGE (Li et al. 2024), SeqXGPT (Wang et al. 2023) and
DAIGT v2 (Kłeczek 2023). Full refs are in the report.

## Setup

```bash
conda env create -f environment.yml
conda activate aivhuman
cp .env.example .env        # Windows cmd: copy .env.example .env
python -m pytest            # should all pass
```

This gets you Python 3.13, PyTorch (CUDA 12.8 build), the spaCy English model and the
package in editable mode, so `aivhuman-data`, `aivhuman-baseline` and `aivhuman-mil` end up
on the path. Works on CPU too if you dont have an NVIDIA GPU. `.env` defaults are fine,
`HF_TOKEN` can be left empty.

## Vetting features

Checks for corpus artefacts, length confounding and transfer (Table 4 in the report):

```bash
aivhuman-mil vet            # writes reports/phase4/feature_vetting.json
```

Standardisation (z-scores fit on clean train, missing -> mean) and dropping the excluded
features both happen in `train`, nothing else to run first.

### Rebuilding from raw (optional, slow)

Only if you want to regenerate `data/` from scratch. ~12 GB download, and feature
extraction wants a GPU (I used an A100).

```bash
aivhuman-data acquire       # download RAID, MAGE, SeqXGPT and DAIGT into data/raw
aivhuman-data derive        # stream RAID's CSV into Parquet (clean rows + one partition per attack)
aivhuman-data ingest        # normalise (NFC), segment into sentences, write data/processed/phase1/*.jsonl
aivhuman-data verify        # re-check every document's offsets and labels
aivhuman-data report        # dataset summary in reports/phase1
aivhuman-data split         # write the split manifests in manifests/
aivhuman-mil extract --split train dev raid-ood mage-x mage-para seqxgpt-calib seqxgpt-test daigt
aivhuman-data attacks       # attacked RAID documents and their manifests
aivhuman-mil extract --split train-adv dev-adv raid-ood-adv
aivhuman-mil splice         # spliced human/machine documents from train and dev
aivhuman-mil splice --adv   # the same from the attacked documents
```

## Training

Final model is `p5-adv`: GAM head, length normalised log-sum-exp pooling (temp = 2,
L1 = 1e-4), trained on clean + spliced + attacked RAID docs. About a minute on CPU.

```bash
aivhuman-mil train --kept --mix spliced-adv --run p5-adv
aivhuman-mil calibrate --run p5-adv
```

`train` writes `data/checkpoints/p5-adv/model.pt` and the selection results to
`reports/phase5/p5-adv/`. `calibrate` fits the per length calibrator (`calibrator.json`)
and writes `reports/phase6/calibration.json`. Drop `--kept` to run the full pooling / L1
sweep instead, it keeps the best config.

Baselines train on the text in `data/processed`:

```bash
aivhuman-baseline tfidf     # tf-idf + logistic regression
aivhuman-baseline encoder   # fine-tuned ModernBERT (GPU)
```

## Scoring

First run pulls GPT-2 (~500 MB) into `data/hf`. Scoring is on CPU.

```bash
aivhuman-mil score "Paste a paragraph of text here."
aivhuman-mil score --file essay.txt
aivhuman-mil score --file essay.txt --json     # full result: offsets, scores, contributions
```

Per sentence you get the calibrated probability, `FLAG` if its over the 1% sentence FPR
threshold, `short` if under 15 tokens (capped, never flagged), and the top 3 feature
contributions (positive = more machine). Then the doc probability and the error rates
measured at those thresholds.

Whole prepared splits:

```bash
aivhuman-mil predict --run p5-adv     # writes data/predictions/p5-adv/{split}.parquet
```

`report` compares against the baselines on every test split, so run `predict` and both
baseline commands before it:

```bash
aivhuman-mil report --run p5-adv      # writes reports/phase7/evaluation.json
aivhuman-mil cluster --run p5-adv     # k-means clusters of machine sentences on dev
```

## Tests

```bash
python -m pytest            # unit tests, no data needed
ruff check . && mypy        # lint + strict types (CI runs these too)
```

## Source

| Path | Contents |
| --- | --- |
| `cli.py`, `acquire.py`, `sources/`, `text/`, `ingest.py`, `verify.py`, `splits.py`, `group.py`, `labels.py`, `schema.py`, `report.py` | `aivhuman-data`: download, normalise, segment, validate and split the corpora |
| `baselines/` | `aivhuman-baseline`: tf-idf + LR and ModernBERT |
| `features/` | Feature extraction, vetting and splicing |
| `mil/` | `aivhuman-mil`: MIL model, training, prediction, calibration, clustering, evaluation report, text scoring |
| `evaluate.py` | Shared metrics (TPR at fixed FPR, partial AUROC, bootstrap intervals) |
