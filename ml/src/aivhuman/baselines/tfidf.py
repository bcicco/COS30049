"""tfidf + LR baseline"""

# hashing bc a real vocab over RAID bigrams + the matrix wont fit in 16GB

from pathlib import Path
from typing import Final

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from aivhuman.evaluate import EvalDoc, tpr_at_fpr, write_predictions

MODEL_NAME: Final = "tfidf-lr"
C_GRID: Final = (0.1, 1.0, 10.0)
N_FEATURES: Final = 2**21  # 2^21


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        ngram_range=(1, 2),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )


def run(
    splits: dict[str, list[EvalDoc]],
    predictions_dir: Path,
    c_grid: tuple[float, ...] = C_GRID,
) -> dict[float, float]:
    """Fit on train, pick C on dev TPR@1%FPR, write preds. Returns the grid."""
    hasher = _vectorizer()
    tfidf = TfidfTransformer(sublinear_tf=True)
    train = splits["train"]
    x_train = tfidf.fit_transform(hasher.transform(d.text for d in train))
    y_train = np.array([d.label for d in train])

    def features(docs: list[EvalDoc]) -> csr_matrix:
        return tfidf.transform(hasher.transform(d.text for d in docs))

    x_dev = features(splits["dev"])
    y_dev = np.array([d.label for d in splits["dev"]])

    grid: dict[float, float] = {}
    best, best_tpr = None, -1.0
    for c in c_grid:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        clf.fit(x_train, y_train)
        grid[c], _ = tpr_at_fpr(y_dev, clf.predict_proba(x_dev)[:, 1], 0.01)
        print(f"  C={c:g}: dev TPR@1%FPR {grid[c]:.4f}", flush=True)
        if grid[c] > best_tpr:
            best, best_tpr = clf, grid[c]
    assert best is not None

    for split, docs in splits.items():
        if split == "train":
            continue
        scores = best.predict_proba(features(docs))[:, 1]
        write_predictions(
            predictions_dir / MODEL_NAME / f"{split}.parquet",
            [d.doc_id for d in docs],
            scores,
        )
    return grid
