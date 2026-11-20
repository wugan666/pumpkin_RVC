"""Numerically stable feature retrieval shared by offline and realtime RVC."""

import numpy as np


def retrieve_features(index, vectors, queries):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if index.ntotal < 1 or index.d != queries.shape[1]:
        raise ValueError("索引为空或与模型版本不匹配，请重新生成该音色的索引")
    scores, ids = index.search(queries, k=min(8, index.ntotal))
    valid = (ids >= 0) & (ids < len(vectors)) & np.isfinite(scores)
    # Exact matches have distance zero; missing IVF neighbours have ID -1.
    # Relative distances keep the squared inverse weights within [0, 1].
    distances = np.where(valid, np.maximum(scores, 1e-6), np.inf)
    nearest = np.min(distances, axis=1, keepdims=True)
    nearest = np.where(np.isfinite(nearest), nearest, 1.0)
    weights = np.square(nearest / distances)
    totals = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(totals, 1e-12)
    result = np.sum(vectors[np.clip(ids, 0, len(vectors) - 1)] * weights[..., None], axis=1)
    return np.where(totals > 0, result, queries).astype(np.float32, copy=False)
