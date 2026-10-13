# -*- coding: utf-8 -*-
"""多路召回结果融合：Reciprocal Rank Fusion (RRF)。

RRF 公式：score(d) = Σ_r w_r / (k + rank_r(d))，k 默认 60。

为什么用 RRF 而不是分数加权：
BM25 输出的是无界的词频统计分，TF-IDF 余弦输出的是 [0,1] 相似度，
两者量纲不可比，直接加权会严重偏向分值域大的一路。
RRF 只使用**排名信息**，天然规避量纲问题，且实现只需几行。
"""
DEFAULT_K = 60


def rrf_fuse(rank_lists, k=DEFAULT_K, weights=None):
    """rank_lists: list[list[(idx, score)]]，返回按融合分降序的 [(idx, score)]。"""
    fused = {}
    for r, ranking in enumerate(rank_lists):
        w = 1.0 if weights is None else weights[r]
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def normalize_max(scores):
    """把任意分值域压缩到 [0,1]，用于展示与调试。"""
    if not scores:
        return []
    m = max(scores)
    if m <= 0:
        return [0.0] * len(scores)
    return [s / m for s in scores]
