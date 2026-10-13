# -*- coding: utf-8 -*-
"""TF-IDF 稠密向量检索（余弦相似度），纯标准库实现。

要点：
- 使用 sublinear tf（1 + ln(tf)），避免长文本中重复词主导向量；
- 平滑 idf：ln((N+1)/(df+1)) + 1，避免除零；
- 向量 L2 归一化后余弦相似度 = 点积，检索时只需一次点积，复杂度 O(非零项)。
"""
import math
from collections import Counter

from tokenizer import tokenize_for_vector


class TfidfVectorizer:
    def __init__(self, doc_tokens):
        self.n = len(doc_tokens)
        df = Counter()
        for d in doc_tokens:
            for t in set(d):
                df[t] += 1
        self.df = df
        self.idf = {t: (math.log((self.n + 1) / (c + 1)) + 1.0) for t, c in df.items()}
        self.vectors = [self._vector(d) for d in doc_tokens]

    def _vector(self, tokens):
        tf = Counter(tokens)
        vec = {}
        for t, f in tf.items():
            vec[t] = (1.0 + math.log(f)) * self.idf.get(t, 1.0)
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {t: v / norm for t, v in vec.items()}
        return vec

    def encode(self, text):
        return self._vector(tokenize_for_vector(text))

    def scores(self, query_text):
        qv = self.encode(query_text)
        out = []
        for v in self.vectors:
            out.append(_dot(qv, v))
        return out

    def search(self, query, top_k=10):
        scores = self.scores(query)
        ranked = sorted(range(self.n), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]


def _dot(a, b):
    """稀疏向量点积，遍历较短的一方。"""
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for t, v in a.items():
        bv = b.get(t)
        if bv is not None:
            s += v * bv
    return s
