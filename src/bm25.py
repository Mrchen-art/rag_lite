# -*- coding: utf-8 -*-
"""BM25 稀疏检索，纯标准库实现。

BM25 打分公式：
    score(q, d) = Σ_{t∈q∩d} idf(t) * (f*(k1+1)) / (f + k1*(1 - b + b*|d|/avgdl))

- idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
- k1 控制词频饱和度，b 控制文档长度惩罚强度
- 相比 TF-IDF：BM25 对词频做了饱和处理，长文档不会因为重复词而无限加分
"""
import math
from collections import Counter

from tokenizer import tokenize_for_bm25


class BM25:
    def __init__(self, doc_tokens, k1=1.5, b=0.75):
        self.docs = doc_tokens
        self.n = len(doc_tokens)
        self.k1 = k1
        self.b = b
        self.lens = [len(d) for d in doc_tokens]
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0

        df = Counter()
        tf = []
        for d in doc_tokens:
            c = Counter(d)
            tf.append(c)
            for t in c:
                df[t] += 1
        self.df = df
        self.tf = tf
        self.idf = {t: math.log(1 + (self.n - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def scores(self, query_tokens):
        """返回与 docs 等长的分数列表。"""
        q = [t for t in set(query_tokens) if t in self.idf]
        out = [0.0] * self.n
        if not q:
            return out
        for i in range(self.n):
            tf_i = self.tf[i]
            if not tf_i:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1.0))
            s = 0.0
            for t in q:
                f = tf_i.get(t, 0)
                if f:
                    s += self.idf[t] * (f * (self.k1 + 1)) / (f + norm)
            out[i] = s
        return out

    def search(self, query, top_k=10):
        scores = self.scores(tokenize_for_bm25(query))
        ranked = sorted(range(self.n), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]
