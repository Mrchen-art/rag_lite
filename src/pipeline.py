# -*- coding: utf-8 -*-
"""RAG 流水线编排：切分 -> 双路索引 -> 混合召回 -> 答案生成。

检索模式：
- bm25   ：稀疏检索，擅长精确关键词、专有名词、型号/术语匹配
- vector ：稠密检索（TF-IDF 余弦），擅长同义表述、语序不同的语义匹配
- hybrid ：两路各召回 top_n 后用 RRF 融合（默认，效果最好）
"""
import time

from chunker import build_chunks
from tokenizer import tokenize_for_bm25, tokenize_for_vector
from bm25 import BM25
from embedder import TfidfVectorizer
from fusion import rrf_fuse, normalize_max
from reader import extractive_answer, build_prompt, LLMClient
from loader import load_corpus, load_demo_docs

# 融合权重由 tests/qa.json 上的网格搜索确定（见 README「调参实验」一节），
# 并非拍脑袋取值：向量路在本语料上的排序质量更好，因此获得更高权重。
FUSION_WEIGHTS = [0.8, 0.6]

# RRF 的平滑常数：k 越小，越强调头部排名（榜单靠前的差异被放大）
RRF_K = 60


class RagPipeline:
    def __init__(self, docs=None, data_dir=None, chunk_size=320, overlap=60,
                 llm_api_key=None, llm_base_url=None, llm_model=None):
        if docs is None:
            docs = load_corpus(data_dir) if data_dir else []
        if not docs:
            docs = load_demo_docs()

        self.docs = docs
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = build_chunks(docs, chunk_size, overlap)
        self.bm25_tokens = [tokenize_for_bm25(c["text"]) for c in self.chunks]
        self.vec_tokens = [tokenize_for_vector(c["text"]) for c in self.chunks]
        self.bm25 = BM25(self.bm25_tokens)
        self.tfidf = TfidfVectorizer(self.vec_tokens)
        self.llm = LLMClient(api_key=llm_api_key, base_url=llm_base_url, model=llm_model)
        self._bm25_cache = {}
        self._vec_cache = {}
        self.suggestions = [
            "RAG 的标准流程包含哪些环节",
            "BM25 和稠密向量检索各有什么优势",
            "为什么 ResNet 可以训练很深的网络",
            "Transformer 为什么要加入位置编码",
            "缓存穿透、击穿和雪崩分别怎么解决",
        ]

    # ---------- 索引信息 ----------
    def stats(self):
        return {
            "doc_count": len(self.docs),
            "chunk_count": len(self.chunks),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "vocab_size": len(self.bm25.df),
            "avg_chunk_len": round(
                sum(len(c["text"]) for c in self.chunks) / max(1, len(self.chunks)), 1),
            "llm_enabled": self.llm.available,
            "model": self.llm.model if self.llm.available else None,
            "titles": [d[1] for d in self.docs],
        }

    # ---------- 检索 ----------
    def _bm25_scores(self, query):
        if query not in self._bm25_cache:
            self._bm25_cache[query] = self.bm25.scores(tokenize_for_bm25(query))
        return self._bm25_cache[query]

    def _vec_scores(self, query):
        if query not in self._vec_cache:
            self._vec_cache[query] = self.tfidf.scores(query)
        return self._vec_cache[query]

    def search(self, query, top_k=5, mode="hybrid", recall_n=30):
        """返回 list[hit dict]，hit 含分数明细。"""
        if not query or not query.strip():
            return []

        bm = self._bm25_scores(query)
        vc = self._vec_scores(query)
        bm_n = normalize_max(bm)
        vc_n = normalize_max(vc)

        ranked_bm = sorted(range(len(bm)), key=lambda i: -bm[i])[:recall_n]
        ranked_vc = sorted(range(len(vc)), key=lambda i: -vc[i])[:recall_n]

        if mode == "bm25":
            order = [(i, bm_n[i]) for i in ranked_bm[:top_k] if bm[i] > 0]
        elif mode == "vector":
            order = [(i, vc_n[i]) for i in ranked_vc[:top_k] if vc[i] > 0]
        else:
            fused = rrf_fuse(
                [[(i, bm[i]) for i in ranked_bm], [(i, vc[i]) for i in ranked_vc]],
                k=RRF_K, weights=FUSION_WEIGHTS,
            )
            order = fused[:top_k]

        hits = []
        for rank, (idx, score) in enumerate(order, 1):
            c = self.chunks[idx]
            hits.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "text": c["text"],
                "score": score,
                "bm25_score": round(bm_n[idx], 4),
                "vector_score": round(vc_n[idx], 4),
            })
        return hits

    # ---------- 生成 ----------
    def answer(self, query, top_k=3, mode="hybrid", use_llm=None):
        t0 = time.time()
        hits = self.search(query, top_k=top_k, mode=mode)
        use_llm = self.llm.available if use_llm is None else (use_llm and self.llm.available)

        # extractive_answer 同时产出"来源引用"，大模型模式也复用它做溯源
        _, citations = extractive_answer(query, hits)

        if use_llm:
            try:
                answer = self.llm.chat(build_prompt(query, hits))
                gen_mode = "llm"
            except Exception as exc:  # 网络/额度异常时自动降级，保证服务不中断
                answer = extractive_answer(query, hits)[0]
                gen_mode = "extractive(fallback)"
                answer += f"\n\n[提示] 大模型调用失败({type(exc).__name__})，已降级为抽取式答案。"
        else:
            answer = extractive_answer(query, hits)[0]
            gen_mode = "extractive"

        return {
            "query": query,
            "mode": mode,
            "generation": gen_mode,
            "answer": answer,
            "citations": citations,
            "hits": hits,
            "latency_ms": int((time.time() - t0) * 1000),
        }
