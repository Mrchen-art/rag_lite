# -*- coding: utf-8 -*-
"""文档切分（Chunking）。

为什么不用"固定长度硬切"：
固定长度切分会把一句话拦腰截断，导致检索到的块语义不完整，
LLM 或抽取式阅读器拿到断句后生成的答案质量会明显下降。

本实现的策略：
1. 先按句子边界切分，得到天然语义单元；
2. 贪心地把句子装入块，直到接近 chunk_size；
3. 装不下时先收尾，再**按句子为粒度回退 overlap 长度**，
   保证相邻块之间重叠的是完整句子，而不是半个词。
"""
from tokenizer import split_sentences


def chunk_text(text, chunk_size=320, overlap=60, min_chunk=20):
    """把一篇文档切成若干重叠块，返回 list[str]。"""
    sents = split_sentences(text)
    if not sents:
        return []

    chunks = []
    cur, cur_len = [], 0

    for s in sents:
        s_len = len(s)
        # 单句超长：强制再按字符切，避免块过大
        if s_len > chunk_size:
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            for i in range(0, s_len, chunk_size):
                piece = s[i:i + chunk_size]
                if len(piece) >= min_chunk:
                    chunks.append(piece)
            continue

        if cur_len + s_len > chunk_size and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = _tail_overlap(cur, overlap)

        cur.append(s)
        cur_len += s_len

    if cur:
        chunks.append("\n".join(cur))

    return [c for c in chunks if len(c) >= min_chunk]


def _tail_overlap(sent_list, overlap):
    """从句子列表尾部回退，保留累计长度不超过 overlap 的若干完整句子。"""
    keep, total = [], 0
    for s in reversed(sent_list):
        if total + len(s) > overlap:
            break
        keep.insert(0, s)
        total += len(s)
    return keep, total


def build_chunks(docs, chunk_size=320, overlap=60):
    """docs: list[(doc_id, title, text)] -> list[dict]"""
    out = []
    for doc_id, title, text in docs:
        for i, c in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append({
                "chunk_id": f"{doc_id}#{i}",
                "doc_id": doc_id,
                "title": title,
                "text": c,
            })
    return out
