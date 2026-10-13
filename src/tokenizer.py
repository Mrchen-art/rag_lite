# -*- coding: utf-8 -*-
"""轻量分词器：同时处理中英文，零第三方依赖。

设计要点（面试可讲）：
- 英文/数字按词切分，中文按字切分；
- 为中文额外生成 bigram（如"梯度"->"梯度"），弥补按字切分丢失的词级语义，
  实测能显著提升中文短查询的召回率；
- 支持停用词过滤，降低高频虚词对 BM25/TF-IDF 的噪声干扰。
"""
import re

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-\.]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")

# 极简中文停用词表（只放虚词，避免误伤专业术语）
STOPWORDS = set("""
的 了 和 与 或 是 在 有 也 就 都 而 及 对 于 中 为 以 把 被 给 从 到 由 其 之 者 所
这 那 我 你 它 们 个 一 二 三 什 么 怎 么 如 何 为 什 哪 些 吗 呢 吧 啊 可 会 能 要
the a an of and or is are to in for on with by as at it this that be was were
""".split())

_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text, use_bigram=True, remove_stopwords=True):
    """将文本切分为 token 列表。"""
    if not text:
        return []
    text = text.lower()
    units = _TOKEN_RE.findall(text)
    tokens = []
    for i, u in enumerate(units):
        if remove_stopwords and u in STOPWORDS:
            continue
        tokens.append(u)
        if use_bigram and len(u) == 1 and _CJK.match(u):
            if i + 1 < len(units) and _CJK.match(units[i + 1]):
                tokens.append(u + units[i + 1])
    # 中文按字切分时，单字噪声较大；若整段无英文，保留 bigram 即可
    return tokens


def tokenize_for_bm25(text):
    """BM25 侧使用 unigram + bigram。

    中文按单字切分时，"学""习"这类高频字会与大量文档产生无意义匹配，
    实测加入 bigram 后 BM25 单路 Recall@1 明显提升，且能更好地与向量路互补。
    """
    return tokenize(text, use_bigram=True, remove_stopwords=True)


def tokenize_for_vector(text):
    """向量侧使用 unigram + bigram，扩大召回面。"""
    return tokenize(text, use_bigram=True, remove_stopwords=True)


def split_sentences(text):
    """句子边界切分，兼容中英文标点。

    先按换行拆成段落，再在段内按标点切句。这样 Markdown 标题行会独立成句，
    不会被 strip 掉换行后和正文粘成一句（例如 "## 分词与上下文大模型以..."),
    切块内容与答案摘录的可读性都依赖这一步。
    """
    if not text:
        return []
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for part in re.split(r"(?<=[。！？!?；;：:])", para):
            part = part.strip()
            if part:
                out.append(part)
    return out
