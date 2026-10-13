# -*- coding: utf-8 -*-
"""答案生成层（Reader）。

本项目的关键工程决策：**双模式生成**
1. extractive（默认，零依赖）：不调用任何大模型，从命中块中筛选与查询最相关的
   句子，去重后拼接成段落式答案，并附来源引用。保证没有 API key 也能完整演示。
2. llm（可选）：传入 LLM 客户端后，用检索到的块作为上下文走 RAG 生成。
   接口保持统一，方便后续替换成 DeepSeek / 通义 / OpenAI 兼容接口。

抽句式答案的核心打分：句子与查询的 token 覆盖率 + 句子在块中的原始检索分。
"""
from tokenizer import split_sentences, tokenize_for_bm25


def _sentence_score(sent, q_tokens, base_score):
    if not q_tokens:
        return 0.0
    s_tokens = set(tokenize_for_bm25(sent))
    if not s_tokens:
        return 0.0
    hit = len(q_tokens & s_tokens)
    coverage = hit / len(q_tokens)
    # 只蹭上一个单字就混进答案的句子，是抽取式答案跑题的主要来源，直接淘汰
    if coverage < MIN_COVERAGE:
        return 0.0
    # 惩罚过短句子（信息量低）和过长句子（噪声多）
    length_penalty = 1.0 if 12 <= len(sent) <= 200 else 0.6
    return (coverage * 2.0 + base_score * 0.5) * length_penalty


# 句子至少要命中这么多比例的查询词，才够格进入答案。
MIN_COVERAGE = 0.15

# 只让排名前几的块参与答案生成。按分数过滤是无效的：RRF 融合分差异极小
# （实测 0.0229 vs 0.0226，比值 0.99），筛不掉任何东西，只能按排名卡。
MAX_ANSWER_CHUNKS = 2


def extractive_answer(query, hits, max_sentences=4):
    """hits: list[dict]，需含 text / score / title / chunk_id。返回 (answer, citations)

    抽取式答案有个致命弱点：它会把检索结果"照单全收"。
    实测中问"RAG 的标准流程"，top-1 命中正确文档，但 top-2、top-3 是别的文档，
    答案里就混进了"特征工程""自我介绍三十秒版本"这类完全跑题的句子。
    所以这里按排名卡死参与答案的块数，并跳过 Markdown 标题行。
    """
    hits = hits[:MAX_ANSWER_CHUNKS]
    q_tokens = set(tokenize_for_bm25(query))
    candidates = []

    for h in hits:
        for sent in split_sentences(h["text"]):
            # "## RAG 的标准流程" 这类标题是结构标记，不是答案内容
            if sent.startswith("#"):
                continue
            sc = _sentence_score(sent, q_tokens, h.get("score", 0.0))
            if sc > 0:
                candidates.append((sc, sent.strip(), h))

    # 按分数排序，同时做字面去重
    candidates.sort(key=lambda x: -x[0])
    picked, seen_sent, seen_chunk = [], set(), set()
    citations = []
    for sc, sent, h in candidates:
        key = sent[:40]
        if key in seen_sent:
            continue
        seen_sent.add(key)
        picked.append(sent)
        # 同一文本块只引用一次，避免答案下方出现重复的来源卡片
        if h["chunk_id"] not in seen_chunk:
            seen_chunk.add(h["chunk_id"])
            citations.append({
                "title": h["title"],
                "chunk_id": h["chunk_id"],
                "snippet": h["text"][:160].replace("\n", " "),
                "score": round(float(h.get("score", 0.0)), 4),
            })
        if len(picked) >= max_sentences:
            break

    if not picked:
        # 兜底：直接回显最高分的块
        if hits:
            top = hits[0]
            return top["text"][:400], [{
                "title": top["title"],
                "chunk_id": top["chunk_id"],
                "snippet": top["text"][:160],
                "score": round(float(top.get("score", 0.0)), 4),
            }]
        return "未检索到相关内容，请尝试更换关键词或补充更多文档。", []

    # 按引用顺序输出，读起来更连贯
    answer = "\n".join(f"· {s}" for s in picked)
    return answer, citations[:max_sentences]


def build_prompt(query, hits, max_chars=1800):
    """给大模型用的 RAG Prompt（llm 模式下使用）。"""
    ctx, used = [], 0
    for i, h in enumerate(hits, 1):
        piece = f"[{i}] 来源：{h['title']}\n{h['text']}\n"
        if used + len(piece) > max_chars:
            break
        ctx.append(piece)
        used += len(piece)
    context = "\n".join(ctx)
    return (
        "你是严谨的问答助手。请只依据下方资料回答，资料中没有的信息要说明无法回答。\n"
        "回答时请在句末用 [编号] 标注依据来源。\n\n"
        f"资料：\n{context}\n\n问题：{query}\n回答："
    )


# 默认 DeepSeek，但接口是 OpenAI 兼容格式，换成通义、Kimi、本地 Ollama 都能直接用
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class LLMClient:
    """OpenAI 兼容接口的极简客户端，未配置 key 时自动不可用。

    用法：
        client = LLMClient(api_key="sk-xxx")
        if client.available: client.chat(prompt)

    base_url 留空则用 DeepSeek；因为走的是 OpenAI 兼容协议，
    指向任何兼容端点（通义 / Kimi / 本地 Ollama / vLLM）都不需要改代码。
    未配置 key 时 available 为 False，上层会自动降级为抽取式答案。
    """

    def __init__(self, api_key=None, base_url=None,
                 model=None, timeout=30):
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    @property
    def available(self):
        return bool(self.api_key)

    def chat(self, prompt, temperature=0.2):
        import json
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
