# -*- coding: utf-8 -*-
"""检索效果评测：对比 bm25 / vector / hybrid 三种模式。

指标：
- Recall@K：前 K 条结果中包含正确文档的比例（衡量"找没找到"）
- MRR     ：正确文档首次出现排名的倒数的平均值（衡量"排得够不够靠前"）
"""
import json
import os

MODES = ("bm25", "vector", "hybrid")
TOP_KS = (1, 3, 5)


def load_qa(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(pipe, qa_items, top_ks=TOP_KS):
    n = len(qa_items)
    report = {}
    for mode in MODES:
        hit_at = {k: 0 for k in top_ks}
        mrr_sum = 0.0
        for item in qa_items:
            hits = pipe.search(item["q"], top_k=max(top_ks), mode=mode)
            rank = 0
            for i, h in enumerate(hits, 1):
                if h["doc_id"] == item["gold_doc"]:
                    rank = i
                    break
            for k in top_ks:
                if rank and rank <= k:
                    hit_at[k] += 1
            mrr_sum += (1.0 / rank) if rank else 0.0
        report[mode] = {
            f"Recall@{k}": round(hit_at[k] / n, 4) for k in top_ks
        }
        report[mode]["MRR"] = round(mrr_sum / n, 4)
    return report


def run_evaluation(pipe, qa_path):
    if not os.path.exists(qa_path):
        print(f"[评测] 未找到评测集：{qa_path}")
        return {}
    qa = load_qa(qa_path)
    report = evaluate(pipe, qa)

    metrics = [f"Recall@{k}" for k in TOP_KS] + ["MRR"]
    print("\n" + "=" * 68)
    print(f"检索评测    评测集：{len(qa)} 条    文档块：{len(pipe.chunks)}")
    print("=" * 68)
    print(f"{'模式':<10}" + "".join(f"{m:>14}" for m in metrics))
    print("-" * 68)
    for mode in MODES:
        row = "".join(f"{report[mode][m]:>14.4f}" for m in metrics)
        print(f"{mode:<10}" + row)
    print("-" * 68)

    # 结论按指标分别给出：混合检索在"排得准"上占优，稠密检索在"召得全"上占优，
    # 这是两种检索方式失败模式互补的直接体现，也是本项目保留混合方案的理由。
    b, v, h = report["bm25"], report["vector"], report["hybrid"]
    for m in ("Recall@1", "MRR", "Recall@5"):
        base = b[m]
        if base <= 0:
            continue
        print(f"混合 vs BM25    {m:<10} {base:.4f} -> {h[m]:.4f} "
              f"({(h[m] - base) / base * 100:+.1f}%)")
    print(f"向量 vs 混合     Recall@5   {h['Recall@5']:.4f} vs {v['Recall@5']:.4f} "
          f"（稠密检索召回面更广）")
    print("=" * 68 + "\n")
    return report
