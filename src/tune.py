# -*- coding: utf-8 -*-
"""融合权重调参：在 dev 集上网格搜索，在 test 集上报告最终指标。

为什么要划分 dev / test：
直接在全量评测集上搜超参，相当于用测试集调参，指标会偏乐观，
这是初学者项目里最常见也最容易被面试官抓到的漏洞。
这里采用固定随机种子做 6:4 划分，权重只在 dev 上选，最终数字来自未参与调参的 test 集。
"""
import json
import os
import random
from itertools import product

import pipeline as P
from evaluate import evaluate

GRID = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
SEED = 42
DEV_RATIO = 0.6


def split(qa_items, seed=SEED, dev_ratio=DEV_RATIO):
    idx = list(range(len(qa_items)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * dev_ratio)
    dev = [qa_items[i] for i in idx[:cut]]
    test = [qa_items[i] for i in idx[cut:]]
    return dev, test


def search_weights(pipe, dev):
    """在 dev 集上搜索，综合指标 = MRR + 0.3*Recall@1 + 0.3*Recall@5"""
    results = []
    for wb, wv in product(GRID, repeat=2):
        P.FUSION_WEIGHTS = [wb, wv]
        r = evaluate(pipe, dev)["hybrid"]
        score = r["MRR"] + 0.3 * r["Recall@1"] + 0.3 * r["Recall@5"]
        results.append((score, (wb, wv), r))
    results.sort(key=lambda x: -x[0])
    return results


def run_tune(pipe, qa_path):
    with open(qa_path, "r", encoding="utf-8") as f:
        qa = json.load(f)
    dev, test = split(qa)

    print("\n" + "=" * 68)
    print(f"融合权重调参    总样本 {len(qa)} = dev {len(dev)} + test {len(test)}")
    print("=" * 68)

    results = search_weights(pipe, dev)
    print("\n[dev 集] 权重搜索 Top 5")
    print(f"{'w_bm25':>8}{'w_vec':>8}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}{'综合':>9}")
    print("-" * 68)
    for score, (wb, wv), r in results[:5]:
        print(f"{wb:>8}{wv:>8}{r['Recall@1']:>9.4f}{r['Recall@3']:>9.4f}"
              f"{r['Recall@5']:>9.4f}{r['MRR']:>9.4f}{score:>9.4f}")

    best_w = results[0][1]
    P.FUSION_WEIGHTS = list(best_w)

    print(f"\n[test 集] 采用权重 BM25={best_w[0]} / Vector={best_w[1]}（未参与调参）")
    test_report = evaluate(pipe, test)
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    print(f"{'模式':<10}" + "".join(f"{m:>14}" for m in metrics))
    print("-" * 68)
    for mode in ("bm25", "vector", "hybrid"):
        print(f"{mode:<10}" + "".join(f"{test_report[mode][m]:>14.4f}" for m in metrics))
    print("-" * 68)

    b, h = test_report["bm25"], test_report["hybrid"]
    print(f"混合检索 vs 纯 BM25   ：MRR {b['MRR']:.4f} -> {h['MRR']:.4f} "
          f"({(h['MRR'] - b['MRR']) / b['MRR'] * 100:+.1f}%)，"
          f"Recall@5 {b['Recall@5']:.4f} -> {h['Recall@5']:.4f} "
          f"({(h['Recall@5'] - b['Recall@5']) / b['Recall@5'] * 100:+.1f}%)")
    print(f"混合检索 vs 纯向量    ：MRR {test_report['vector']['MRR']:.4f} -> {h['MRR']:.4f}")
    print("=" * 68 + "\n")
    return best_w, test_report
