# -*- coding: utf-8 -*-
"""语料加载：支持 .txt / .md，若环境装了 pypdf 则自动支持 .pdf。

零依赖设计：PDF 解析是"可选增强"，没装 pypdf 时跳过 .pdf 并给出提示，
不会因为缺少第三方包导致整个项目起不来。
"""
import os

SUPPORTED = (".txt", ".md", ".markdown")


def read_text_file(path):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return None
    try:
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return None


def load_corpus(data_dir):
    """返回 list[(doc_id, title, text)]。"""
    docs = []
    if not os.path.isdir(data_dir):
        return docs

    files = sorted(f for f in os.listdir(data_dir)
                   if os.path.isfile(os.path.join(data_dir, f)))
    for fname in files:
        path = os.path.join(data_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".pdf":
            text = read_pdf(path)
        elif ext in SUPPORTED:
            text = read_text_file(path)
        else:
            continue
        if not text or not text.strip():
            continue
        title = os.path.splitext(fname)[0]
        docs.append((fname, title, text.strip()))
    return docs


def load_demo_docs():
    """未准备语料时的兜底内容，保证系统永远有东西可查。"""
    return [(
        "demo.md", "RAG 简介",
        "检索增强生成（Retrieval-Augmented Generation, RAG）是一种把信息检索与"
        "大模型生成结合的技术范式。它先从外部知识库中检索与问题相关的片段，"
        "再把这些片段作为上下文喂给大模型，让模型基于给定资料作答。\n"
        "RAG 的主要优势是缓解大模型幻觉、支持知识实时更新、答案可溯源。\n"
        "典型流程包括：文档解析、切分、向量化与索引、召回、重排、生成。\n"
        "常见检索方案有稀疏检索（BM25）与稠密检索（向量相似度）两类，"
        "工业界通常使用混合检索并通过 RRF 融合多路结果。",
    )]
