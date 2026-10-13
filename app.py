# -*- coding: utf-8 -*-
"""RAG-Lite 一键入口。

用法：
    python app.py                      # 建索引并启动 Web 服务（自动打开浏览器）
    python app.py --ask "什么是过拟合"  # 命令行直接提问
    python app.py --eval               # 跑检索效果评测
    python app.py --port 9000          # 指定端口
"""
import argparse
import json
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pipeline import RagPipeline          # noqa: E402
from serve import make_server             # noqa: E402
from evaluate import run_evaluation       # noqa: E402
from tune import run_tune                 # noqa: E402

DEFAULT_DATA = os.path.join(ROOT, "data")
DEFAULT_WEB = os.path.join(ROOT, "web")
DEFAULT_QA = os.path.join(ROOT, "tests", "qa.json")


PLACEHOLDERS = ("在这里粘贴", "your", "xxx", "粘贴", "example")


def load_config():
    """读取项目根目录的 config.json；不存在或格式错误时返回空字典。"""
    path = os.path.join(ROOT, "config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[警告] config.json 读取失败（{exc}），已忽略该配置文件")
        return {}


def clean_key(key):
    """把占位符文本当作未配置处理，避免用户复制了模板却以为配好了。"""
    if not key:
        return None
    low = key.lower()
    if any(p in low for p in PLACEHOLDERS):
        return None
    return key.strip()


def build_pipeline(args):
    """Key 取值优先级：命令行 --llm-key > 环境变量 > config.json"""
    cfg = load_config()
    api_key = (clean_key(args.llm_key)
               or clean_key(os.environ.get("LLM_API_KEY"))
               or clean_key(cfg.get("llm_api_key")))
    base_url = (args.llm_base_url
                or os.environ.get("LLM_BASE_URL")
                or cfg.get("llm_base_url"))
    model = args.llm_model or cfg.get("llm_model")

    pipe = RagPipeline(data_dir=args.data, chunk_size=args.chunk_size,
                       overlap=args.overlap, llm_api_key=api_key,
                       llm_base_url=base_url, llm_model=model)
    return pipe


def cmd_ask(pipe, query, mode, top_k):
    res = pipe.answer(query, top_k=top_k, mode=mode)
    print("=" * 68)
    print(f"问题：{res['query']}    [检索模式] {res['mode']}    "
          f"[生成] {res['generation']}    [耗时] {res['latency_ms']}ms")
    print("=" * 68)
    print("\n答案：")
    print(res["answer"])
    if res["citations"]:
        print("\n来源：")
        for c in res["citations"]:
            print(f"  - 《{c['title']}》 (相关度 {c['score']})")
            print(f"    {c['snippet']}...")
    print()


def cmd_serve(pipe, host, port):
    st = pipe.stats()
    print("-" * 68)
    print("  RAG-Lite 本地知识库问答系统")
    print("-" * 68)
    print(f"  文档数：{st['doc_count']}    切块数：{st['chunk_count']}    "
          f"词表：{st['vocab_size']}    平均块长：{st['avg_chunk_len']} 字")
    if st["llm_enabled"]:
        print(f"  大模型生成：已启用（模型 {st.get('model', 'deepseek-chat')}）")
    else:
        cfg_path = os.path.join(ROOT, "config.json")
        hint = "未配置" if os.path.exists(cfg_path) else "未配置"
        print(f"  大模型生成：{hint}，当前使用抽取式答案")
    print(f"  服务地址：http://{host}:{port}")
    print("  按 Ctrl+C 退出")
    print("-" * 68)

    url = f"http://{host}:{port}"
    if host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server = make_server(pipe, DEFAULT_WEB, host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.server_close()


def main():
    p = argparse.ArgumentParser(description="RAG-Lite：零依赖本地知识库问答系统")
    p.add_argument("--data", default=DEFAULT_DATA, help="知识库目录")
    p.add_argument("--qa", default=DEFAULT_QA, help="评测集路径")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=320)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--ask", help="命令行直接提问")
    p.add_argument("--eval", action="store_true", help="运行检索评测")
    p.add_argument("--tune", action="store_true", help="划分 dev/test 并搜索融合权重")
    p.add_argument("--llm-key", default=None,
                   help="可选：大模型 API Key，填入后启用大模型生成答案。"
                        "不填则回落到环境变量 LLM_API_KEY")
    p.add_argument("--llm-base-url", default=None,
                   help="可选：大模型接口地址，兼容任何 OpenAI 格式端点"
                        "（默认 DeepSeek，也可指向通义/Kimi/本地 Ollama）")
    p.add_argument("--llm-model", default=None, help="可选：模型名，默认 deepseek-chat")
    args = p.parse_args()

    pipe = build_pipeline(args)

    if args.tune:
        run_tune(pipe, args.qa)
        return
    if args.eval:
        run_evaluation(pipe, args.qa)
        return
    if args.ask:
        cmd_ask(pipe, args.ask, args.mode, args.top_k)
        return
    cmd_serve(pipe, args.host, args.port)


if __name__ == "__main__":
    main()
