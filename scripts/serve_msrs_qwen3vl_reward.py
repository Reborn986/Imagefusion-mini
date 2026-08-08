#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefusion_r1.rl.qwen3vl_reward_judge import (
    DEFAULT_QWEN3VL8B_PATH,
    JudgeInput,
    Qwen3VLImageRewardJudge,
)


def _field(payload: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value is not None and str(value):
            return str(value)
    return ""


def _make_judge_input(payload: Mapping[str, Any]) -> JudgeInput:
    return JudgeInput(
        infrared_degraded_path=_field(payload, "infrared_degraded_path", "ir_path"),
        visible_degraded_path=_field(payload, "visible_degraded_path", "vis_path"),
        fused_image_path=_field(payload, "fused_image_path", "fused_path"),
        infrared_label=_field(payload, "infrared_label", "ir_label"),
        visible_label=_field(payload, "visible_label", "vis_label"),
        sample_id=_field(payload, "sample_id", "id"),
    )


def _require_nonempty(item: JudgeInput) -> None:
    missing = []
    for name in (
        "infrared_degraded_path",
        "visible_degraded_path",
        "fused_image_path",
        "infrared_label",
        "visible_label",
    ):
        if not getattr(item, name):
            missing.append(name)
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")


class RewardServer:
    def __init__(self, judge: Qwen3VLImageRewardJudge) -> None:
        self.judge = judge
        # vLLM's synchronous LLM object is not a per-request thread-safe API.
        # The HTTP server may receive one request from every policy rank at once,
        # so serialize calls instead of racing the shared engine state.
        self.judge_lock = threading.Lock()

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MSRSQwen3VLReward/0.1"

            def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _read_json(self) -> Mapping[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("empty request body")
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("request body must be a JSON object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/health":
                    self._send_json(404, {"ok": False, "error": "unknown endpoint"})
                    return
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "model_loaded": outer.judge.llm is not None,
                        "model_path": outer.judge.model_path,
                    },
                )

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/score":
                    self._send_json(404, {"ok": False, "error": "unknown endpoint"})
                    return
                try:
                    payload = self._read_json()
                    raw_items = payload.get("items", payload.get("samples"))
                    if raw_items is None:
                        raw_items = [payload]
                    if not isinstance(raw_items, list):
                        raise ValueError("'items' must be a list when provided")

                    items = []
                    for raw_item in raw_items:
                        if not isinstance(raw_item, Mapping):
                            raise ValueError("each item must be a JSON object")
                        item = _make_judge_input(raw_item)
                        _require_nonempty(item)
                        items.append(item)

                    lock_started = time.time()
                    print(
                        f"[reward-server] request received items={len(items)} "
                        f"sample_ids={[item.sample_id for item in items]}",
                        flush=True,
                    )
                    with outer.judge_lock:
                        print(
                            f"[reward-server] judge lock acquired wait_sec={time.time() - lock_started:.2f} "
                            f"items={len(items)}",
                            flush=True,
                        )
                        results = outer.judge.score_batch(items)
                    print(
                        f"[reward-server] request scored items={len(results)} "
                        f"elapsed_sec={time.time() - lock_started:.2f}",
                        flush=True,
                    )
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "num_items": len(results),
                            "results": [asdict(result) for result in results],
                        },
                    )
                except Exception as exc:  # keep the server alive for the trainer
                    self._send_json(400, {"ok": False, "error": repr(exc)})

            def log_message(self, fmt: str, *args: Any) -> None:
                sys.stderr.write("[reward-server] " + fmt % args + "\n")

        return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen3-VL-8B MSRS image reward over HTTP.")
    parser.add_argument("--model-path", default=DEFAULT_QWEN3VL8B_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--load-on-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = Qwen3VLImageRewardJudge(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        enforce_eager=args.enforce_eager,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
    )
    if args.load_on_start:
        judge.load()

    server = ThreadingHTTPServer((args.host, args.port), RewardServer(judge).handler_class())
    print(
        f"[reward-server] serving Qwen3-VL reward at http://{args.host}:{args.port} "
        f"model={args.model_path}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
