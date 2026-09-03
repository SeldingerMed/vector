"""Child process for the OR-Audit JSON plugin protocol."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from or_audit.eval.integrity import package_file
from or_audit.eval.plugins import load_entrypoint


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _jsonable(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _build_runtime(args: argparse.Namespace) -> Any:
    root = Path(args.root).resolve()
    factory = load_entrypoint(root, args.entrypoint, label=args.role)
    if args.role in {"policy", "predictor"}:
        weights = package_file(root, args.weights_path, label="weights")
        return factory(root=root, weights_path=weights)
    return factory(root=root)


def _dispatch(runtime: Any, role: str, op: str, payload: dict[str, Any]) -> Any:
    if op == "close":
        return None
    if role == "policy" and op == "reset":
        runtime.reset(seed=int(payload["seed"]))
        return None
    if role == "policy" and op == "act":
        return runtime.act(payload.get("observation"), step=int(payload["step"]))
    if role == "predictor" and op == "predict":
        return runtime.predict(payload["item"])
    if role == "verifier" and op == "score":
        return runtime.score(payload["context"])
    raise ValueError(f"role {role!r} does not implement operation {op!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("policy", "predictor", "verifier"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--weights-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        runtime = _build_runtime(args)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            op = str(request.get("op"))
            payload = request.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            with redirect_stdout(sys.stderr):
                result = _dispatch(runtime, args.role, op, payload)
            response = {"request_id": request_id, "ok": True, "result": _jsonable(result)}
        except Exception as exc:
            response = {
                "request_id": locals().get("request_id"),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        protocol_stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        protocol_stdout.flush()
        if locals().get("op") == "close":
            break
    close = getattr(runtime, "close", None)
    if callable(close):
        with redirect_stdout(sys.stderr):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
