#!/usr/bin/env python3
"""Check the lightweight UNIV runtime environment."""
from __future__ import annotations

import importlib.util
import platform
from pathlib import Path


def check_import(module: str, package: str | None = None) -> tuple[str, bool, str]:
    try:
        imported = __import__(module)
        version = getattr(imported, "__version__", "unknown")
        return package or module, True, str(version)
    except Exception as exc:  # environment diagnostic script: report all failures
        return package or module, False, str(exc)


def main() -> int:
    print("UNIV environment check", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    torch_ok = importlib.util.find_spec("torch") is not None
    if torch_ok:
        import torch
        print(f"torch: {torch.__version__}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}", flush=True)
    else:
        print("torch: NOT INSTALLED", flush=True)

    checks = [
        ("timm", None), ("einops", None), ("cv2", "opencv-python"),
        ("PIL", "pillow"), ("yaml", "pyyaml"),
    ]
    failed = []
    for module, package in checks:
        name, ok, detail = check_import(module, package)
        status = "OK" if ok else "MISSING"
        print(f"{name}: {status} ({detail})", flush=True)
        if not ok:
            failed.append(name)

    ckpt = Path(__file__).resolve().parents[1] / "pretrained" / "checkpoint0400.pth"
    print(f"checkpoint0400.pth: {'FOUND' if ckpt.exists() else 'not found'} ({ckpt})", flush=True)
    print("Summary: " + ("PASS" if not failed and torch_ok else f"CHECK DEPENDENCIES: {failed}"), flush=True)
    return 0 if torch_ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
