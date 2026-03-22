#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warm-start an EQNet runtime from an inner_os transfer package and optionally
prepare a model-swap bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-start runtime state from an inner_os transfer package and optionally prepare a model swap bundle."
    )
    parser.add_argument(
        "--transfer-package",
        required=True,
        help="Path to the transfer package JSON file.",
    )
    parser.add_argument(
        "--target-model",
        default="",
        help="Preferred target model to set in model cache for the next runtime.",
    )
    parser.add_argument(
        "--target-base-url",
        default="",
        help="Optional base URL for the target model endpoint.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional path to write the warm-start summary JSON.",
    )
    parser.add_argument(
        "--bundle-out",
        default="",
        help="Optional path to write the model swap bundle JSON.",
    )
    parser.add_argument(
        "--persist-normalized",
        action="store_true",
        help="Rewrite the input package as normalized v1 after loading.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    transfer_package_path = Path(args.transfer_package)
    if not transfer_package_path.exists():
        raise FileNotFoundError(f"transfer package not found: {transfer_package_path}")
    if args.bundle_out and not args.target_model:
        raise ValueError("--bundle-out requires --target-model")

    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=False))
    if args.persist_normalized:
        runtime._transfer_package_path = transfer_package_path

    summary = runtime.warm_start_from_transfer_package(
        transfer_package_path,
        target_model=args.target_model,
        target_base_url=args.target_base_url,
        persist_normalized=args.persist_normalized,
    )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.bundle_out:
        bundle = runtime.build_inner_os_model_swap_bundle(
            target_model=args.target_model,
            target_base_url=args.target_base_url,
        )
        bundle_path = Path(args.bundle_out)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
