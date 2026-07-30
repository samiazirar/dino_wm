#!/usr/bin/env python3
"""Inventory and permanently delete an exact Rope/Granular bad-depth allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_target(project: Path, target: dict[str, Any]) -> dict[str, Any]:
    path = Path(target["path"])
    if not path.is_absolute() or path == project:
        raise RuntimeError(f"unsafe target path: {path}")
    try:
        path.relative_to(project)
    except ValueError as exc:
        raise RuntimeError(f"target escapes project root: {path}") from exc
    if path.is_symlink():
        raise RuntimeError(f"symlink target is forbidden: {path}")
    entries = []
    logical_bytes = allocated_bytes = 0
    if not path.exists():
        raise RuntimeError(f"target is absent: {path}")
    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    for item in paths:
        relative = item.relative_to(path).as_posix() if item != path else "."
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink inside target is forbidden: {item}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            digest = None
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            digest = sha256_file(item)
            logical_bytes += metadata.st_size
            allocated_bytes += metadata.st_blocks * 512
        else:
            raise RuntimeError(f"special file inside target is forbidden: {item}")
        entries.append(
            {
                "relative_path": relative,
                "kind": kind,
                "bytes": metadata.st_size if kind == "file" else 0,
                "sha256": digest,
            }
        )
    tree_sha256 = hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "target_type": "file" if path.is_file() else "directory",
        "causal_class": target["causal_class"],
        "live_consumer": target["live_consumer"],
        "file_count": sum(entry["kind"] == "file" for entry in entries),
        "directory_count": sum(
            entry["kind"] == "directory" for entry in entries
        ),
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "tree_sha256": tree_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "delete"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    project = Path(spec["project_root"])
    if not project.is_absolute() or not project.is_dir():
        raise RuntimeError("project root is not an existing absolute directory")
    if not spec.get("targets"):
        raise RuntimeError("deletion allowlist is empty")
    paths = [target["path"] for target in spec["targets"]]
    if len(paths) != len(set(paths)):
        raise RuntimeError("deletion allowlist contains duplicate paths")
    resolved = [inventory_target(project, target) for target in spec["targets"]]
    if any(item["live_consumer"] != "none" for item in resolved):
        raise RuntimeError("a deletion target still has a live consumer")
    if args.command == "inventory":
        receipt = {
            "schema": "dinocular.rg-bad-depth-deletion-inventory.v1",
            "state": "RESOLVED_NOT_DELETED",
            "project_root": str(project),
            "target_count": len(resolved),
            "total_logical_bytes": sum(
                item["logical_bytes"] for item in resolved
            ),
            "total_allocated_bytes": sum(
                item["allocated_bytes"] for item in resolved
            ),
            "targets": resolved,
        }
    else:
        if args.inventory is None:
            raise RuntimeError("delete requires --inventory")
        prior = json.loads(args.inventory.read_text())
        if (
            prior.get("state") != "RESOLVED_NOT_DELETED"
            or prior.get("targets") != resolved
        ):
            raise RuntimeError("current targets differ from the resolved inventory")
        for item in resolved:
            path = Path(item["path"])
            if item["target_type"] == "directory":
                shutil.rmtree(path)
            else:
                path.unlink()
        remaining = [item["path"] for item in resolved if Path(item["path"]).exists()]
        if remaining:
            raise RuntimeError(f"deletion confirmation failed: {remaining}")
        receipt = {
            "schema": "dinocular.rg-bad-depth-deletion-receipt.v1",
            "state": "PERMANENTLY_DELETED",
            "project_root": str(project),
            "target_count": len(resolved),
            "total_logical_bytes": sum(
                item["logical_bytes"] for item in resolved
            ),
            "total_allocated_bytes": sum(
                item["allocated_bytes"] for item in resolved
            ),
            "retained_bad_depth_bytes": False,
            "archive_created": False,
            "targets": [
                {
                    **item,
                    "deletion_confirmed": True,
                }
                for item in resolved
            ],
        }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
