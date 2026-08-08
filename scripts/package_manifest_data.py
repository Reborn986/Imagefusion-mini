from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


PATH_KEYS = (
    "infrared_degraded_path",
    "visible_degraded_path",
    "infrared_clean_path",
    "visible_clean_path",
    "fused_gt_path",
)


def safe_name(value: object, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value or "")).strip("_")
    return name[:180] or fallback


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect only files referenced by an ImageFusion manifest.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("--output-dir must be inside --project-root for portable relative paths") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    rows = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("input manifest must be a JSON list of objects")

    bundled: list[dict[str, Any]] = []
    checksums: list[dict[str, Any]] = []
    total_bytes = 0
    for index, raw in enumerate(rows):
        row = dict(raw)
        sample_id = safe_name(row.get("id"), f"sample_{index:06d}")
        for key in PATH_KEYS:
            source_value = row.get(key)
            if not source_value:
                raise KeyError(f"sample index={index} id={sample_id} missing {key}")
            source = Path(str(source_value)).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"sample index={index} id={sample_id} missing file: {source}")
            suffix = source.suffix.lower() or ".bin"
            destination = output_dir / "files" / sample_id / f"{key}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            relative = destination.relative_to(project_root)
            row[key] = relative.as_posix()
            size = destination.stat().st_size
            total_bytes += size
            checksums.append(
                {
                    "path": relative.as_posix(),
                    "bytes": size,
                    "sha256": file_sha256(destination),
                }
            )
        bundled.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(bundled, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "source_manifest": str(args.input_manifest.resolve()),
        "samples": len(bundled),
        "files": len(checksums),
        "bytes": total_bytes,
        "checksums": checksums,
    }
    (output_dir / "bundle_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[DONE] samples={len(bundled)} files={len(checksums)} "
        f"bytes={total_bytes} manifest={output_dir / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
