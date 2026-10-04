"""Rewrite absolute artifact paths in an MLflow file store after moving it to another machine.

    python tools/fix_mlruns_paths.py <mlruns_dir>
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/mlruns").resolve()
new_prefix = root.as_uri() + "/"
pat = re.compile(r"(artifact_(?:uri|location):\s*)file:/*.*?/mlruns/")
n = 0
for meta in root.rglob("meta.yaml"):
    try:
        txt = meta.read_text()
        new = pat.sub(lambda m: m.group(1) + new_prefix, txt)
        if new != txt:
            meta.write_text(new)
            n += 1
    except OSError:
        pass  # read-only mount: metrics/params still display
print(f"fixed {n} meta.yaml files under {root}")
