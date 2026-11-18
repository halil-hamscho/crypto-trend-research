"""Check every sealed manifest's pinned files: run from the repository root, `python verify_manifests.py`.

Each sealed run fingerprinted (SHA256) the code, configs and specifications it ran with. A file that later
stages changed is kept at the path it had, under reports/pinned_versions/<sha256>/, recovered byte for byte
from the private repository's git history. Exit code 1 if any pin is unmatched.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pins(manifest: dict) -> dict:
    out = {}
    for key in ("code_sha256", "configs"):
        out.update(manifest.get(key) or {})
    for key, value in manifest.items():
        if key.endswith("_specification") and isinstance(value, dict) and value.get("path") and value.get("sha256"):
            out[value["path"]] = value["sha256"]
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


failures = 0
for path in sorted((ROOT / "reports").rglob("manifest.json")):
    manifest = json.loads(path.read_text())
    current = archived = 0
    for rel, digest in pins(manifest).items():
        if (ROOT / rel).exists() and sha256(ROOT / rel) == digest:
            current += 1
        elif (ROOT / "reports" / "pinned_versions" / digest / rel).exists() and \
                sha256(ROOT / "reports" / "pinned_versions" / digest / rel) == digest:
            archived += 1
        else:
            failures += 1
            print(f"UNMATCHED {path.parent.relative_to(ROOT)}: {rel} {digest[:12]}")
    print(f"{path.parent.relative_to(ROOT)}: {current} pins match the tree, {archived} match an archived version")
print("all pins matched" if not failures else f"{failures} unmatched pin(s)")
sys.exit(1 if failures else 0)
