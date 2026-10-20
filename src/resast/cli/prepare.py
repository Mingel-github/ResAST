"""Verify the pinned source data and the published analysis-ID split."""

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from resast.data.dataset import load_clean_manifest
from resast.data.features import load_author_features
from resast.training.common import sha256, write_json


def verify_source(root, upstream, *, author_features=True, audio_dir=None):
    root, upstream = Path(root).resolve(), Path(upstream).resolve()
    provenance = json.loads((root / "data/provenance.json").read_text(encoding="utf-8"))
    for path, key in (("data/manifest.csv", "manifest_sha256"), ("splits/roles.csv", "roles_sha256")):
        if sha256(root / path) != provenance[key]:
            raise ValueError(f"Release metadata changed: {path}")
    revision = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != provenance["revision"]:
        raise ValueError("Upstream checkout is not the pinned revision")
    manifest = pd.read_csv(root / "data/manifest.csv", dtype=str).fillna("")
    audio_root = Path(audio_dir).resolve() if audio_dir is not None else upstream
    for row in manifest.itertuples(index=False):
        path = (audio_root / (row.filename if audio_dir is not None else row.local_relpath)).resolve()
        if not path.is_relative_to(audio_root) or sha256(path) != row.sha256:
            raise ValueError(f"Audio differs from the source manifest: {row.filename}")
    clean = load_clean_manifest(root / "data/manifest.csv")
    if len(clean) != 792 or clean.analysis_cat_id.nunique() != 111:
        raise ValueError("Analysis cohort differs")
    roles = pd.read_csv(root / "splits/roles.csv", dtype={"cat_id": str})
    if len(roles) != 1332 or set(roles.cat_id) != set(clean.analysis_cat_id):
        raise ValueError("Role bank differs from the cohort")
    counts = {}
    if author_features:
        config = json.loads((root / "configs/paper.json").read_text(encoding="utf-8"))
        for name in ("vggish129", "perch1281", "yamnet"):
            source = config["data"][name]
            path = upstream / "dataset/embeddings" / Path(source["path"]).name
            if sha256(path) != source["sha256"]:
                raise ValueError(f"Author feature checksum differs: {name}")
            counts[name] = len(load_author_features(path, name, roles.cat_id.unique()).labels)
    return {"status": "verified", "source_revision": revision, "audio_files": len(manifest),
            "analysis_calls": len(clean), "analysis_ids": clean.analysis_cat_id.nunique(),
            "author_feature_rows": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, help="Existing flat AudioCropped directory, if stored separately")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify_source(args.root, args.upstream, audio_dir=args.audio_dir)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
