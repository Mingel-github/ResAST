"""Evaluate saved predictions or combine four held-out folds."""

import argparse
import json
from pathlib import Path

from resast.data.dataset import load_roles
from resast.evaluation.metrics import collect_oof, metrics, read_predictions, summarize
from resast.training.common import sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--roles", type=Path, help="With this option, supply fold 0,1,2,3 predictions in that order")
    parser.add_argument("--repeat", type=int, choices=range(3))
    parser.add_argument("--model", help="Model name for OOF provenance")
    parser.add_argument("--seed", type=int, choices=(17, 43, 101))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    frames = [read_predictions(path) for path in args.predictions]
    if args.roles is not None:
        if args.repeat is None or args.model is None or args.seed is None or len(frames) != 4:
            parser.error("OOF collection needs --model, --seed, --repeat and four fold files")
        combined = collect_oof(dict(enumerate(frames)), load_roles(args.roles), args.repeat)
        result = metrics(combined)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            combined.to_csv(args.output / "oof_predictions.csv", index=False, float_format="%.17g")
            write_json(args.output / "oof.json", {
                "model": args.model, "repeat": args.repeat, "seed": args.seed,
                "roles_sha256": sha256(args.roles),
                "predictions_sha256": sha256(args.output / "oof_predictions.csv"),
                "fold_sources": [{"path": str(p), "sha256": sha256(p)} for p in args.predictions],
            })
    else:
        scores = [metrics(frame) for frame in frames]
        result = {"scores": scores}
        if len(scores) > 1:
            result["summary"] = summarize(scores)
    if args.output:
        write_json(args.output / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
