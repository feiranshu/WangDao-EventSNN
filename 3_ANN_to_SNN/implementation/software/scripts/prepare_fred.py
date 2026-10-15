"""Independent FRED download/adaptation entry point (no model training)."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    download = sub.add_parser("download")
    download.add_argument("--directory", type=Path, required=True)
    download.add_argument("--sequences", nargs="+", type=int, default=[0, 1])
    download.add_argument("--split", choices=["train", "test"], default="train")
    convert = sub.add_parser("adapt")
    convert.add_argument("--downloads", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--runtime", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--runtime", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    preview = sub.add_parser("preview")
    preview.add_argument("--dataset", type=Path, required=True)
    preview.add_argument("--downloads", type=Path, required=True)
    preview.add_argument("--runtime", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "download":
        from wangdao.fred_adapter_v1.download import download_sequences
        result = download_sequences(args.directory, args.sequences, args.split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.action == "adapt":
        from wangdao.fred_adapter_v1.workflow import adapt
        adapt(args.downloads, args.output, args.runtime)
    elif args.action == "verify":
        from wangdao.fred_adapter_v1.verify import verify_dataset
        verify_dataset(args.dataset, args.runtime, args.report)
    elif args.action == "preview":
        from wangdao.fred_adapter_v1.verify import make_previews
        make_previews(args.dataset, args.downloads, args.runtime, args.output)


if __name__ == "__main__":
    main()
