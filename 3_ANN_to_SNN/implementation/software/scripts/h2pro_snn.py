#!/usr/bin/env python3
"""No-training H2-pro SNN foundation: audit, development diagnosis, mechanism demo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.h2pro_snn.foundation import ANN_ROOT, DEVELOPMENT_ROLES, audit, research_sources, write_report
from wangdao.h2pro_snn.transport import mechanism_demo

# A separate interpreter prevents the old and frozen wangdao.h2 packages from
# silently sharing sys.modules. ANN files are never copied or edited.
BOOTSTRAP = """import sys,runpy
ann,snn=sys.argv[1:3]
sys.argv=['h2pro_snn.ann_worker']+sys.argv[3:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
runpy.run_module('wangdao.h2pro_snn.ann_worker',run_name='__main__')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "mechanism-demo", "diagnose"):
        child = commands.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name == "diagnose":
            child.add_argument("--dataset-root", type=Path, default=ROOT / "data/EV-UAV-dataset")
            child.add_argument("--role", choices=DEVELOPMENT_ROLES, default="train_dev")
            child.add_argument("--limit", type=int)
            child.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new experiment path")
    if args.command == "audit":
        report = audit()
    elif args.command == "mechanism-demo":
        report = mechanism_demo()
        report['research_sources_utf8_lf'] = research_sources()
    else:
        if ((args.limit is not None and args.limit < 1) or
                (args.max_windows is not None and args.max_windows < 1)):
            parser.error("Limits must be positive integers")
        command = [sys.executable, "-B", "-c", BOOTSTRAP, str(ANN_ROOT / "software"),
                   str(ROOT / "software/wangdao"), "--dataset-root", str(args.dataset_root.absolute()),
                   "--role", args.role]
        for name in ("limit", "max_windows"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(command, cwd=str(ROOT), env=environment, encoding="utf-8",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            sys.stderr.write(completed.stderr)
            raise SystemExit(completed.returncode)
        report = json.loads(completed.stdout)
    write_report(args.output, report)
    print(json.dumps({"status": report["status"], "training_started": False,
                      "primary_metrics": report["primary_metrics"],
                      "output": str(args.output.absolute())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
