"""Run the bounded independent temporal segmentation study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.temporal_study import main

if __name__ == "__main__":
    main()
