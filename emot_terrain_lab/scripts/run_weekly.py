# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from terrain.system import EmotionalMemorySystem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, default="data/state")
    args = ap.parse_args()
    sys = EmotionalMemorySystem(args.state)
    sys.weekly_abstraction()

if __name__ == "__main__":
    main()
