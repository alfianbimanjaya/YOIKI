# -*- coding: utf-8 -*-
"""YOIKI H1 V2.6 — one-shot GitHub Actions runner (NO GPG)."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# GitHub Secrets -> env vars; local telegram_config is optional and not needed here.
os.environ["TELEGRAM_SEND_NO_SIGNAL"] = "false"
os.environ["TELEGRAM_POLL_SECONDS"] = "30"

import _EXECUTE_Arealtime_telegram_h1_v26_USE_EXECUTE_SCREENER as bot

# Use compact public seed instead of the private/local raw file.
PUBLIC_SEED = ROOT / "v26_initial_history.parquet"
if not PUBLIC_SEED.exists():
    raise FileNotFoundError(f"Missing {PUBLIC_SEED}")

bot.live.RAW_FILE = PUBLIC_SEED
bot.LOG_DIR = ROOT / "cloud_runtime"
bot.STATE_FILE = bot.LOG_DIR / "telegram_runtime_state_v26.json"


def main():
    bot.init_dirs()
    print("=" * 100)
    print("YOIKI H1 V2.6 — PUBLIC GITHUB ACTIONS / ONE SHOT")
    print("=" * 100)

    bot.telegram_test_connection()
    bot.ensure_artifacts()
    bot.verify_model_metadata()

    model, calibrator, feature_cols = bot.load_model()
    master = pd.read_parquet(PUBLIC_SEED)
    universe = bot.load_universe()
    state = bot.load_state()

    print(f"Universe: {len(universe)}")
    print(f"Public seed rows: {len(master):,}")
    print(f"Threshold: P >= {bot.FROZEN_THRESHOLD:.4f}")
    print(f"Signal gap: <= {bot.FROZEN_SIGNAL_GAP_1H * 100:.2f}%")
    print(f"Execution gap: <= {bot.MAX_ENTRY_GAP_PCT * 100:.2f}%")

    # Exactly one frozen V2.6 scan. Then the runner exits.
    bot.screen_once(
        state=state,
        master=master,
        universe=universe,
        model=model,
        calibrator=calibrator,
        feature_cols=feature_cols,
    )

    print("GitHub Actions run finished.")


if __name__ == "__main__":
    main()
