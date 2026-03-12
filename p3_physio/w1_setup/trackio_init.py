"""
W1: Trackio experiment tracking initialization and helper utilities.

Provides a unified logging interface for P3. Falls back to CSV if Trackio unavailable.
Import this in all training/eval scripts.

Usage (in other scripts):
    from w1_setup.trackio_init import ExperimentLogger
    logger = ExperimentLogger(project="p3_physio_deepfake", run_name="w3_initial_train")
    logger.log({"auc": 0.85, "eer": 0.12, "epoch": 1})
    logger.log_table("ablation_results", ablation_df)
    logger.finish()
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Try trackio
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False


class ExperimentLogger:
    """
    Unified experiment logger — uses Trackio when available, falls back to CSV + JSON.
    Compatible with Google Colab and Kaggle (no persistent server required).
    """

    def __init__(
        self,
        project: str = "p3_physio_deepfake",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        local_log_dir: str = "./logs",
    ):
        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.local_log_dir = Path(local_log_dir)
        self.local_log_dir.mkdir(parents=True, exist_ok=True)

        self._step = 0
        self._csv_path = self.local_log_dir / f"{self.run_name}_metrics.csv"
        self._csv_writer = None
        self._csv_file = None

        if TRACKIO_AVAILABLE:
            trackio.init(project=project, name=self.run_name, config=config, space_id="GoodBoyXD/bioforensics")
            print(f"[Trackio] Initialized run '{self.run_name}' in project '{project}'")
            print(f"          Dashboard: check trackio output above for URL")
        else:
            print(f"[Logger] Trackio unavailable — logging to {self._csv_path}")
            print(f"         Install: pip install trackio")

        # Always save config locally
        config_path = self.local_log_dir / f"{self.run_name}_config.json"
        with open(config_path, "w") as f:
            json.dump({"project": project, "run_name": self.run_name, **self.config}, f, indent=2)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dict of scalar metrics."""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        # Add step to metrics
        metrics_with_step = {"step": self._step, **metrics}

        if TRACKIO_AVAILABLE:
            trackio.log(metrics, step=self._step)

        # Always log to CSV as backup
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(metrics_with_step.keys()))
            self._csv_writer.writeheader()

        # Handle new keys appearing mid-run
        for key in metrics_with_step:
            if key not in self._csv_writer.fieldnames:
                self._csv_writer.fieldnames.append(key)

        self._csv_writer.writerow(metrics_with_step)
        self._csv_file.flush()

    def log_table(self, table_name: str, data: Any):
        """Log a pandas DataFrame as a table (local JSON; Trackio doesn't have tables)."""
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            out = self.local_log_dir / f"{self.run_name}_{table_name}.csv"
            data.to_csv(out, index=False)
            print(f"[Logger] Table '{table_name}' saved → {out}")

    def log_image(self, name: str, image_path: str):
        """Log an image path (Trackio may support images in future versions)."""
        print(f"[Logger] Image artifact: {name} → {image_path}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log final summary metrics (end of run)."""
        self.log(metrics, step=self._step)
        summary_path = self.local_log_dir / f"{self.run_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Logger] Summary saved → {summary_path}")

    def finish(self):
        """Close logger."""
        if self._csv_file:
            self._csv_file.close()
        if TRACKIO_AVAILABLE:
            try:
                trackio.finish()
            except Exception:
                pass
        print(f"[Logger] Run '{self.run_name}' finished.")


# ─── Convenience functions for quick logging ──────────────────────────────────

_global_logger: Optional[ExperimentLogger] = None


def init(project: str = "p3_physio_deepfake", run_name: str = None, config: dict = None):
    global _global_logger
    _global_logger = ExperimentLogger(project=project, run_name=run_name, config=config)
    return _global_logger


def log(metrics: dict, step: int = None):
    if _global_logger:
        _global_logger.log(metrics, step=step)
    else:
        print(f"[Logger WARN] Not initialized. Call trackio_init.init() first. Metrics: {metrics}")


def finish():
    if _global_logger:
        _global_logger.finish()


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ExperimentLogger...")
    logger = ExperimentLogger(
        project="p3_physio_deepfake",
        run_name="test_run",
        config={"lr": 1e-4, "batch_size": 8},
        local_log_dir="./logs",
    )
    for i in range(5):
        logger.log({"loss": 1.0 - i * 0.1, "auc": 0.5 + i * 0.08, "epoch": i})

    logger.log_summary({"final_auc": 0.90, "final_eer": 0.10})
    logger.finish()
    print("Logger test complete.")
