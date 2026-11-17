"""Matplotlib charts of evaluation quality."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.analytics.frames import SCORE_COLUMNS  # noqa: E402
from app.utils.paths import CHARTS_DIR  # noqa: E402

DPI: int = 150
HIST_BINS: int = 10
PALETTE: tuple[str, ...] = ("#1F3864", "#2E75B6", "#70AD47", "#ED7D31", "#7030A0", "#C00000", "#FFC000")
CHART_FILES: dict[str, str] = {
    "language_distribution": "language_distribution.png",
    "quality_score_distribution": "quality_score_distribution.png",
    "hallucination_distribution": "hallucination_distribution.png",
    "safety_score_distribution": "safety_score_distribution.png",
    "category_distribution": "category_distribution.png",
    "correlation_heatmap": "correlation_heatmap.png",
}
HEATMAP_LABELS: dict[str, str] = {
    "relevance_score": "Relevance",
    "correctness_score": "Correctness",
    "clarity_score": "Clarity",
    "safety_score": "Safety",
    "hallucination_score": "Hallucination",
    "duplicate_percentage": "Duplicate %",
    "overall_score": "Overall",
}


class ChartGenerator:
    """Creates the standard set of analytics charts from a score frame."""

    def __init__(self, output_dir: Path = CHARTS_DIR) -> None:
        self._output_dir = output_dir

    def generate_all(self, frame: pd.DataFrame) -> dict[str, Path]:
        """Render every chart and return ``{name: path}``."""
        if frame.empty:
            raise ValueError("There is no evaluation data to chart.")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "language_distribution": self._bar(frame, "language", "Language distribution", "Language", "language_distribution"),
            "quality_score_distribution": self._hist(frame, "overall_score", "Quality (overall) score distribution", "Overall score", "quality_score_distribution"),
            "hallucination_distribution": self._hist(frame, "hallucination_score", "Hallucination risk distribution", "Hallucination risk score", "hallucination_distribution"),
            "safety_score_distribution": self._hist(frame, "safety_score", "Safety score distribution", "Safety score", "safety_score_distribution"),
            "category_distribution": self._bar(frame, "category", "Category distribution", "Category", "category_distribution"),
            "correlation_heatmap": self._heatmap(frame),
        }

    def _save(self, fig, key: str) -> Path:
        path = self._output_dir / CHART_FILES[key]
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return path

    def _bar(self, frame: pd.DataFrame, column: str, title: str, xlabel: str, key: str) -> Path:
        counts = frame[column].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[: len(counts)] or PALETTE[0])
        ax.bar_label(bars, padding=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Evaluations")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        return self._save(fig, key)

    def _hist(self, frame: pd.DataFrame, column: str, title: str, xlabel: str, key: str) -> Path:
        values = frame[column].astype(float)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(values, bins=np.linspace(0, 100, HIST_BINS + 1), color=PALETTE[1], edgecolor="white")
        ax.axvline(values.mean(), color=PALETTE[5], linestyle="--", label=f"Mean {values.mean():.1f}")
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Evaluations")
        ax.legend()
        return self._save(fig, key)

    def _heatmap(self, frame: pd.DataFrame) -> Path:
        columns = [c for c in SCORE_COLUMNS if c in frame.columns]
        correlation = frame[columns].astype(float).corr().fillna(0.0)
        labels = [HEATMAP_LABELS[c] for c in columns]
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        image = ax.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for (row, col), value in np.ndenumerate(correlation.values):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "black", fontsize=8)
        ax.set_title("Correlation between quality metrics")
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        return self._save(fig, "correlation_heatmap")
