"""Write run artifacts to disk."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Exporter(ABC):
    """Writes one kind of artifact into an output directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def _path(self, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / name

    @abstractmethod
    def export(self, data: Any, name: str) -> Path:
        """Write ``data`` and return the path of the created file."""


class VideoExporter(Exporter):
    """Encodes frames (HxWx3 uint8 arrays) to H.264 MP4 via imageio-ffmpeg."""

    def __init__(self, output_dir: str | Path, fps: int = 20) -> None:
        super().__init__(output_dir)
        self.fps = fps

    def open(self, name: str, fps: int | None = None):
        """Return a streaming writer (``append_data(frame)``, ``close()``)."""
        import imageio

        return imageio.get_writer(
            self._path(name),
            fps=fps or self.fps,
            codec="libx264",
            quality=7,
            macro_block_size=8,
            pixelformat="yuv420p",
        )

    def export(self, data: Iterable, name: str) -> Path:
        writer = self.open(name)
        try:
            for frame in data:
                writer.append_data(frame)
        finally:
            writer.close()
        return self._path(name)


class ReportExporter(Exporter):
    """Writes run reports / evaluation metrics as JSON, Markdown or CSV (by extension)."""

    def export(self, data: Any, name: str) -> Path:
        path = self._path(name)
        if name.endswith(".json"):
            path.write_text(json.dumps(data, indent=2, default=str))
        elif name.endswith(".csv"):
            rows = list(data)
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        elif name.endswith(".md"):
            path.write_text(data if isinstance(data, str) else self.markdown(data))
        else:
            raise ValueError(f"unsupported report extension: {name}")
        return path

    @staticmethod
    def markdown(report: dict) -> str:
        """Render a single-run report dict as Markdown."""
        lines = ["# Fetch run report", ""]
        lines += [f"| {k} | {v} |" for k, v in [("field", "value"), ("---", "---")]]
        for k, v in report.items():
            if k != "events":
                lines.append(f"| {k} | {v} |")
        if report.get("events"):
            lines += ["", "## Event log", ""]
            lines += [f"- t={t:.1f}s: {msg}" for t, msg in report["events"]]
        return "\n".join(lines) + "\n"


class MapExporter(Exporter):
    """Saves the occupancy grid + semantic overlay (a PIL image or array) as PNG."""

    def export(self, data: Any, name: str) -> Path:
        from PIL import Image

        img = data if isinstance(data, Image.Image) else Image.fromarray(data)
        path = self._path(name)
        img.save(path)
        return path
