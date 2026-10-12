"""Dependency-free static HTML for independently ranked scalar metrics."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import isfinite
from pathlib import PurePosixPath
from typing import Sequence


@dataclass(frozen=True, slots=True)
class MetricReportRow:
    rank: int
    candidate: str
    checkpoint_update: int
    value: float

    def __post_init__(self) -> None:
        if self.rank <= 0 or not self.candidate or self.checkpoint_update < 0:
            raise ValueError("metric report row identity is invalid")
        if not isfinite(self.value) or self.value < 0.0:
            raise ValueError("metric report row value must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MetricReportTable:
    key: str
    title: str
    unit: str
    decimal_places: int
    rows: tuple[MetricReportRow, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.title or not self.unit:
            raise ValueError("metric report table text fields must be non-empty")
        if self.decimal_places < 0:
            raise ValueError("metric report decimal places must be non-negative")
        if len(self.rows) < 2:
            raise ValueError("metric report table requires at least two candidates")
        expected_ranks = tuple(range(1, len(self.rows) + 1))
        if tuple(row.rank for row in self.rows) != expected_ranks:
            raise ValueError("metric report rows must use consecutive ranks")
        if any(
            left.value > right.value
            for left, right in zip(self.rows, self.rows[1:])
        ):
            raise ValueError("metric report rows must be sorted with lower values first")


@dataclass(frozen=True, slots=True)
class MetricReportVideo:
    candidate: str
    checkpoint_update: int
    source: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.source)
        if (
            not self.candidate
            or self.checkpoint_update < 0
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix.lower() != ".mp4"
        ):
            raise ValueError("metric report video must use a safe relative MP4 path")


def _metric_table_html(table: MetricReportTable) -> str:
    maximum = max(row.value for row in table.rows)
    body: list[str] = []
    for row in table.rows:
        width = 0.0 if maximum <= 0.0 else 100.0 * row.value / maximum
        winner_class = ' class="best"' if row.rank == 1 else ""
        body.append(
            f"<tr{winner_class}>"
            f'<td class="rank">#{row.rank}</td>'
            f"<td>{escape(row.candidate)}</td>"
            f'<td class="number">{row.checkpoint_update}</td>'
            '<td class="bar-cell">'
            f'<span class="bar" style="width:{width:.3f}%"></span>'
            f'<span class="value">{row.value:.{table.decimal_places}f} '
            f"{escape(table.unit)}</span>"
            "</td>"
            "</tr>"
        )
    return (
        '<section class="metric-card">'
        f"<h2>{escape(table.title)}</h2>"
        '<p class="direction">独立排序 · 数值越小越好</p>'
        "<table>"
        "<thead><tr><th>排名</th><th>策略</th><th>Update</th><th>评估值</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</section>"
    )


def _video_html(video: MetricReportVideo) -> str:
    return (
        '<article class="video-card">'
        f"<h3>{escape(video.candidate)}</h3>"
        f"<p>checkpoint update {video.checkpoint_update}</p>"
        '<video controls preload="metadata" playsinline>'
        f'<source src="{escape(video.source, quote=True)}" type="video/mp4">'
        "当前浏览器不支持 MP4 播放。"
        "</video>"
        "</article>"
    )


def build_independent_metric_report(
    *,
    title: str,
    subtitle: str,
    tables: Sequence[MetricReportTable],
    videos: Sequence[MetricReportVideo],
) -> str:
    """Build a self-contained dashboard with no aggregate score or winner."""

    frozen_tables = tuple(tables)
    frozen_videos = tuple(videos)
    if not title or not subtitle:
        raise ValueError("metric report title and subtitle must be non-empty")
    if not 3 <= len(frozen_tables) <= 4:
        raise ValueError("metric report must show three or four independent metrics")
    keys = [table.key for table in frozen_tables]
    if len(keys) != len(set(keys)):
        raise ValueError("metric report table keys must be unique")
    if frozen_videos:
        video_labels = [video.candidate for video in frozen_videos]
        if len(video_labels) != len(set(video_labels)):
            raise ValueError("metric report video candidates must be unique")

    metric_cards = "".join(_metric_table_html(table) for table in frozen_tables)
    metric_count_zh = {3: "三", 4: "四"}[len(frozen_tables)]
    video_section = ""
    if frozen_videos:
        video_section = (
            '<section class="videos">'
            "<div><h2>策略视频</h2>"
            "<p>每个策略固定录制一个 rollout；其余 rollout 仍参与表格统计。</p></div>"
            f'<div class="video-grid">{"".join(_video_html(video) for video in frozen_videos)}</div>'
            "</section>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0b1020;
  --panel: #141b2d;
  --panel-2: #19233a;
  --line: #2a3856;
  --text: #ecf3ff;
  --muted: #9fb0cb;
  --accent: #5eead4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: radial-gradient(circle at top left, #172449 0, var(--bg) 42%);
  color: var(--text);
  font: 15px/1.55 Inter, ui-sans-serif, system-ui, sans-serif;
}}
main {{ width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 36px 0 60px; }}
header {{ margin-bottom: 26px; }}
h1 {{ margin: 0 0 8px; font-size: clamp(26px, 4vw, 42px); letter-spacing: -0.03em; }}
h2, h3 {{ margin: 0; }}
header p, .videos > div > p {{ color: var(--muted); margin: 8px 0 0; }}
.notice {{
  display: inline-block;
  margin-top: 16px;
  padding: 8px 12px;
  border: 1px solid #2d7e75;
  border-radius: 999px;
  color: var(--accent);
  background: #12302f;
}}
.metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.metric-card, .videos {{
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--panel);
  box-shadow: 0 18px 55px rgba(0,0,0,.22);
}}
.metric-card {{ padding: 20px; overflow: hidden; }}
.direction {{ margin: 4px 0 16px; color: var(--muted); font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
tbody tr:last-child td {{ border-bottom: 0; }}
tr.best {{ background: rgba(94,234,212,.07); }}
tr.best .rank {{ color: var(--accent); font-weight: 800; }}
.number {{ font-variant-numeric: tabular-nums; }}
.bar-cell {{ position: relative; min-width: 170px; overflow: hidden; }}
.bar {{
  position: absolute;
  inset: 7px auto 7px 0;
  border-radius: 6px;
  background: linear-gradient(90deg, rgba(89,119,200,.18), rgba(89,119,200,.52));
}}
.best .bar {{ background: linear-gradient(90deg, rgba(94,234,212,.18), rgba(94,234,212,.44)); }}
.value {{ position: relative; z-index: 1; font-variant-numeric: tabular-nums; font-weight: 650; }}
.videos {{ margin-top: 22px; padding: 22px; }}
.video-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }}
.video-card {{ padding: 14px; border: 1px solid var(--line); border-radius: 13px; background: var(--panel-2); }}
.video-card p {{ color: var(--muted); margin: 2px 0 10px; font-size: 13px; }}
video {{ display: block; width: 100%; border-radius: 9px; background: #05070c; aspect-ratio: 16 / 9; }}
@media (max-width: 900px) {{
  .metric-grid, .video-grid {{ grid-template-columns: 1fr; }}
  main {{ width: min(100% - 20px, 760px); padding-top: 22px; }}
  .metric-card {{ padding: 14px; }}
  th, td {{ padding: 9px 7px; }}
  .bar-cell {{ min-width: 120px; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>{escape(title)}</h1>
  <p>{escape(subtitle)}</p>
  <span class="notice">无综合分 · 无总排名 · {metric_count_zh}项指标分别比较</span>
</header>
<div class="metric-grid">{metric_cards}</div>
{video_section}
</main>
</body>
</html>
"""


__all__ = [
    "MetricReportRow",
    "MetricReportTable",
    "MetricReportVideo",
    "build_independent_metric_report",
]
