"""Contracts for the static independent-metric comparison dashboard."""

from __future__ import annotations

import unittest

from trajectory_tracking.visualization.metric_report import (
    MetricReportRow,
    MetricReportTable,
    MetricReportVideo,
    build_independent_metric_report,
)


def _table(key: str, title: str) -> MetricReportTable:
    return MetricReportTable(
        key=key,
        title=title,
        unit="mm",
        decimal_places=1,
        rows=(
            MetricReportRow(
                rank=1,
                candidate="policy_a",
                checkpoint_update=100,
                value=1.0,
            ),
            MetricReportRow(
                rank=2,
                candidate="policy_b",
                checkpoint_update=200,
                value=2.0,
            ),
        ),
    )


class IndependentMetricReportTest(unittest.TestCase):
    def test_page_has_three_tables_and_one_video_per_candidate(self) -> None:
        page = build_independent_metric_report(
            title="策略对比",
            subtitle="相同 reference suffix",
            tables=(
                _table("link7_mean", "机械臂末端平均误差"),
                _table("link7_max", "机械臂末端最大误差"),
                _table("hand", "手指"),
            ),
            videos=(
                MetricReportVideo(
                    candidate="policy_a",
                    checkpoint_update=100,
                    source="videos/policy_a.mp4",
                ),
                MetricReportVideo(
                    candidate="policy_b",
                    checkpoint_update=200,
                    source="videos/policy_b.mp4",
                ),
            ),
        )

        self.assertEqual(page.count('<section class="metric-card">'), 3)
        self.assertEqual(page.count("<video "), 2)
        self.assertIn("三项指标分别比较", page)
        self.assertIn("数值越小越好", page)
        self.assertNotIn("Task success", page)
        self.assertNotIn("Object flow", page)
        self.assertNotIn("balanced", page.lower())
        self.assertNotIn("总冠军", page)

    def test_video_source_must_be_safe_relative_mp4(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative MP4"):
            MetricReportVideo(
                candidate="policy",
                checkpoint_update=1,
                source="../outside.mp4",
            )

    def test_table_rejects_unsorted_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            MetricReportTable(
                key="arm",
                title="Arm",
                unit="rad",
                decimal_places=4,
                rows=(
                    MetricReportRow(1, "slow", 1, 2.0),
                    MetricReportRow(2, "fast", 1, 1.0),
                ),
            )


if __name__ == "__main__":
    unittest.main()
