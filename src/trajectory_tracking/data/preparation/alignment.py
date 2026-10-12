"""Verify genuine, signed source-state restoration evidence."""
import ast
from hashlib import sha256
import json
from pathlib import Path

from ...workspace import alignment_source_path

def verify_alignment(root: Path, prepared: Path, report_path: Path, contract: dict) -> dict:
    from trajectory_tracking.data.tro_object_replay import digest
    if digest(report_path) != contract["alignment_report_sha256"]:
        raise ValueError("one-time scene alignment report signature changed")
    report = json.loads(report_path.read_text())
    if (report.get("passed") is not True or report.get("producer_success_required") is not contract.get("producer_success_required", True)
            or report.get("action_replay_performed") is not False
            or report.get("source_actions_executed") != 0 or not report.get("restore_checks")):
        raise ValueError("missing genuine scene restoration evidence; replay was not certified")
    expected = contract["static_alignment_sha256"]
    if report["static_alignment_sha256"] != expected or digest(prepared / "static_alignment.json") != expected:
        raise ValueError("static physics/asset alignment changed")
    for relative, expected in contract["physics_file_sha256"].items():
        if digest(alignment_source_path(root, relative)) != expected:
            raise ValueError(f"physics bytes changed; recertify alignment: {relative}")
    for item in contract["physics_symbols"]:
        text = alignment_source_path(root, item["path"]).read_text()
        node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == item["symbol"])
        if sha256(ast.get_source_segment(text, node).encode()).hexdigest() != item["sha256"]:
            raise ValueError("physics backend changed; recertify alignment")
    return dict(report_sha256=contract["alignment_report_sha256"],
                static_sha256=contract["static_alignment_sha256"],
                physics_bytes_identical=True, action_replay_performed=False,
                restore_checks=report["restore_checks"])
