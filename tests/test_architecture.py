"""Application and viewer dependencies must remain outside core modules."""
import ast
from pathlib import Path

def test_core_does_not_import_cli_or_visualization():
    package=Path(__file__).resolve().parents[1] / "src/trajectory_tracking"
    failures=[]
    for directory in ("data", "environment", "training"):
        for path in (package/directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names=([a.name for a in node.names] if isinstance(node,ast.Import)
                    else [node.module or ""] if isinstance(node,ast.ImportFrom) else [])
                if any(name.startswith(("trajectory_tracking.cli","trajectory_tracking.visualization"))
                       or name.startswith(("cli.","visualization.")) for name in names):
                    failures.append(str(path.relative_to(package)))
    assert not failures, failures
