"""Check native physics logs without entering Python from simulator threads."""
import json
import re
import time
from pathlib import Path


class PhysicsHealthGuard:
    def __init__(self, path):
        self.path = Path(path)
        self.failure = None
        self.stream = None
        self.pending = ""

    def on_log(self, source, level, filename, line, message):
        if self.failure is not None or "physx" not in source.lower():
            return
        text = message.lower()
        if any(term in text for term in (
            "physx error", "physx abort", "buffer overflow", "contacts have been dropped",
            "failed to allocate", "simulation cannot continue", "cuda error",
        )):
            self.failure = dict(unix_seconds=time.time(), source=source,
                                level=level, file=filename, line=line, message=message)

    def watch(self, native_log):
        # A Python Carb callback can deadlock native cooking/import worker threads
        # against the GIL. Read Kit's explicit per-process log at step boundaries.
        self.stream = Path(native_log).open(encoding="utf-8", errors="replace")

    def check(self):
        if self.stream is not None and self.failure is None:
            while chunk := self.stream.read(65536):
                lines = (self.pending + chunk).split("\n")
                self.pending = lines.pop()
                for line in lines:
                    source = re.search(r"\[([^]\n]*physx[^]\n]*)\]", line, re.IGNORECASE)
                    if source is not None:
                        self.on_log(source[1], None, self.stream.name, None, line)
                if self.failure is not None:
                    break
        if self.failure is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.failure, indent=2) + "\n")
            raise RuntimeError("PhysX simulation invalid; training aborted: " + self.failure["message"])

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
