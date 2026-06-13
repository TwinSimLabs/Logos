import json
from pathlib import Path


class WorldDataLoader:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)

    def load(self):
        return {
            "beliefs": self._read("beliefs.json"),
            "observations": self._read("observations.json"),
            "sources": self._read("sources.json"),
            "communities": self._read("communities.json"),
            "knowledge_base": self._read("knowledge_base.json"),
            "benchmark_statistics": self._read(
                "benchmark_statistics.json"
            ),
            "reasoning_examples": self._read(
                "reasoning_examples.json"
            ),
        }

    def _read(self, filename):
        path = self.output_dir / filename

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)