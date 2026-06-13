import json
from pathlib import Path

from .loader import WorldDataLoader
from .context_builder import build_world_summary
from .prompts import REPORT_PROMPT
from .ollama_client import OllamaClient


class WorldReportGenerator:

    def __init__(
        self,
        output_dir="output",
        report_dir="outputs/v1",
        model="qwen3:1.7b"
    ):
        self.output_dir = output_dir
        self.report_dir = Path(report_dir)

        self.client = OllamaClient(
            model=model
        )

    def run(self):

        loader = WorldDataLoader(
            self.output_dir
        )

        data = loader.load()

        summary = build_world_summary(
            data
        )

        prompt = REPORT_PROMPT.format(
            world_summary=json.dumps(
                summary,
                indent=2,
            ),
            benchmark_stats=json.dumps(
                data["benchmark_statistics"],
                indent=2,
            ),
            reasoning_examples=json.dumps(
                data["reasoning_examples"][:20],
                indent=2,
            ),
        )

        report = self.client.generate(
            prompt
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = (
            self.report_dir
            / "final_world_report.md"
        )

        report_path.write_text(
            report,
            encoding="utf-8"
        )

        return report_path