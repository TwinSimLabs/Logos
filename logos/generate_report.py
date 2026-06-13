from logos.reporting.report_generator import (
    WorldReportGenerator
)

generator = WorldReportGenerator(
    output_dir="output",
    report_dir="outputs/v1",
    model="qwen3:1.7b",
)

path = generator.run()

print(f"Report written to: {path}")