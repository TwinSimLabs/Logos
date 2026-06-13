# logos/main.py

import json
import os
from datetime import date, datetime

from logos.utils.logging import setup_logging, get_logger
from logos.world.generator import WorldGenerator
from logos.world.stats import (
    calculate_benchmark_statistics,
    save_benchmark_statistics,
)
from logos.world.auditor import (
    analyze_sccs,
    extract_reasoning_chains,
    generate_topology_report,
)
from logos.sig.sources.generator import SourceGenerator
from logos.sig.communities.generator import CommunityGenerator
from logos.sig.observations.generator import ObservationGenerator
from logos.sig.beliefs.generator import BeliefGenerator

logger = get_logger(__name__)


def generate_sig_pipeline(
    entity_count: int = 1000,
    fact_count: int = 5000,
    source_count: int = 100,
    world_config=None,
):
    """Generate the Phase 1 SIG outputs: sources, communities, observations, beliefs."""

    world_generator = WorldGenerator()
    if world_config is not None:
        truth_graph = world_generator.generate(config=world_config)
    else:
        truth_graph = world_generator.generate(
            entity_count=entity_count,
            fact_count=fact_count,
        )

    source_generator = SourceGenerator()
    sources = source_generator.generate(count=source_count)

    community_generator = CommunityGenerator()
    communities = community_generator.process(sources)

    observation_generator = ObservationGenerator()
    observations = observation_generator.process(
        truth_graph,
        sources,
    )

    belief_generator = BeliefGenerator()
    beliefs = belief_generator.process(observations)

    return {
        "truth_graph": truth_graph,
        "sources": sources,
        "communities": communities,
        "observations": observations,
        "beliefs": beliefs,
    }


def _get_field(value, field_name):
    if isinstance(value, dict):
        return value.get(field_name)

    if hasattr(value, field_name):
        return getattr(value, field_name)

    if hasattr(value, "model_dump"):
        return value.model_dump().get(field_name)

    return None


def _json_safe(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value


def build_knowledge_base_dataset(result):
    """Construct a final knowledge-base dataset from the Phase 1 SIG outputs."""

    knowledge_base = []

    for source in result.get("sources", []):
        source_payload = _json_safe(source)
        source_id = _get_field(source, "id")

        observations = [
            item
            for item in result.get("observations", [])
            if _get_field(item, "source_id") == source_id
        ]

        beliefs = [
            item
            for item in result.get("beliefs", [])
            if _get_field(item, "source_id") == source_id
        ]

        community = None

        for item in result.get("communities", []):
            member_ids = (
                _get_field(item, "member_ids")
                or []
            )

            if source_id in member_ids:
                community = _json_safe(item)
                break

        knowledge_base.append(
            {
                "source": source_payload,
                "community": community,
                "observation_count": len(observations),
                "belief_count": len(beliefs),
                "beliefs": [
                    _json_safe(item)
                    for item in beliefs
                ],
                "observations": [
                    _json_safe(item)
                    for item in observations
                ],
            }
        )

    return {
        "summary": {
            "source_count": len(
                result.get("sources", [])
            ),
            "community_count": len(
                result.get("communities", [])
            ),
            "observation_count": len(
                result.get("observations", [])
            ),
            "belief_count": len(
                result.get("beliefs", [])
            ),
        },
        "knowledge_base": knowledge_base,
    }


def save_knowledge_base_dataset(
    result,
    output_dir: str = None,
):
    """Write a final knowledge-base dataset file for the Phase 1 SIG outputs."""

    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(),
            "output",
        )

    os.makedirs(output_dir, exist_ok=True)

    dataset = build_knowledge_base_dataset(result)

    path = os.path.join(
        output_dir,
        "knowledge_base.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(dataset),
            handle,
            indent=2,
        )

    return "knowledge_base.json"


def save_sig_pipeline_outputs(
    result,
    output_dir: str = None,
):
    """Write the Phase 1 SIG outputs to JSON files in the given directory."""

    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(),
            "output",
        )

    os.makedirs(output_dir, exist_ok=True)

    file_map = {
        "sources.json": result.get("sources", []),
        "communities.json": result.get(
            "communities",
            [],
        ),
        "observations.json": result.get(
            "observations",
            [],
        ),
        "beliefs.json": result.get("beliefs", []),
    }

    written_files = []

    for file_name, payload in file_map.items():
        path = os.path.join(
            output_dir,
            file_name,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                [
                    _json_safe(item)
                    for item in payload
                ],
                handle,
                indent=2,
            )

        written_files.append(file_name)

    return written_files


def main():
    setup_logging()

    logger.info(
        "Starting Logos Synthetic Information Ecosystem Pipeline"
    )

    result = generate_sig_pipeline()

    # Generate benchmark statistics
    stats = calculate_benchmark_statistics(
        result["truth_graph"]
    )

    # Preserve the richer topology-health assessment produced by stats.py.
    # Older SCC-only labels are kept as compatibility fallbacks.
    if stats.get("TopologyHealthReasons"):
        health_status = "UNHEALTHY"
    elif stats["Entities"] == 0:
        health_status = "TOPOLOGY_FAILURE"
    elif (
        stats["NumberOfSCCs"]
        / stats["Entities"]
        > 0.5
    ):
        health_status = "TOO_DISCONNECTED"
    elif (
        stats["LargestSCCSize"]
        / stats["Entities"]
        > 0.8
    ):
        health_status = "TOO_CYCLIC"
    else:
        health_status = "HEALTHY"

    stats["BenchmarkHealth"] = health_status

    save_benchmark_statistics(
        stats,
        output_dir="output",
    )

    # Generate graph audit reports
    output_dir = os.path.join(
        os.getcwd(),
        "output",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    analyze_sccs(
        result["truth_graph"],
        output_dir=output_dir,
    )

    extract_reasoning_chains(
        result["truth_graph"],
        output_dir=output_dir,
    )

    generate_topology_report(
        result["truth_graph"],
        stats,
        output_dir=output_dir,
    )

    save_sig_pipeline_outputs(
        result,
        output_dir="output",
    )

    save_knowledge_base_dataset(
        result,
        output_dir="output",
    )
    from logos.reporting.report_generator import (
    WorldReportGenerator
)
    generator = WorldReportGenerator(
    output_dir="output",
    report_dir="outputs/v1",
    model="qwen3:14b"
)

    generator.run()

    logger.info(
        (
            "Pipeline completed with %d sources, "
            "%d observations, %d beliefs. "
            "Benchmark Health: %s"
        ),
        len(result["sources"]),
        len(result["observations"]),
        len(result["beliefs"]),
        stats["BenchmarkHealth"],
    )


if __name__ == "__main__":
    main()
