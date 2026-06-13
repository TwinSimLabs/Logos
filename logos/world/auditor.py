# logos/world/auditor.py

from typing import Dict, Any
import json
import networkx as nx
from logos.world.graph import TruthGraph
from collections import Counter
import logging
import random

logger = logging.getLogger(__name__)


def analyze_sccs(
    truth_graph: TruthGraph,
    output_dir: str = "output",
):
    G = nx.DiGraph()

    for fact in truth_graph.get_all_facts():
        G.add_edge(fact.subject_id, fact.object_id)

    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = sorted(
        [len(scc) for scc in sccs],
        reverse=True,
    )

    output_content = ""
    output_content += "## SCC Analysis Report\n\n"
    output_content += f"Number of SCCs: {len(sccs)}\n\n"
    output_content += (
        f"Largest SCC Size: {scc_sizes[0] if scc_sizes else 0}\n\n"
    )

    output_content += "### SCC Size Distribution (Top 20)\n"

    for i, size in enumerate(scc_sizes[:20]):
        output_content += f"{i + 1}. {size}\n"

    output_content += "\n"

    if (
        sccs
        and len(G.nodes()) > 0
        and scc_sizes[0] / len(G.nodes()) > 0.5
    ):
        output_content += "### Observation\n"
        output_content += (
            "The largest SCC contains more than 50% of all entities, "
            "indicating a highly interconnected core in the graph.\n\n"
        )

    with open(f"{output_dir}/graph_audit.md", "w") as f:
        f.write(output_content)


def extract_reasoning_chains(
    truth_graph: TruthGraph,
    num_chains: int = 50,
    output_dir: str = "output",
):
    G = nx.DiGraph()

    for fact in truth_graph.get_all_facts():
        G.add_edge(fact.subject_id, fact.object_id)

    graph_nodes = list(G.nodes())

    sampled_nodes = random.sample(
        graph_nodes,
        min(len(graph_nodes), 100),
    )

    all_simple_paths = []

    for source_node in sampled_nodes:
        for target_node in sampled_nodes:
            if source_node == target_node:
                continue

            try:
                for path in nx.all_simple_paths(
                    G,
                    source_node,
                    target_node,
                    cutoff=7,
                ):
                    if len(path) > 2:
                        all_simple_paths.append(
                            {
                                "length": len(path) - 1,
                                "chain": path,
                            }
                        )

            except nx.NetworkXNoPath:
                pass

    all_simple_paths_sorted = sorted(
        all_simple_paths,
        key=lambda x: x["length"],
        reverse=True,
    )

    longest_chains = all_simple_paths_sorted[:num_chains]

    with open(
        f"{output_dir}/reasoning_examples.json",
        "w",
    ) as f:
        json.dump(longest_chains, f, indent=2)


def generate_topology_report(
    truth_graph: TruthGraph,
    stats: Dict[str, Any],
    output_dir: str = "output",
):
    G = nx.DiGraph()

    for fact in truth_graph.get_all_facts():
        G.add_edge(fact.subject_id, fact.object_id)

    output_content = ""
    output_content += "## Graph Topology Report\n\n"

    output_content += "### Overall Graph Statistics\n"
    for key, value in stats.items():
        output_content += f"- {key}: {value}\n"

    output_content += "\n"

    # Weakly Connected Components
    wccs = list(nx.weakly_connected_components(G))
    num_wccs = len(wccs)

    largest_wcc_size = (
        max(len(wcc) for wcc in wccs)
        if wccs
        else 0
    )

    coverage_pct = (
        round(
            largest_wcc_size / len(G.nodes()) * 100,
            2,
        )
        if len(G.nodes()) > 0
        else 0
    )

    output_content += (
        "### Connected Components Analysis "
        "(Weakly Connected Components)\n"
    )

    output_content += (
        f"Number of Weakly Connected Components (WCCs): "
        f"{num_wccs}\n"
    )

    output_content += (
        f"Largest WCC Size: {largest_wcc_size} "
        f"(covering {coverage_pct}% of nodes)\n\n"
    )

    # Degree distributions
    in_degrees = [
        G.in_degree(node)
        for node in G.nodes()
    ]

    out_degrees = [
        G.out_degree(node)
        for node in G.nodes()
    ]

    in_degree_counts = Counter(in_degrees)
    out_degree_counts = Counter(out_degrees)

    output_content += "### Degree Distributions\n"

    output_content += "**In-Degree Distribution:**\n"
    for degree, count in sorted(
        in_degree_counts.items()
    ):
        output_content += (
            f"- Degree {degree}: {count} nodes\n"
        )

    output_content += "\n"

    output_content += "**Out-Degree Distribution:**\n"
    for degree, count in sorted(
        out_degree_counts.items()
    ):
        output_content += (
            f"- Degree {degree}: {count} nodes\n"
        )

    output_content += "\n"

    output_content += (
        "### Longest Chains and Reasoning Depth "
        "(from Benchmark Statistics)\n"
    )

    output_content += (
        f"- Longest Acyclic Chain (across SCCs): "
        f"{stats.get('LongestAcyclicChain', 0)}\n"
    )

    output_content += (
        f"- Estimated Reasoning Depth "
        f"(across & within SCCs): "
        f"{stats.get('EstimatedReasoningDepth', 0)}\n\n"
    )

    output_content += "### Interpretation\n"

    if (
        len(G.nodes()) > 0
        and num_wccs > 1
        and largest_wcc_size / len(G.nodes()) < 0.8
    ):
        output_content += (
            "1. **Is the graph mostly disconnected?** "
            "Yes, the graph appears to have several "
            "disconnected components or many smaller "
            "subgraphs.\n"
        )
    else:
        output_content += (
            "1. **Is the graph mostly disconnected?** "
            "No, the graph appears largely connected "
            "through a dominant weakly connected "
            "component.\n"
        )

    output_content += "\n"

    if (
        len(G.nodes()) > 0
        and stats.get("LargestSCCSize", 0)
        / len(G.nodes())
        > 0.8
    ):
        output_content += (
            "2. **Is the graph mostly one giant SCC?** "
            "Yes, a large fraction of the graph belongs "
            "to one SCC, indicating extensive cycles.\n"
        )
    else:
        output_content += (
            "2. **Is the graph mostly one giant SCC?** "
            "No, the graph is not dominated by a single "
            "strongly connected component.\n"
        )

    output_content += "\n"

    if stats.get("EstimatedReasoningDepth", 0) > 2:
        output_content += (
            "3. **Are meaningful multi-hop chains "
            "present?** Yes. The estimated reasoning "
            f"depth of {stats.get('EstimatedReasoningDepth')} "
            "suggests non-trivial inference chains.\n"
        )
    else:
        output_content += (
            "3. **Are meaningful multi-hop chains "
            "present?** The estimated reasoning depth "
            f"of {stats.get('EstimatedReasoningDepth')} "
            "suggests limited multi-hop structure.\n"
        )

    output_content += "\n"

    output_content += (
        "4. **Are reasoning chains realistic for retrieval "
        "evaluation?** A larger reasoning depth and richer "
        "SCC structure generally create a more challenging "
        "retrieval environment, requiring systems to traverse "
        "long inference chains and cyclic dependencies.\n"
    )

    with open(
        f"{output_dir}/topology_report.md",
        "w",
    ) as f:
        f.write(output_content)