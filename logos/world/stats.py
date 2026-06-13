import json
import logging
import math
from collections import Counter
from itertools import combinations
from typing import Iterable, List, Sequence

import networkx as nx

from logos.world.graph import TruthGraph

logger = logging.getLogger(__name__)


def calculate_benchmark_statistics(truth_graph: TruthGraph):
    facts = truth_graph.get_all_facts()

    G = nx.DiGraph()
    for entity_id in truth_graph.entities:
        G.add_node(entity_id)

    triples = []
    predicates = []
    edge_roles = []
    for fact in facts:
        G.add_edge(fact.subject_id, fact.object_id)
        triples.append((fact.subject_id, fact.predicate, fact.object_id))
        predicates.append(fact.predicate)
        role = getattr(fact, "metadata", {}).get("topology_role")
        if role:
            edge_roles.append(role)

    num_entities_in_graph = len(G.nodes())
    num_relationships_in_graph = len(G.edges())
    fact_count = len(facts)
    unique_triples = len(set(triples))
    duplicate_edge_ratio = _safe_ratio(
        fact_count - num_relationships_in_graph,
        fact_count,
    )
    duplicate_triple_ratio = _safe_ratio(
        fact_count - unique_triples,
        fact_count,
    )

    if num_entities_in_graph == 0:
        return {
            "Entities": 0,
            "Facts": fact_count,
            "Relationships": 0,
            "UniqueSubjectObjectEdges": 0,
            "UniqueTriples": 0,
            "DuplicateEdgeRatio": 0,
            "DuplicateTripleRatio": 0,
            "NumberOfSCCs": 0,
            "LargestSCCSize": 0,
            "LargestSCCFraction": 0,
            "AverageInDegree": 0,
            "AverageOutDegree": 0,
            "AverageShortestPathLength": 0,
            "GraphDiameter": 0,
            "LongestAcyclicChain": 0,
            "EstimatedReasoningDepth": 0,
            "EffectiveBranchingFactor": 0,
            "PathDiversityP50": 0,
            "PathDiversityP90": 0,
            "BridgeDensity": 0,
            "TopHubDegreeShare": 0,
            "InDegreeEntropy": 0,
            "OutDegreeEntropy": 0,
            "PredicateEntropy": 0,
            "BenchmarkHealth": "TOPOLOGY_FAILURE",
            "TopologyHealthReasons": ["empty_graph"],
        }

    sccs = list(nx.strongly_connected_components(G))
    num_sccs = len(sccs)
    largest_scc_size = max(len(scc) for scc in sccs) if sccs else 0
    largest_scc_fraction = _safe_ratio(largest_scc_size, num_entities_in_graph)

    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    total_degrees = [
        G.in_degree(node) + G.out_degree(node)
        for node in G.nodes()
    ]

    avg_in_degree = _safe_ratio(sum(in_degrees), num_entities_in_graph)
    avg_out_degree = _safe_ratio(sum(out_degrees), num_entities_in_graph)

    avg_shortest_path_length, graph_diameter = _path_metrics(G)
    longest_acyclic_chain, estimated_reasoning_depth = _reasoning_depth(G, sccs)

    effective_branching_factor = _effective_branching_factor(G)
    path_diversity_p50, path_diversity_p90 = _path_diversity(G)
    bridge_density = _bridge_density(facts)
    top_hub_degree_share = _top_degree_share(total_degrees, 0.01)
    in_degree_entropy = _entropy_from_values(in_degrees)
    out_degree_entropy = _entropy_from_values(out_degrees)
    predicate_entropy = _entropy_from_values(predicates)

    health_reasons = _topology_health_reasons(
        G=G,
        fact_count=fact_count,
        duplicate_edge_ratio=duplicate_edge_ratio,
        largest_scc_fraction=largest_scc_fraction,
        effective_branching_factor=effective_branching_factor,
        out_degree_entropy=out_degree_entropy,
        predicate_entropy=predicate_entropy,
        estimated_reasoning_depth=estimated_reasoning_depth,
    )

    stats = {
        "Entities": num_entities_in_graph,
        "Facts": fact_count,
        "Relationships": num_relationships_in_graph,
        "UniqueSubjectObjectEdges": num_relationships_in_graph,
        "UniqueTriples": unique_triples,
        "DuplicateEdgeRatio": round(duplicate_edge_ratio, 3),
        "DuplicateTripleRatio": round(duplicate_triple_ratio, 3),
        "NumberOfSCCs": num_sccs,
        "LargestSCCSize": largest_scc_size,
        "LargestSCCFraction": round(largest_scc_fraction, 3),
        "AverageInDegree": round(avg_in_degree, 3),
        "AverageOutDegree": round(avg_out_degree, 3),
        "AverageShortestPathLength": round(avg_shortest_path_length, 3),
        "GraphDiameter": graph_diameter,
        "LongestAcyclicChain": longest_acyclic_chain,
        "EstimatedReasoningDepth": estimated_reasoning_depth,
        "EffectiveBranchingFactor": round(effective_branching_factor, 3),
        "PathDiversityP50": round(path_diversity_p50, 3),
        "PathDiversityP90": round(path_diversity_p90, 3),
        "BridgeDensity": round(bridge_density, 3),
        "TopHubDegreeShare": round(top_hub_degree_share, 3),
        "InDegreeEntropy": round(in_degree_entropy, 3),
        "OutDegreeEntropy": round(out_degree_entropy, 3),
        "PredicateEntropy": round(predicate_entropy, 3),
        "TopologyRoleDistribution": dict(Counter(edge_roles)),
        "TopologyHealthReasons": health_reasons,
    }

    stats["BenchmarkHealth"] = "HEALTHY" if not health_reasons else "UNHEALTHY"
    return stats


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0


def _path_metrics(G: nx.DiGraph) -> tuple[float, int]:
    undirected_graph = G.to_undirected()
    if not undirected_graph.nodes():
        return 0, 0

    largest_cc_nodes = max(
        nx.connected_components(undirected_graph),
        key=len,
    )
    largest_cc = undirected_graph.subgraph(largest_cc_nodes)
    if largest_cc.number_of_nodes() <= 1:
        return 0, 0

    try:
        if largest_cc.number_of_nodes() <= 2000:
            return (
                nx.average_shortest_path_length(largest_cc),
                nx.diameter(largest_cc),
            )

        sample_nodes = list(largest_cc.nodes())[:200]
        lengths = []
        eccentricities = []
        for source in sample_nodes:
            source_lengths = nx.single_source_shortest_path_length(
                largest_cc,
                source,
                cutoff=20,
            )
            lengths.extend(length for target, length in source_lengths.items() if target != source)
            if source_lengths:
                eccentricities.append(max(source_lengths.values()))
        return (
            sum(lengths) / len(lengths) if lengths else 0,
            max(eccentricities) if eccentricities else 0,
        )
    except nx.NetworkXError as error:
        logger.warning("Error calculating path metrics: %s", error)
        return 0, 0


def _reasoning_depth(
    G: nx.DiGraph,
    sccs: Sequence[set[str]],
) -> tuple[int, int]:
    if nx.is_directed_acyclic_graph(G):
        longest = nx.dag_longest_path_length(G) if G.number_of_edges() > 0 else 0
        return longest, longest

    condensation = nx.condensation(G)
    longest_acyclic_chain = (
        nx.dag_longest_path_length(condensation)
        if condensation.number_of_edges() > 0
        else 0
    )

    max_local_depth = 0
    for scc_nodes in sorted(sccs, key=len, reverse=True)[:25]:
        if len(scc_nodes) <= 1:
            continue
        scc_subgraph = G.subgraph(scc_nodes)
        for source_node in list(scc_subgraph.nodes())[:25]:
            lengths = nx.single_source_shortest_path_length(
                scc_subgraph,
                source_node,
                cutoff=8,
            )
            if lengths:
                max_local_depth = max(max_local_depth, max(lengths.values()))

    return longest_acyclic_chain, longest_acyclic_chain + max_local_depth


def _effective_branching_factor(G: nx.DiGraph, depth: int = 2) -> float:
    values = []
    for node in list(G.nodes())[:500]:
        previous = {node}
        reached = {node}
        ratios = []
        frontier = {node}
        for _ in range(depth):
            next_frontier = {
                neighbor
                for item in frontier
                for neighbor in G.successors(item)
                if neighbor not in reached
            }
            if previous:
                ratios.append(len(next_frontier) / len(previous))
            reached.update(next_frontier)
            previous = next_frontier
            frontier = next_frontier
        if ratios:
            values.append(sum(ratios) / len(ratios))
    return sum(values) / len(values) if values else 0


def _path_diversity(G: nx.DiGraph) -> tuple[float, float]:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0, 0

    pairs = list(combinations(nodes[:80], 2))[:250]
    counts: List[int] = []
    for source, target in pairs:
        try:
            count = sum(
                1
                for _ in nx.all_simple_paths(
                    G,
                    source,
                    target,
                    cutoff=5,
                )
            )
            if count:
                counts.append(min(count, 25))
        except nx.NetworkXNoPath:
            continue

    if not counts:
        return 0, 0
    counts.sort()
    return _percentile(counts, 50), _percentile(counts, 90)


def _bridge_density(facts: Iterable) -> float:
    total = 0
    bridge_count = 0
    for fact in facts:
        total += 1
        if getattr(fact, "metadata", {}).get("topology_role") == "bridge":
            bridge_count += 1
    return _safe_ratio(bridge_count, total)


def _top_degree_share(values: Sequence[int], fraction: float) -> float:
    if not values or sum(values) == 0:
        return 0
    count = max(1, int(len(values) * fraction))
    top_values = sorted(values, reverse=True)[:count]
    return sum(top_values) / sum(values)


def _entropy_from_values(values: Sequence) -> float:
    if not values:
        return 0
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log(count / total, 2)
        for count in counts.values()
    )


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0
    index = int(round((percentile / 100) * (len(values) - 1)))
    return values[index]


def _topology_health_reasons(
    G: nx.DiGraph,
    fact_count: int,
    duplicate_edge_ratio: float,
    largest_scc_fraction: float,
    effective_branching_factor: float,
    out_degree_entropy: float,
    predicate_entropy: float,
    estimated_reasoning_depth: int,
) -> List[str]:
    reasons = []
    entity_count = G.number_of_nodes()

    if fact_count == 0:
        reasons.append("no_facts")

    if largest_scc_fraction > 0.35 and entity_count > 20:
        reasons.append("giant_scc")

    degree_counts = Counter(
        (G.in_degree(node), G.out_degree(node))
        for node in G.nodes()
    )
    ring_like_fraction = _safe_ratio(degree_counts.get((1, 1), 0), entity_count)
    if ring_like_fraction > 0.80 and entity_count > 20:
        reasons.append("ring_graph_collapse")

    if duplicate_edge_ratio > 0.25:
        reasons.append("duplicate_edge_collapse")

    if effective_branching_factor < 1.2 and entity_count > 20:
        reasons.append("low_branching_factor")

    if out_degree_entropy < 1.0 and entity_count > 20:
        reasons.append("low_degree_diversity")

    if predicate_entropy < 2.0 and fact_count > 20:
        reasons.append("low_predicate_diversity")

    if estimated_reasoning_depth < 4 and entity_count > 20:
        reasons.append("insufficient_reasoning_depth")

    undirected_graph = G.to_undirected()
    if undirected_graph.nodes():
        largest_wcc_size = max(
            len(component)
            for component in nx.connected_components(undirected_graph)
        )
        if _safe_ratio(largest_wcc_size, entity_count) < 0.50 and entity_count > 20:
            reasons.append("disconnected_graph_collapse")

    return reasons


def save_benchmark_statistics(
    stats,
    output_dir: str = "output",
):
    path = f"{output_dir}/benchmark_statistics.json"

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
