# Logos Synthetic Information Ecosystem

## 1. Executive Summary

Project Logos is a synthetic information ecosystem for evaluating retrieval, graph traversal, source attribution, trust ranking, and multi-hop reasoning systems. The system separates objective world truth from subjective information flow:

1. `WorldGenerator` creates a ground-truth `TruthGraph`.
2. `SourceGenerator` creates information sources with reliability and bias.
3. `CommunityGenerator` groups sources into social communities.
4. `ObservationGenerator` turns ground-truth facts into noisy source-specific observations.
5. `BeliefGenerator` aggregates observations into source-level beliefs.
6. `main.py` exports benchmark statistics, topology reports, SIG artifacts, and a consolidated knowledge-base dataset.

The current implementation is topology-driven. It no longer treats `fact_count` as a simple loop over adjacent entities. Instead, it generates communities, directed acyclic regions, local strongly connected components, hubs, bridges, hierarchy edges, conflict edges, and controlled random edges. It then materializes those topology edges into `Fact` objects.

The key design correction is that fact volume and topology volume are measured separately:

- `Facts`: number of stored `Fact` objects in `TruthGraph.facts`.
- `Relationships` / `UniqueSubjectObjectEdges`: number of unique directed `(subject_id, object_id)` edges after projection into `networkx.DiGraph`.
- `UniqueTriples`: number of unique `(subject_id, predicate, object_id)` triples.
- `DuplicateEdgeRatio`: how much fact volume collapses onto already existing subject/object edges.

This distinction is essential for benchmark validity. A large number of facts does not imply a structurally rich graph.

---

## 2. Current Pipeline Architecture

### 2.1 Execution Chain

`logos/main.py` orchestrates the pipeline:

```text
generate_sig_pipeline()
  -> WorldGenerator.generate()
  -> SourceGenerator.generate()
  -> CommunityGenerator.process()
  -> ObservationGenerator.process()
  -> BeliefGenerator.process()
  -> output serialization
```

`generate_sig_pipeline` accepts the legacy scalar interface:

```python
generate_sig_pipeline(
    entity_count=1000,
    fact_count=5000,
    source_count=100,
)
```

It also accepts a topology configuration object:

```python
generate_sig_pipeline(
    source_count=100,
    world_config=WorldConfig(...),
)
```

If `world_config` is supplied, it is passed to `WorldGenerator.generate(config=world_config)`. Otherwise, scalar `entity_count` and `fact_count` are converted into a default `WorldConfig`.

### 2.2 Core Data Structures

`TruthGraph` remains intentionally simple:

- `entities: Dict[str, Entity]`
- `facts: List[Fact]`
- `add_entity(entity)`
- `add_fact(fact)`
- `get_all_facts()`
- `query(**filters)`

This preserves the existing SIG stages. The topology redesign is implemented inside world generation and statistics, not by replacing the graph storage contract.

`Entity` includes:

- `id`
- `name`
- `description`
- `metadata`
- `domain`
- `category`
- `type`

`Fact` includes:

- `id`
- `metadata`
- `subject_id`
- `predicate`
- `object_id`
- `truth_value`

The generator uses `metadata` to store topology annotations such as:

- `community_id`
- `node_role`
- `topology_role`
- `duplicate_semantics`

These metadata fields are additive. Existing consumers can ignore them safely.

---

## 3. World Generation Architecture

### 3.1 `WorldConfig`

The implemented configuration model is:

```python
WorldConfig(
    entities=1000,
    facts=5000,
    community_count=20,
    hub_fraction=0.05,
    bridge_fraction=0.02,
    local_scc_fraction=0.15,
    dag_fraction=0.50,
    hierarchy_fraction=0.10,
    conflict_fraction=0.05,
    random_edge_fraction=0.10,
    average_reasoning_depth=8,
    average_branching_factor=4,
    max_duplicate_edge_ratio=0.15,
    max_largest_scc_fraction=0.35,
    seed=13,
)
```

These values are topology targets, not hard mathematical guarantees. Generation is stochastic, but deterministic for a fixed seed. For non-trivial worlds, the generated graph is validated before it is returned.

### 3.2 Node Roles

The generator assigns node roles through `Entity.metadata["node_role"]`:

- `ordinary_entity`: default entity role.
- `hub_entity`: selected by `hub_fraction`; used to create high-degree retrieval distractors and central concepts.
- `bridge_entity`: selected by `bridge_fraction`; used to connect communities.
- `cycle_entity`: selected by `local_scc_fraction`; reserved for local SCC generation.

Cycle entities are intentionally isolated from most non-SCC edge generation. This prevents local cycles from merging into one giant SCC.

### 3.3 Edge Roles

Topology edges are created internally as `_TopologyEdge` objects before becoming `Fact` objects. Each edge has:

- `subject_id`
- `object_id`
- `predicate`
- `truth_value`
- `role`

Implemented topology roles:

- `dag`: directed causal, temporal, or dependency structure.
- `hierarchy`: membership, subtype, part-whole, or governance structure.
- `local_scc`: bounded feedback loops inside communities.
- `hub`: high-degree central-node structure.
- `bridge`: cross-community links.
- `conflict`: contradictory or weakening pathways.
- `random`: controlled background connectivity.
- `singleton`: fallback for one-entity worlds.

When materialized as `Fact` objects, the edge role is stored in `Fact.metadata["topology_role"]`.

### 3.4 Community Structure

Community sizes are allocated using a log-normal distribution rather than equal partitioning. This creates uneven communities, which better resembles real information domains.

For small worlds, the effective community count is capped so communities are not all singletons:

```text
effective_community_count = min(config.community_count, max(1, config.entities // 3))
```

Every entity receives a `community_id` metadata value.

### 3.5 DAG Regions

DAG edges are generated inside communities. They use predicates such as:

- `affects`
- `enables`
- `precedes`
- `constrains`
- `amplifies`

Non-SCC edges are oriented by entity rank. This creates broad directed flow and prevents accidental global cyclic collapse.

DAG regions exist to support:

- ordered multi-hop reasoning
- dependency tracing
- temporal and causal traversal
- long and short directed paths

### 3.6 Hierarchies

Hierarchy edges are generated within communities using predicates such as:

- `member_of`
- `subtype_of`
- `part_of`
- `governed_by`

These edges produce retrieval tasks that require category/entity reasoning rather than only direct entity-to-entity lookup.

### 3.7 Local SCCs

Local SCCs are generated as bounded cycles inside eligible communities using predicates such as:

- `feeds_back_to`
- `stabilizes`
- `destabilizes`
- `reinforces`

Local SCCs model feedback loops and mutually reinforcing systems. They are desirable when bounded. They are harmful when they merge into a giant SCC. The generator therefore prevents most other edge roles from connecting directly to `cycle_entity` nodes.

### 3.8 Hubs

Hub edges are generated around selected `hub_entity` nodes using predicates such as:

- `influences`
- `coordinates`
- `indexes`
- `broadcasts`

Hubs create high-degree nodes that challenge retrieval systems with plausible distractors. They also model central institutions, platforms, categories, or widely referenced concepts.

### 3.9 Bridges

Bridge edges connect different communities using predicates such as:

- `bridges_to`
- `depends_on`
- `exports_to`
- `references`

Bridges are deliberately sparse. They force cross-community traversal and prevent every task from being solved within one local neighborhood.

### 3.10 Conflict Edges

Conflict edges use predicates such as:

- `contradicts`
- `disputes`
- `weakens`
- `rebuts`

They create competing information pathways. These are important for fact verification, source disagreement, trust ranking, and belief formation.

### 3.11 Controlled Duplicates

The generator first creates a target number of unique topology edges:

```text
target_unique_edges = facts * (1 - max_duplicate_edge_ratio)
```

It then materializes each topology edge as a `Fact`. If additional facts are needed to reach `config.facts`, it creates controlled semantic duplicates using predicates such as:

- `reported_by`
- `observed_in`
- `independently_observed`
- `reconfirmed_by`

These duplicates are explicitly marked:

```python
metadata={
    "topology_role": edge.role,
    "duplicate_semantics": "evidence_restatement",
}
```

This preserves some evidence redundancy without allowing the benchmark to collapse into repeated identical edges.

---

## 4. Graph Generation Algorithm

The implemented generation flow is:

1. Resolve `WorldConfig`.
2. Initialize deterministic random generator from `config.seed`.
3. Generate ontology with domains, categories, and types.
4. Allocate skewed community sizes.
5. Create entities and assign ontology plus `community_id`.
6. Assign node roles: hubs, bridges, cycle entities, ordinary entities.
7. Compute target unique edge count from `facts` and `max_duplicate_edge_ratio`.
8. Allocate edge budget by topology role.
9. Generate local DAG edges.
10. Generate hierarchy edges.
11. Generate local SCC edges.
12. Generate hub edges.
13. Generate bridge edges.
14. Generate conflict edges.
15. Generate controlled random edges.
16. Materialize topology edges into `Fact` objects.
17. Add controlled semantic duplicate facts until `config.facts` is reached.
18. Validate topology.
19. Return `TruthGraph` or raise `ValueError` for unhealthy non-trivial worlds.

The generator tracks seen subject/object pairs and seen triples while constructing topology edges. This avoids accidental duplicate collapse during unique-edge generation.

Non-SCC edge orientation is rank-based. This is a deliberate protection against reconstructing the original problem: a large graph that is technically connected but topologically collapsed into one giant directed cycle.

---

## 5. Benchmark Statistics and Health Validation

`logos/world/stats.py` builds a `networkx.DiGraph` projection from the `TruthGraph`:

```text
Fact(subject_id, predicate, object_id) -> DiGraph edge(subject_id, object_id)
```

This projection intentionally collapses parallel facts between the same subject and object. Therefore the statistics report both fact counts and unique edge counts.

### 5.1 Preserved Metrics

The following legacy-compatible fields are preserved:

- `Entities`
- `Relationships`
- `NumberOfSCCs`
- `LargestSCCSize`
- `AverageInDegree`
- `AverageOutDegree`
- `AverageShortestPathLength`
- `GraphDiameter`
- `LongestAcyclicChain`
- `EstimatedReasoningDepth`
- `BenchmarkHealth`

Important: `Relationships` means unique directed subject/object edges in the `networkx.DiGraph`, not stored fact objects.

### 5.2 Added Metrics

The current implementation adds:

- `Facts`
- `UniqueSubjectObjectEdges`
- `UniqueTriples`
- `DuplicateEdgeRatio`
- `DuplicateTripleRatio`
- `LargestSCCFraction`
- `EffectiveBranchingFactor`
- `PathDiversityP50`
- `PathDiversityP90`
- `BridgeDensity`
- `TopHubDegreeShare`
- `InDegreeEntropy`
- `OutDegreeEntropy`
- `PredicateEntropy`
- `TopologyRoleDistribution`
- `TopologyHealthReasons`

### 5.3 Metric Definitions

`DuplicateEdgeRatio`:

```text
(Facts - UniqueSubjectObjectEdges) / Facts
```

This detects whether high fact volume is merely repeated subject/object pairs.

`DuplicateTripleRatio`:

```text
(Facts - UniqueTriples) / Facts
```

This detects literal triple repetition.

`LargestSCCFraction`:

```text
LargestSCCSize / Entities
```

This detects giant SCC collapse.

`EffectiveBranchingFactor`:

For sampled nodes, the implementation compares frontier growth across limited directed expansion depth. A ring graph has effective branching near 1. Richer retrieval graphs should exceed that.

`PathDiversityP50` and `PathDiversityP90`:

For sampled source-target pairs, the system counts simple directed paths up to a cutoff and reports distribution percentiles. This captures whether the graph offers multiple routes between entities.

`BridgeDensity`:

```text
bridge-role facts / total facts
```

This approximates how much of the fact set contributes to cross-community traversal.

`TopHubDegreeShare`:

```text
sum(degrees of top 1 percent nodes) / sum(all degrees)
```

This measures hub concentration.

`InDegreeEntropy` and `OutDegreeEntropy`:

Entropy over degree distributions. Low entropy indicates structurally repetitive topology.

`PredicateEntropy`:

Entropy over fact predicates. Low entropy indicates semantic monotony.

### 5.4 Health Checks

The system flags unhealthy worlds with `TopologyHealthReasons`.

Current failure modes:

- `empty_graph`
- `no_facts`
- `giant_scc`
- `ring_graph_collapse`
- `duplicate_edge_collapse`
- `low_branching_factor`
- `low_degree_diversity`
- `low_predicate_diversity`
- `insufficient_reasoning_depth`
- `disconnected_graph_collapse`

For worlds with more than 20 entities, the generator rejects unhealthy outputs by raising `ValueError`.

Small worlds are exempt from full topology rejection because unit tests and toy examples often use 1 to 20 entities, where many topology metrics are not meaningful.

---

## 6. Current Default Benchmark Profile

With the default configuration:

```python
WorldGenerator().generate(entity_count=1000, fact_count=5000)
```

the current deterministic seed produces the following smoke-test profile:

```text
Entities: 1000
Facts: 5000
Relationships / UniqueSubjectObjectEdges: 4236
DuplicateEdgeRatio: 0.153
NumberOfSCCs: 224
LargestSCCSize: 82
LargestSCCFraction: 0.082
AverageInDegree: 4.236
AverageOutDegree: 4.236
GraphDiameter: 7
EstimatedReasoningDepth: 42
EffectiveBranchingFactor: 4.406
PathDiversityP50: 8
PathDiversityP90: 25
BridgeDensity: 0.020
OutDegreeEntropy: 3.477
PredicateEntropy: 4.607
BenchmarkHealth: HEALTHY
TopologyHealthReasons: []
```

This is materially different from the previous ring topology. The graph no longer has one 1000-node SCC, no longer has average in-degree/out-degree fixed at 1.0, and no longer collapses 5000 facts into 1000 unique edges.

---

## 7. SIG Layer Behavior

### 7.1 Sources

`SourceGenerator` creates sources with:

- `reliability`
- optional `BiasProfile` entries
- generated names and descriptions

Bias profiles can include:

- `target_predicates`
- `target_entities`
- `skew`

### 7.2 Communities

`CommunityGenerator` groups sources and assigns `community_id` values. This creates social structure in the information layer. It is separate from world-graph communities, which are stored in entity metadata.

The distinction is intentional:

- World communities describe objective topology.
- Source communities describe information-agent grouping.

### 7.3 Observations

`ObservationGenerator` currently iterates over every source and every fact:

```text
observations = source_count * fact_count
```

For the default benchmark:

```text
100 sources * 5000 facts = 500,000 observations
```

Each observation records:

- `source_id`
- `fact_id`
- content containing subject, predicate, object, truth value, and observed value
- uncertainty
- ontology tags
- causal-chain text
- temporal window
- ground-truth quality

Observation uncertainty is influenced by source reliability and bias skew.

### 7.4 Beliefs

`BeliefGenerator` groups observations by:

```text
(source_id, fact_id)
```

It emits one belief per source/fact pair in the current default flow. For the default benchmark, this also yields 500,000 beliefs.

The belief stores:

- `source_id`
- `fact_id`
- `valence`
- `confidence`
- `evidence_ids`
- ontology tags
- causal chain
- temporal window
- ground-truth quality

Current belief formation is source/fact local. It does not yet propagate through graph neighborhoods. The improved topology makes future graph-aware belief propagation meaningful, but the SIG stages themselves remain intentionally preserved.

---

## 8. Retrieval and Reasoning Benchmark Alignment

The redesigned topology supports richer benchmark tasks.

### 8.1 Retrieval Systems

Retrieval difficulty now comes from:

- branching neighborhoods
- high-degree hubs
- sparse bridges
- conflicting facts
- predicate diversity
- multiple directed paths
- community-local and cross-community tasks

This is a better signal than graph diameter alone. A large ring can have a high diameter while remaining structurally trivial.

### 8.2 Graph Traversal Systems

DAG regions test ordered traversal. Local SCCs test bounded cyclic reasoning. Bridges test cross-community routing. Hubs test whether traversal can avoid irrelevant high-degree distractors.

### 8.3 Source Attribution

Controlled duplicate facts represent evidence restatements rather than accidental topology collapse. Because duplicates are marked with `duplicate_semantics`, downstream evaluators can distinguish repeated evidence from unique topology.

### 8.4 Trust Ranking

Trust ranking benefits from:

- conflicting pathways
- different source reliabilities
- repeated observations
- source communities
- bridge-dependent claims

The topology can now produce cases where a claim is locally reinforced but globally contradicted.

### 8.5 Misinformation Spread

Misinformation-like behavior can be modeled through:

- local SCC reinforcement
- hubs broadcasting weak facts
- bridge nodes spreading claims across communities
- conflict predicates creating disagreement paths

This gives future evaluations a realistic substrate for belief propagation and correction.

---

## 9. Output Artifacts

`main.py` writes:

- `output/benchmark_statistics.json`
- `output/graph_audit.md`
- `output/reasoning_examples.json`
- `output/topology_report.md`
- `output/sources.json`
- `output/communities.json`
- `output/observations.json`
- `output/beliefs.json`
- `output/knowledge_base.json`

The SIG output writer intentionally exports only:

- `sources.json`
- `communities.json`
- `observations.json`
- `beliefs.json`

The truth graph itself is used internally for statistics and observation generation but is not currently serialized as a standalone JSON graph artifact.

---

## 10. Testing Strategy

The test suite uses `pytest`.

Important coverage areas:

- `test_world_generator.py`
  - verifies requested entity/fact scale
  - verifies `WorldConfig` support
  - verifies benchmark-scale generation avoids ring collapse
  - checks duplicate-edge ratio, degree entropy, predicate entropy, bridge density, and branching factor

- `test_world_graph.py`
  - verifies `TruthGraph` fact storage and query behavior

- `test_sig_pipeline.py`
  - verifies end-to-end pipeline outputs, including `truth_graph`

- SIG feature and schema tests
  - verify source, community, observation, belief, and serialization contracts

Current verification command:

```text
python -m pytest -q
```

Current result:

```text
26 passed
```

---

## 11. Known Limitations and Future Work

The topology generator is now structurally richer, but several areas remain intentionally evolutionary rather than fully redesigned:

1. `ObservationGenerator` still has full source-by-fact coverage. This creates large observation sets but does not model selective source access.
2. `BeliefGenerator` aggregates by `(source_id, fact_id)` and does not yet perform graph-aware propagation.
3. World communities and source communities are separate. Future work may align or deliberately misalign them for more complex trust experiments.
4. `CommunityModularity` is not yet reported as a first-class metric.
5. Bridge density currently uses fact metadata rather than computed community boundary crossings.
6. Reasoning examples are sampled from the projected `DiGraph`; they do not yet preserve predicate-level path explanations.
7. The generator rejects unhealthy worlds but does not yet retry with adjusted seeds.

These are acceptable next steps because the main benchmark failure has been corrected: the default topology is no longer a directed ring, and scale is no longer confused with structural diversity.

---

## 12. Design Principle

Project Logos should treat generated topology as a benchmark contract. A world is not valid merely because it contains the requested number of entities and facts. It is valid only if it satisfies structural properties that challenge retrieval and reasoning systems:

- meaningful branching
- bounded cycles
- non-trivial communities
- sparse bridges
- predicate diversity
- controlled evidence redundancy
- multiple paths
- detectable conflicts
- measurable health

The code is the source of truth. Documentation should be updated whenever topology generation or benchmark statistics change.
