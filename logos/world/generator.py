import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logos.core.base import BaseGenerator
from logos.world.graph import TruthGraph
from logos.world.schemas import Entity, Fact


@dataclass(frozen=True)
class WorldConfig:
    """Topology targets for synthetic world generation."""

    entities: int = 1000
    facts: int = 5000
    community_count: int = 20
    hub_fraction: float = 0.05
    bridge_fraction: float = 0.02
    local_scc_fraction: float = 0.15
    dag_fraction: float = 0.50
    hierarchy_fraction: float = 0.10
    conflict_fraction: float = 0.05
    random_edge_fraction: float = 0.10
    average_reasoning_depth: int = 8
    average_branching_factor: int = 4
    max_duplicate_edge_ratio: float = 0.15
    max_largest_scc_fraction: float = 0.35
    seed: int = 13


@dataclass(frozen=True)
class _TopologyEdge:
    subject_id: str
    object_id: str
    predicate: str
    truth_value: float
    role: str


class WorldGenerator(BaseGenerator[TruthGraph]):
    """Generate a topology-rich truth graph for retrieval and reasoning benchmarks."""

    def _generate_ontology(
        self,
        rng: random.Random,
        domains_count: int = 15,
    ) -> Dict[str, Dict[str, List[str]]]:
        ontology = {}
        for d in range(domains_count):
            domain_name = f"Domain_{d + 1}"
            ontology[domain_name] = {}
            for c in range(rng.randint(5, 10)):
                cat_name = f"{domain_name}_Cat_{c + 1}"
                ontology[domain_name][cat_name] = [
                    f"{cat_name}_Type_{t + 1}"
                    for t in range(rng.randint(4, 5))
                ]
        return ontology

    def generate(self, *args: Any, **kwargs: Any) -> TruthGraph:
        config = self._resolve_config(kwargs)
        rng = random.Random(config.seed)

        ontology = self._generate_ontology(rng)
        graph = TruthGraph()

        type_list = [
            (domain, category, entity_type)
            for domain, categories in ontology.items()
            for category, types in categories.items()
            for entity_type in types
        ]

        community_count = max(
            1,
            min(config.community_count, max(1, config.entities // 3)),
        )
        community_sizes = self._allocate_community_sizes(
            config.entities,
            community_count,
            rng,
        )

        communities: List[List[Entity]] = []
        entity_index = 0
        for community_index, size in enumerate(community_sizes):
            community_entities = []
            for _ in range(size):
                domain, category, entity_type = rng.choice(type_list)
                entity = Entity(
                    name=f"Entity_{entity_index + 1}",
                    description=f"Synthetic entity {entity_index + 1}",
                    domain=domain,
                    category=category,
                    type=entity_type,
                    metadata={
                        "community_id": f"community_{community_index + 1}",
                        "node_role": "ordinary_entity",
                    },
                )
                graph.add_entity(entity)
                community_entities.append(entity)
                entity_index += 1
            communities.append(community_entities)

        entities = list(graph.entities.values())
        if not entities:
            return graph

        self._assign_roles(entities, config, rng)

        target_unique_edges = self._target_unique_edges(config)
        edges: List[_TopologyEdge] = []
        seen_pairs: set[Tuple[str, str]] = set()
        seen_triples: set[Tuple[str, str, str]] = set()

        edge_budget = self._edge_budget(config, target_unique_edges)

        self._generate_local_dags(
            communities,
            config,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["dag"],
        )
        self._generate_hierarchies(
            communities,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["hierarchy"],
        )
        self._generate_local_sccs(
            communities,
            config,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["scc"],
        )
        self._generate_hubs(
            entities,
            config,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["hub"],
        )
        self._generate_bridges(
            communities,
            config,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["bridge"],
        )
        self._generate_conflicts(
            entities,
            config,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            edge_budget["conflict"],
        )
        self._generate_random_edges(
            entities,
            rng,
            edges,
            seen_pairs,
            seen_triples,
            target_unique_edges - len(edges),
        )

        if not edges and len(entities) == 1:
            subject = entities[0]
            edges.append(
                _TopologyEdge(
                    subject.id,
                    subject.id,
                    "self_observed",
                    0.5,
                    "singleton",
                )
            )

        self._materialize_facts(graph, edges, config, rng)
        self._reject_unhealthy_world(graph, config)
        return graph

    def _resolve_config(self, kwargs: Dict[str, Any]) -> WorldConfig:
        supplied_config: Optional[WorldConfig] = kwargs.get("config")
        if supplied_config is not None:
            return supplied_config

        return WorldConfig(
            entities=int(kwargs.get("entity_count", kwargs.get("entities", 1000))),
            facts=int(kwargs.get("fact_count", kwargs.get("facts", 5000))),
            community_count=int(kwargs.get("community_count", 20)),
            hub_fraction=float(kwargs.get("hub_fraction", 0.05)),
            bridge_fraction=float(kwargs.get("bridge_fraction", 0.02)),
            local_scc_fraction=float(kwargs.get("local_scc_fraction", 0.15)),
            dag_fraction=float(kwargs.get("dag_fraction", 0.50)),
            hierarchy_fraction=float(kwargs.get("hierarchy_fraction", 0.10)),
            conflict_fraction=float(kwargs.get("conflict_fraction", 0.05)),
            random_edge_fraction=float(kwargs.get("random_edge_fraction", 0.10)),
            average_reasoning_depth=int(kwargs.get("average_reasoning_depth", 8)),
            average_branching_factor=int(kwargs.get("average_branching_factor", 4)),
            max_duplicate_edge_ratio=float(kwargs.get("max_duplicate_edge_ratio", 0.15)),
            max_largest_scc_fraction=float(kwargs.get("max_largest_scc_fraction", 0.35)),
            seed=int(kwargs.get("seed", 13)),
        )

    def _allocate_community_sizes(
        self,
        entity_count: int,
        community_count: int,
        rng: random.Random,
    ) -> List[int]:
        if entity_count <= 0:
            return []

        weights = [rng.lognormvariate(0.0, 0.65) for _ in range(community_count)]
        total_weight = sum(weights)
        sizes = [
            max(1, int(round(entity_count * weight / total_weight)))
            for weight in weights
        ]

        while sum(sizes) > entity_count:
            index = max(range(len(sizes)), key=lambda item: sizes[item])
            sizes[index] -= 1

        while sum(sizes) < entity_count:
            index = rng.randrange(len(sizes))
            sizes[index] += 1

        return sizes

    def _assign_roles(
        self,
        entities: Sequence[Entity],
        config: WorldConfig,
        rng: random.Random,
    ) -> None:
        shuffled = list(entities)
        rng.shuffle(shuffled)

        hub_count = max(1, int(len(entities) * config.hub_fraction))
        bridge_count = max(1, int(len(entities) * config.bridge_fraction))
        scc_count = max(1, int(len(entities) * config.local_scc_fraction))

        role_slices = [
            ("hub_entity", shuffled[:hub_count]),
            ("bridge_entity", shuffled[hub_count:hub_count + bridge_count]),
            (
                "cycle_entity",
                shuffled[hub_count + bridge_count:hub_count + bridge_count + scc_count],
            ),
        ]

        for role, role_entities in role_slices:
            for entity in role_entities:
                entity.metadata["node_role"] = role

    def _target_unique_edges(self, config: WorldConfig) -> int:
        if config.entities <= 1:
            return min(config.facts, 1)

        possible_without_self_loops = config.entities * (config.entities - 1)
        desired = int(config.facts * (1.0 - config.max_duplicate_edge_ratio))
        return max(
            1,
            min(config.facts, possible_without_self_loops, desired),
        )

    def _edge_budget(
        self,
        config: WorldConfig,
        target_unique_edges: int,
    ) -> Dict[str, int]:
        fractions = {
            "dag": config.dag_fraction,
            "hierarchy": config.hierarchy_fraction,
            "scc": config.local_scc_fraction,
            "bridge": config.bridge_fraction,
            "hub": config.hub_fraction,
            "conflict": config.conflict_fraction,
            "random": config.random_edge_fraction,
        }
        fraction_total = sum(max(0.0, value) for value in fractions.values()) or 1.0
        budgets = {
            key: int(target_unique_edges * max(0.0, value) / fraction_total)
            for key, value in fractions.items()
        }
        budgets["random"] += target_unique_edges - sum(budgets.values())
        return budgets

    def _add_edge(
        self,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        subject: Entity,
        object_entity: Entity,
        predicate: str,
        truth_value: float,
        role: str,
        allow_self_loop: bool = False,
    ) -> bool:
        if (
            role not in {"local_scc", "singleton"}
            and (
                subject.metadata.get("node_role") == "cycle_entity"
                or object_entity.metadata.get("node_role") == "cycle_entity"
            )
        ):
            return False

        if role not in {"local_scc", "singleton"}:
            subject, object_entity = self._orient_by_rank(subject, object_entity)

        if subject.id == object_entity.id and not allow_self_loop:
            return False

        pair = (subject.id, object_entity.id)
        triple = (subject.id, predicate, object_entity.id)
        if pair in seen_pairs or triple in seen_triples:
            return False

        seen_pairs.add(pair)
        seen_triples.add(triple)
        edges.append(
            _TopologyEdge(
                subject.id,
                object_entity.id,
                predicate,
                truth_value,
                role,
            )
        )
        return True

    def _orient_by_rank(
        self,
        subject: Entity,
        object_entity: Entity,
    ) -> Tuple[Entity, Entity]:
        if self._entity_rank(subject) <= self._entity_rank(object_entity):
            return subject, object_entity
        return object_entity, subject

    def _entity_rank(self, entity: Entity) -> int:
        try:
            return int(entity.name.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            return abs(hash(entity.id))

    def _generate_local_dags(
        self,
        communities: Sequence[Sequence[Entity]],
        config: WorldConfig,
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        predicates = ["affects", "enables", "precedes", "constrains", "amplifies"]
        attempts = 0
        while budget > 0 and attempts < budget * 30:
            attempts += 1
            community = rng.choice([list(item) for item in communities if len(item) > 1])
            ordered = sorted(community, key=lambda entity: entity.name)
            source_index = rng.randrange(0, len(ordered) - 1)
            max_jump = max(1, min(config.average_reasoning_depth, len(ordered) - source_index - 1))
            jump = rng.randint(1, max_jump)
            subject = ordered[source_index]
            object_entity = ordered[source_index + jump]
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                subject,
                object_entity,
                rng.choice(predicates),
                round(rng.uniform(0.55, 0.95), 3),
                "dag",
            ):
                budget -= 1

    def _generate_hierarchies(
        self,
        communities: Sequence[Sequence[Entity]],
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        predicates = ["member_of", "subtype_of", "part_of", "governed_by"]
        attempts = 0
        while budget > 0 and attempts < budget * 30:
            attempts += 1
            community = rng.choice([list(item) for item in communities if len(item) > 1])
            parent = rng.choice(community)
            child = rng.choice(community)
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                child,
                parent,
                rng.choice(predicates),
                round(rng.uniform(0.65, 0.98), 3),
                "hierarchy",
            ):
                budget -= 1

    def _generate_local_sccs(
        self,
        communities: Sequence[Sequence[Entity]],
        config: WorldConfig,
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        predicates = ["feeds_back_to", "stabilizes", "destabilizes", "reinforces"]
        eligible = [
            list(community)
            for community in communities
            if len(community) >= 3
        ]
        attempts = 0
        while budget > 0 and eligible and attempts < budget * 30:
            attempts += 1
            community = rng.choice(eligible)
            cycle_size = min(len(community), rng.randint(3, 8))
            cycle = rng.sample(community, cycle_size)
            for index, subject in enumerate(cycle):
                if budget <= 0:
                    break
                object_entity = cycle[(index + 1) % len(cycle)]
                if self._add_edge(
                    edges,
                    seen_pairs,
                    seen_triples,
                    subject,
                    object_entity,
                    rng.choice(predicates),
                    round(rng.uniform(0.45, 0.90), 3),
                    "local_scc",
                ):
                    budget -= 1

    def _generate_hubs(
        self,
        entities: Sequence[Entity],
        config: WorldConfig,
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        hubs = [
            entity
            for entity in entities
            if entity.metadata.get("node_role") == "hub_entity"
        ] or list(entities[:1])
        predicates = ["influences", "coordinates", "indexes", "broadcasts"]
        attempts = 0
        while budget > 0 and attempts < budget * 40:
            attempts += 1
            hub = rng.choice(hubs)
            other = rng.choice(list(entities))
            subject, object_entity = (
                (hub, other)
                if rng.random() < 0.65
                else (other, hub)
            )
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                subject,
                object_entity,
                rng.choice(predicates),
                round(rng.uniform(0.35, 0.88), 3),
                "hub",
            ):
                budget -= 1

    def _generate_bridges(
        self,
        communities: Sequence[Sequence[Entity]],
        config: WorldConfig,
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        if len(communities) < 2:
            return

        bridge_nodes = [
            entity
            for community in communities
            for entity in community
            if entity.metadata.get("node_role") == "bridge_entity"
        ]
        predicates = ["bridges_to", "depends_on", "exports_to", "references"]
        attempts = 0
        while budget > 0 and attempts < budget * 50:
            attempts += 1
            left_index = rng.randrange(len(communities))
            right_index = rng.randrange(len(communities))
            if left_index == right_index:
                continue

            left = list(communities[left_index])
            right = list(communities[right_index])
            subject = rng.choice(bridge_nodes) if bridge_nodes and rng.random() < 0.5 else rng.choice(left)
            object_entity = rng.choice(right)
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                subject,
                object_entity,
                rng.choice(predicates),
                round(rng.uniform(0.40, 0.92), 3),
                "bridge",
            ):
                budget -= 1

    def _generate_conflicts(
        self,
        entities: Sequence[Entity],
        config: WorldConfig,
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        predicates = ["contradicts", "disputes", "weakens", "rebuts"]
        attempts = 0
        while budget > 0 and attempts < budget * 40:
            attempts += 1
            subject = rng.choice(list(entities))
            object_entity = rng.choice(list(entities))
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                subject,
                object_entity,
                rng.choice(predicates),
                round(rng.uniform(0.10, 0.45), 3),
                "conflict",
            ):
                budget -= 1

    def _generate_random_edges(
        self,
        entities: Sequence[Entity],
        rng: random.Random,
        edges: List[_TopologyEdge],
        seen_pairs: set[Tuple[str, str]],
        seen_triples: set[Tuple[str, str, str]],
        budget: int,
    ) -> None:
        predicates = ["related_to", "correlates_with", "observed_near", "co_occurs_with"]
        attempts = 0
        while budget > 0 and attempts < max(100, budget * 60):
            attempts += 1
            subject = rng.choice(list(entities))
            object_entity = rng.choice(list(entities))
            if self._add_edge(
                edges,
                seen_pairs,
                seen_triples,
                subject,
                object_entity,
                rng.choice(predicates),
                round(rng.uniform(0.25, 0.85), 3),
                "random",
            ):
                budget -= 1

    def _materialize_facts(
        self,
        graph: TruthGraph,
        edges: Sequence[_TopologyEdge],
        config: WorldConfig,
        rng: random.Random,
    ) -> None:
        if not edges:
            return

        for edge in edges[:config.facts]:
            graph.add_fact(
                Fact(
                    subject_id=edge.subject_id,
                    predicate=edge.predicate,
                    object_id=edge.object_id,
                    truth_value=edge.truth_value,
                    metadata={"topology_role": edge.role},
                )
            )

        duplicate_predicates = [
            "reported_by",
            "observed_in",
            "independently_observed",
            "reconfirmed_by",
        ]
        while len(graph.facts) < config.facts:
            edge = rng.choice(edges)
            graph.add_fact(
                Fact(
                    subject_id=edge.subject_id,
                    predicate=rng.choice(duplicate_predicates),
                    object_id=edge.object_id,
                    truth_value=max(
                        0.0,
                        min(
                            1.0,
                            round(edge.truth_value + rng.uniform(-0.08, 0.08), 3),
                        ),
                    ),
                    metadata={
                        "topology_role": edge.role,
                        "duplicate_semantics": "evidence_restatement",
                    },
                )
            )

    def _reject_unhealthy_world(
        self,
        graph: TruthGraph,
        config: WorldConfig,
    ) -> None:
        if config.entities <= 20:
            return

        from logos.world.stats import calculate_benchmark_statistics

        stats = calculate_benchmark_statistics(graph)
        health_reasons = stats.get("TopologyHealthReasons", [])
        if health_reasons:
            raise ValueError(
                "Generated world failed topology validation: "
                + ", ".join(health_reasons)
            )
