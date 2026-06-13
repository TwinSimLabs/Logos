from collections import Counter


def build_world_summary(data):

    stats = data["benchmark_statistics"]

    beliefs = data["beliefs"]
    observations = data["observations"]

    predicates = Counter()

    for belief in beliefs[:50000]:
        tags = belief.get("ontology_tags", [])

        for tag in tags:
            if tag.startswith("predicate:"):
                predicates[tag.replace("predicate:", "")] += 1

    return {
        "entities": stats["Entities"],
        "facts": stats["Facts"],
        "sources": len(data["sources"]),
        "communities": len(data["communities"]),
        "reasoning_depth": stats["EstimatedReasoningDepth"],
        "predicate_distribution": dict(
            predicates.most_common(20)
        ),
    }