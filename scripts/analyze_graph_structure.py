import argparse
import json
import math
from collections import Counter, defaultdict


def entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def analyze(path):
    data = json.loads(open(path, "r", encoding="utf-8").read())
    metadata = data.get("metadata", {})
    solutions = data.get("solutions", {})
    thoughts = data.get("thoughts", [])

    print(f"file: {path}")
    print(f"nodes: {metadata.get('total_nodes')}")
    print(f"edges: {metadata.get('total_edges')}")
    print(f"communities: {metadata.get('num_communities')}")

    if not thoughts:
        chunks = data.get("chunks", [])
        by_community = defaultdict(list)
        for chunk in chunks:
            by_community[chunk.get("community", -1)].append(
                str(chunk.get("solution_id"))
            )
        print("community_rows:", len(by_community))
        for community, members in sorted(by_community.items()):
            print(
                community,
                "size=", len(members),
                "unique_sources=", len(set(members)),
                "entropy_bits=", round(entropy(members), 4),
            )
        return

    by_community = defaultdict(list)
    for thought in thoughts:
        by_community[thought.get("community", -1)].append(
            thought.get("solution_key")
        )

    mixed = 0
    for community, members in sorted(by_community.items()):
        unique_sources = len(set(members))
        mixed += unique_sources > 1
        print(
            community,
            "size=", len(members),
            "unique_sources=", unique_sources,
            "entropy_bits=", round(entropy(members), 4),
        )

    print("mixed_communities:", mixed)
    print("total_communities:", len(by_community))
    print(
        "mixed_community_fraction:",
        round(mixed / len(by_community), 4) if by_community else 0.0,
    )

    model_names = [
        solutions.get(key, {}).get("model", "unknown")
        for key in set(members for members in [t.get("solution_key") for t in thoughts])
    ]
    if model_names:
        print("models_present:", sorted(set(model_names)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    analyze(parser.parse_args().input)
