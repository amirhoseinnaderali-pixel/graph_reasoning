# Research Audit — Graph Reasoning

## What the original project actually demonstrates

The repository takes multiple LLM-generated structured solution traces, converts thought nodes and solution chunks into a graph, connects semantically similar chunks, and detects communities.

The checked-in artifact `got_data.json` contains:
- 17 generated solutions
- 212 thought nodes
- 1,339 edges
- 229 hierarchy edges
- 268 dependency edges
- 842 similarity edges
- 10 detected communities

The later `improved_data.json` contains:
- 45 graph nodes
- 67 edges
- 22 chunks
- 3 solution/strategy sources
- 7 communities
- mean similarity edge weight about 0.777

These are structural measurements, not evidence that graph reasoning improves task accuracy.

## Research question

> Can graph-based aggregation identify recurring reasoning strategies across independently generated solution traces, and does graph-guided synthesis improve objective task performance over selecting a single solution?

## Hypothesis

> Cross-solution graph structure will reveal reusable reasoning patterns, and using those patterns for synthesis will improve verified task performance compared with a single-trace baseline.

This is falsifiable.

## Main methodological gaps

1. No objective downstream task metric is reported.
2. Community detection is descriptive; no intervention is tested.
3. Similarity edges depend on embedding model, threshold, and top-k choices.
4. The graph mixes hierarchy/dependency relations with semantic similarity, making causal interpretation difficult.
5. The corpus contains multiple model families and repeated model variants without a controlled sampling protocol.
6. Community counts and similarity values are not accompanied by sensitivity analysis.
7. There is no held-out task set showing that graph-guided synthesis improves correctness.
8. The current pipeline assumes that the generated structured traces are faithful representations of reasoning.
9. The project currently treats graph visualization as an end result rather than an intermediate representation for an evaluated decision procedure.

## Scientific redesign

Separate the work into two stages.

### Stage A — representation study

Measure:
- cross-model community mixing
- community size distribution
- redundancy
- stability under threshold/top-k changes
- similarity-edge statistics

### Stage B — intervention study

For a fixed benchmark:
- Single best/first solution baseline
- Majority/ensemble baseline
- Graph-guided synthesis

Evaluate with an objective verifier.

The graph is useful only if its structure changes a downstream decision and improves a measurable outcome.

## Limitations

The current checked-in results should therefore be described as a **graph-construction and exploratory analysis artifact**, not as evidence that graph reasoning improves LLM reasoning.

## Next experiment

Move from the current mathematical/signal-processing corpus to a small executable benchmark. Store candidate answers separately from graph metadata and use an independent verifier for correctness.
