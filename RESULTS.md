# Results

No causal performance result is claimed in this branch yet.

## Historical structural results

The existing artifacts report 17 generated solutions, 212 thought nodes, 1,339 total edges, and 10 communities in `got_data.json`.

The improved artifact reports 45 nodes, 67 edges, 22 chunks, 23 similarity edges, 7 communities, and mean similarity edge weight ≈ 0.777.

These values describe the graph, not task success.

## Planned evaluation

| Condition | Description |
|---|---|
| Single | Use one candidate solution |
| Ensemble | Aggregate candidates without graph structure |
| Graph | Use graph communities/similarity to select and synthesize information |

Primary metric: objective task accuracy.

Secondary metrics:
- number of candidate solutions used
- graph construction time
- synthesis calls
- answer latency
- community stability

A result should be considered evidence only when the benchmark, verifier, sampling protocol, and graph configuration are fixed in advance.
