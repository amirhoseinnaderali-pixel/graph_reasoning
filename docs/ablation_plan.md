# Ablation Plan

The main question is whether graph structure adds value beyond direct candidate ranking.

| Variant | Similarity | Dependency | Community | Centrality | Multi-model |
|---|---|---|---|---|---|
| A1 Direct ranking | yes | no | no | no | yes |
| A2 No community | yes | yes | no | yes | yes |
| A3 No centrality | yes | yes | yes | no | yes |
| A4 No cross-model restriction | configured | yes | yes | yes | yes |
| A5 Dependency-only | no | yes | yes | yes | yes |
| A6 Similarity-only | yes | no | yes | yes | yes |
| A7 TF-IDF | yes | yes | yes | yes | yes |
| A8 High threshold | yes | yes | yes | yes | yes |
| A9 Low threshold | yes | yes | yes | yes | yes |
| A10 Larger top-k | yes | yes | yes | yes | yes |

Freeze candidate sets and correctness signals across variants. Change one factor at a time where possible, record complete configuration, and repeat across multiple problems. Do not add results until experiments have actually run.

Model-diversity study: vary same-model samples, heterogeneous model families, number of models, and number of trajectories. Record graph density, components, communities, trajectory similarity, and model diversity per community. Whether graph utility changes with diversity is a hypothesis requiring validation.
