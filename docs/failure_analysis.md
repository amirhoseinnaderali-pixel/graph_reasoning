# Failure Analysis

## Candidate failures
Poor candidates; insufficient diversity; correlated model errors; unreliable confidence.

## Graph construction failures
Fragmentation; incorrect similarity links; misleading semantic similarity; duplicate reasoning dominating neighborhoods; threshold sensitivity.

## Ranking failures
Centrality selecting popular but incorrect reasoning; community-size bias; unstable rankings on sparse graphs; semantic overlap without logical validity.

## Community failures
Communities unrelated to meaningful reasoning patterns; single-model dominance; instability across seeds or embedding models.

## Cost failures
Expensive embeddings; pairwise similarity cost; graph overhead without measurable benefit.

For each failure, record problem id, selected candidate, correctness signal, alternatives, local neighborhood, scores, community assignment, configuration, and observed evidence. Separate evidence from interpretation.
