# GCR-LLM

## Graph-Structured Collective Reasoning for Multi-Model Language Model Inference

GCR-LLM is a research software framework for testing whether heterogeneous reasoning trajectories produced by multiple language models can be represented as graphs and whether graph structure provides useful signals for selecting or aggregating candidate reasoning.

This repository is a research prototype. It separates historical exploratory code from the reproducible experimental framework and treats empirical claims as hypotheses until evaluated.

## Research question

Can heterogeneous reasoning trajectories produced by multiple language models be represented as a graph, and can graph structure provide useful signals for selecting or aggregating reasoning paths more effectively than independent candidate ranking?

## Hypothesis

H1 (hypothesis, not established): representing multiple model-generated reasoning trajectories as a similarity/dependency graph exposes cross-solution structural information that can improve reasoning-path selection or aggregation compared with independent candidate ranking.

## Method

~~~text
Problem
  |
  v
Candidate reasoning trajectories
  |
  v
Structured reasoning units
  |
  v
Text embeddings
  |
  v
Dependency + similarity graph
  |
  +--> community analysis
  |
  +--> graph centrality
  |
  v
candidate-level score
  |
  v
selected answer
~~~

The framework consumes frozen candidate outputs so generation can be separated from evaluation.

## Baselines

- Single Model
- Independent Multi-Sample
- Embedding Similarity Ranking
- Majority / Consensus
- Proposed GCR-LLM graph ranking

No method is assumed to be better before measurement.

## Reproducibility

~~~bash
pip install -e .
python scripts/run_experiment.py --input data/raw/candidates.json --config configs/graph_reasoning.yaml
python scripts/evaluate.py --results results/raw
python scripts/run_ablation.py --input data/raw/candidates.json --config configs/ablation.yaml
~~~

## Scientific integrity

Historical graphs, JSON files, matrices, and visualizations are preserved as historical artifacts. They are not presented as benchmark evidence unless linked to a reproducible experiment record.

The repository does not claim state-of-the-art performance, superiority, novelty, statistical significance, or accuracy gains without measurement.

## Documentation

- docs/current_implementation_audit.md
- docs/methodology.md
- docs/ablation_plan.md
- docs/graph_metrics.md
- docs/failure_analysis.md
- docs/research_positioning.md
- docs/paper.md

## Repository identity

Target research identity: GCR-LLM
Target repository slug: graph-structured-collective-reasoning

The connected GitHub interface does not expose repository-slug administration, so the slug change itself must be performed through GitHub repository settings.
