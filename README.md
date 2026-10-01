# Graph Reasoning

This project studies whether multiple LLM-generated reasoning traces can be represented as a graph and then used as a structured intermediate representation for synthesis.

## Research question

> Can graph-based aggregation identify reusable reasoning strategies across independently generated solution traces, and does graph-guided synthesis improve objective task performance over a non-graph baseline?

## Original system

The pipeline:

`LLM solutions → structured thought nodes → semantic similarity graph → communities → visualization/export`

The checked-in artifacts contain substantial graph structure. For example, `got_data.json` contains 17 solutions, 212 thought nodes, 1,339 edges, and 10 communities.

Those are **descriptive structural results only**.

## Research hardening

This branch separates two questions:

### 1. Is the graph structure stable and informative?

We measure:
- community size
- source mixing
- entropy
- sensitivity to graph hyperparameters

### 2. Does graph structure improve decisions?

A future controlled benchmark will compare:
- single-solution selection
- non-graph ensemble
- graph-guided synthesis

with an independent objective verifier.

## Run structural analysis

```bash
pip install -r requirements.txt

python scripts/analyze_graph_structure.py --input got_data.json
python scripts/analyze_graph_structure.py --input improved_data.json
```

No model API calls are needed for this analysis.

## Current status

**Implemented**
- multi-model structured trace collection
- graph construction
- semantic similarity edges
- community detection
- graph export/visualization

**Executed**
- historical artifacts are checked into the repository

**Measured**
- graph nodes/edges
- similarity statistics
- community structure

**Not yet demonstrated**
- improved objective task accuracy from graph-guided reasoning

## Limitations

The current corpus mixes multiple model families and is generated from a mathematical/signal-processing problem. Graph structure can therefore be confounded with model identity, prompt effects, and embedding choices.

The project should not claim that it improves reasoning performance until the graph is used in an intervention with a fixed benchmark and objective verifier.

## Connection to the broader research trajectory

This project extends the test-time reasoning line:

`iterative prompting → multi-stage agents → graph-structured reasoning`

The core research theme remains efficient allocation of inference-time compute: use multiple reasoning attempts, compress them into a structured representation, then spend additional computation only where it is useful.
