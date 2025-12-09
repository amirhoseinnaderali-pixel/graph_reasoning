# 🧠 Multi-Model Creative Reasoning & Graph Analysis Pipeline

A sophisticated system for tackling complex mathematical/engineering problems using multiple AI models, graph-based reasoning, and ensemble ranking methods.

## 📋 Overview

This project implements a multi-stage pipeline that:
1. **Generates creative solutions** using multiple AI models (Google Gemini, DeepSeek, etc.)
2. **Tracks reasoning as graphs** (Chain of Thought → Graph of Thoughts)
3. **Ranks and clusters solutions** using PageRank, K-Means, and similarity analysis
4. **Visualizes reasoning paths** through interactive graph visualizations

## 🏗️ Architecture

```
Problem → Multi-Model Reasoning → Graph Construction → Ranking → Best Solution
          (Chain of Thought)      (GoT + CoT graphs)   (PageRank/K-Means)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -U sentence-transformers torch numpy scikit-learn networkx matplotlib hdbscan google-generativeai
```

### Basic Usage

```python
from creasoning import creasoning
from reasoning_graph import reasoning_graph
from graph import graph_of_thoughts
from page_rank import page_rank
from k_means import k_means

# Define your problem
problem = """
Your complex mathematical/engineering problem here...
"""

# Configure API keys
api_keys_google = [
    "YOUR_GOOGLE_API_KEY_1",
    "YOUR_GOOGLE_API_KEY_2",
    # Add more for load balancing
]
api_key_ollama = "ollama"  # For local models

# Step 1: Generate solutions iteratively
output_dir = "data"
reasoning_1 = creasoning(problem, api_keys_google[0], api_key_ollama, 
                         "creasoning_1.json", [], output_dir)
reasoning_2 = creasoning(problem, api_keys_google[1], api_key_ollama, 
                         "creasoning_2.json", reasoning_1, output_dir)
# ... continue for more iterations

# Step 2: Build reasoning graphs
reasoning_graph()      # Chunk-based similarity graph
graph_of_thoughts()    # Merge all Chain-of-Thought trees

# Step 3: Rank solutions
page_rank()            # Graph-based ranking
k_means()              # Clustering-based ranking
```

## 📦 Core Components

### 1. `creasoning.py` - Creative Reasoning Engine

Orchestrates parallel calls to 30+ AI models to generate diverse solution approaches.

**Key Features:**
- **Chain of Thought (CoT)** tracking with dependency graphs
- Parallel API calls with timeout handling
- Structured JSON validation for reasoning steps
- Iterative refinement using previous models' outputs

**Output Structure:**
```json
{
  "problem_restatement": "...",
  "proposed_approaches": [...],
  "detailed_strategy": {...},
  "chain_of_thought": [
    {
      "thought_id": "T1",
      "thought_type": "problem_analysis",
      "content": "...",
      "dependencies": [],
      "confidence": 0.9
    }
  ]
}
```

### 2. `graph.py` - Graph of Thoughts (GoT)

Merges multiple Chain-of-Thought trees into a unified directed graph.

**Features:**
- Node types: `problem`, `solution`, `thought`
- Edge types: `hierarchy`, `dependency`, `similarity`
- Cross-solution thought similarity detection
- Louvain community detection
- Exports: GEXF (Gephi), JSON metadata, PNG visualizations

**Visualization Outputs:**
- `got_graph_hierarchical.png` - Tree structure view
- `got_graph_spring.png` - Similarity-based layout
- `got_graph.gexf` - Interactive graph for Gephi/Cytoscape

### 3. `reasoning_graph.py` - Chunk-Based Reasoning Graph

Alternative graph construction using text chunking and similarity.

**Features:**
- Splits solutions into semantic chunks (300 words)
- Undirected graph with sparse k-NN connections
- Community detection across solution chunks
- TF-IDF or Transformer embeddings

**Use Case:** Better for long-form solutions where thought-level granularity isn't available.

### 4. `page_rank.py` - Graph-Based Ranking

Ranks solutions using network centrality algorithms.

**Methods:**
- **PageRank**: Damped random walk centrality
- **Eigenvector Centrality**: Principal eigenvector of adjacency matrix

**How It Works:**
1. Embed all solutions using `mixedbread-ai/mxbai-embed-large-v1`
2. Build similarity graph (cosine similarity > threshold)
3. Compute centrality scores
4. Rank by importance in the network

### 5. `k_means.py` - Clustering-Based Ranking

Ranks solutions by finding cluster representatives.

**Methods:**
- **K-Means**: Automatic optimal K selection via silhouette score
- **HDBSCAN**: Density-based clustering (finds K automatically)

**Ranking Strategies:**
- `size`: Larger clusters = more consensus
- `coherence`: Tighter clusters = more specific approaches

**Output:** 2D PCA visualization with cluster representatives highlighted

## 📊 Output Files

| File | Description |
|------|-------------|
| `creasoning_*.json` | Raw solution data from models |
| `got_graph.gexf` | Graph of Thoughts (import to Gephi) |
| `got_data.json` | Full metadata with communities |
| `got_graph_*.png` | Visual graph layouts |
| `improved_graph.gexf` | Chunk-based reasoning graph |
| `clusters_visualization.png` | K-Means clustering results |

## 🔬 Advanced Features

### Iterative Reasoning

Each reasoning round uses previous models' outputs:
```python
reasoning_2 = creasoning(problem, api_key, "output_2.json", 
                         other_model_thinking=reasoning_1, ...)
```

This enables:
- Building on previous insights
- Avoiding redundant approaches
- Converging toward better solutions

### Community Detection

Identifies clusters of related thoughts across solutions:
```python
communities = got.detect_communities()
# Output: Community 0 contains 12 thoughts from 5 solutions
```

### Embedding Models

Default: `mixedbread-ai/mxbai-embed-large-v1` (1024-dim, sota for reasoning)

Change in code:
```python
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Alternative
```

### Graph Layouts

- **Hierarchical**: Best for seeing solution structure
- **Spring**: Best for seeing cross-solution connections
- **Kamada-Kawai**: Balanced aesthetic layout

## 📈 Evaluation Workflow

1. **Generate Diverse Solutions**: Run 3-5 reasoning rounds
2. **Build Graphs**: Visualize reasoning structure
3. **Rank Solutions**:
   - PageRank → consensus in reasoning network
   - K-Means → representative approaches
4. **Manual Review**: Inspect top 3-5 solutions from each method

## ⚙️ Configuration

### Graph of Thoughts
```python
config = GoTConfig(
    similarity_threshold=0.6,   # Min similarity for connections
    top_k_neighbors=3,          # How many similar thoughts to connect
    community_resolution=1.2    # Granularity of communities
)
```

### Chunk-Based Graph
```python
config = GraphConfig(
    chunk_size=300,             # Words per chunk
    similarity_threshold=0.5,   # Min similarity
    top_k_neighbors=5           # Neighbors to connect
)
```

### PageRank
```python
results = ranker.rank_answers(
    answers,
    method="pagerank",
    threshold=0.3,              # Graph edge threshold
    damping=0.85                # PageRank damping factor
)
```

### K-Means
```python
results = ranker.rank_answers(
    answers,
    method="kmeans",
    auto_k=True,                # Auto-detect optimal clusters
    rank_by="size"              # "size" or "coherence"
)
```

## 🎯 Use Cases

- **Research Problems**: Open mathematical conjectures
- **Engineering Design**: Multi-constraint optimization
- **Algorithm Development**: Novel approach discovery
- **Proof Strategies**: Finding attack vectors for theorems

## 🐛 Troubleshooting

**Q: API timeout errors?**  
A: Increase timeout in `creasoning.py` or reduce model count.

**Q: Out of memory?**  
A: Reduce `BATCH_SIZE` in embedding code or use CPU instead of GPU.

**Q: Empty graphs?**  
A: Check that JSON files contain `chain_of_thought` fields. Enable debug logging.

**Q: Poor rankings?**  
A: Adjust similarity thresholds (lower = more connections). Try different ranking methods.

## 📚 Citation

If using this for research:
```bibtex
@software{multi_model_reasoning_2025,
  title={Multi-Model Creative Reasoning Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/project}
}
```

## 📝 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional ranking algorithms (HITS, Katz centrality)
- Real-time visualization dashboards
- LLM-as-judge for solution quality
- Automated proof verification integration

---

**Built with:** Google Gemini, DeepSeek, Sentence-Transformers, NetworkX, scikit-learn