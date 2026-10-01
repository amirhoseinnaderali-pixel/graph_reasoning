# Current Implementation Audit

The pre-transformation repository contained Python source modules, JSON/GEXF graph artifacts, a NumPy similarity matrix, and PNG visualizations.

## Existing implementation

- **creasoning.py:** calls a list of Google/Ollama model identifiers in parallel; validates structured proposed approaches and thought-node JSON; saves model outputs; and passes prior outputs into later rounds. It is generation/orchestration code, not a benchmark runner.
- **graph.py:** constructs a directed NetworkX graph with problem, solution, and thought nodes; adds hierarchy/dependency edges; computes embeddings (SentenceTransformer if importable, otherwise TF-IDF); adds thresholded top-k cross-solution similarity links; detects Louvain communities; exports GEXF/JSON/figures. Defaults: threshold 0.6, top-k 3, resolution 1.2.
- **reasoning_graph.py:** constructs an undirected graph over chunks, using 300-word chunks, minimum 20 words, threshold 0.5, top-k 5, and Louvain communities. It exports GEXF, JSON, and a cosine similarity matrix. Its compute_embeddings() only fills embeddings under the TF-IDF branch; with SentenceTransformer importable it does not populate them, although downstream code expects embeddings.
- **page_rank.py:** provides SentenceTransformer embedding/chunking/normalization, thresholded answer-similarity graphs, PageRank, and eigenvector centrality. The original main script does not connect the selection to a standardized evaluation record.
- **k_means.py:** uses embeddings, K-Means or optional HDBSCAN, silhouette-based K selection, centroid representatives, and PCA visualization. It is a utility, not an integrated baseline.
- **main.py:** hard-codes a signal-processing problem and provider credentials, executes numbered generation rounds, calls graph/ranking utilities, calls graph_of_thoughts twice, and exposes no reusable CLI experiment runner.
- **ollama.py:** small local Ollama HTTP client with request timeouts and response parsing.

## Historical artifacts

Original reasoning JSON, GEXF exports, PNG visualizations, and the NumPy matrix are preserved under artifacts/historical/. They are not benchmark evidence because the original repository does not link them to a standard evaluation schema and correctness labels.

## Security

Credentials were hard-coded in main.py. The active research framework has no hard-coded credentials and provides .env.example. Any committed keys should be treated as exposed and rotated if still valid.

## Not established

The original snapshot does not establish accuracy gains, benchmark wins, statistical significance, semantic validity of communities, causal value of graph propagation, or novelty relative to published work.
